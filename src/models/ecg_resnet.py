from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange

from src.utils.helper import delete_invalid_kwargs, _ovewrite_named_param

def conv3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(
            self, 
            in_planes, 
            planes, 
            stride=1, 
            downsample=None, 
            groups: int = 1,
            base_width: int = 64,
            norm_layer=nn.BatchNorm1d
        ):
        super(BasicBlock1d, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        self.conv1 = conv3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        
    def forward(self, x: Tensor):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity) 

        x += identity
        x = self.relu(x)

        return x


class Bottleneck1d(nn.Module):
    expansion = 4

    def __init__(
        self, 
        in_planes, 
        planes, 
        stride=1, 
        downsample=None, 
        groups: int = 1,
        base_width: int = 64,
        norm_layer=nn.BatchNorm1d
    ):
        super(Bottleneck1d, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x: Tensor):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity) 

        x += identity
        x = self.relu(x)
        
        return x
        
        
class ResNet1d(nn.Module):
    def __init__(
            self, 
            block, 
            layers: List[int], 
            in_chans=12, 
            groups: int = 1,
            width_per_group: int = 64,
            num_classes=1, 
            norm_layer=nn.BatchNorm1d
        ):
        super(ResNet1d, self).__init__()
        self.in_planes = 64
        self.in_chans = in_chans
        self.norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group

        #self.conv1 = nn.Conv1d(in_chans, self.in_planes, kernel_size=49, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv1d(in_chans, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.flatten = nn.Flatten(1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(
            self, 
            block: Type[Union[BasicBlock1d, Bottleneck1d]], 
            planes: int, 
            num_blocks: int, 
            stride: int
        ) -> nn.Sequential:
        norm_layer = self.norm_layer
        layers = []
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1(self.in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers.append(
            block(self.in_planes, planes, stride, downsample, self.groups, self.base_width,  norm_layer)
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, groups=self.groups, base_width=self.base_width, norm_layer=norm_layer)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x) #torch.flatten(x, 1) More flexible as we can set that to nn.Identity()
        x = self.fc(x)

        return x


def resnet1d18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    return ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)

def resnet1d34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    return ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)

def resnet1d50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    return ResNet1d(Bottleneck1d, [3, 4, 6, 3], **kwargs)

def resnet1d101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    return ResNet1d(Bottleneck1d, [3, 4, 23, 3], **kwargs)

def resnet1d152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    return ResNet1d(Bottleneck1d, [3, 8, 36, 3], **kwargs)

def wide_resnet1d50(**kwargs):
    """Constructs a WideResNet-50 model.
    """
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return ResNet1d(Bottleneck1d, [3, 4, 6, 3], **kwargs)

def wide_resnet1d101(**kwargs):
    """Constructs a WideResNet-101 model.
    """
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return ResNet1d(Bottleneck1d, [3, 4, 23, 3], **kwargs)

def wide_resnet1d152(**kwargs):
    """Constructs a WideResNet-152 model.
    """
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return ResNet1d(Bottleneck1d, [3, 8, 36, 3], **kwargs)



class ECGResnet(nn.Module):
    def __init__(self, resnet_fn, patch_embedding, **kwargs) -> None:
        super().__init__()
        self.patch = patch_embedding
        kwargs = delete_invalid_kwargs(kwargs, valid_kwargs=['in_chans'])
        resnet = resnet_fn(**kwargs)
        resnet.fc = nn.Identity() # We remove the fully connected classifier
        self.resnet = resnet
        
        
    def forward(self, x):
        x = self.patch(x)
        x = self.resnet(x)
        return x


def ecgresnet_model(model_name: str, **kwargs):
    prepare_kwargs(model_name, kwargs)
    
    if '1d18' in model_name: return ECGResnet(resnet1d18, **kwargs)
    if '1d34' in model_name: return ECGResnet(resnet1d34, **kwargs)
    if '1d50_wide' in model_name: return ECGResnet(wide_resnet1d50, **kwargs)
    if '1d50' in model_name: return ECGResnet(resnet1d50, **kwargs)
    if '1d101_wide' in model_name: return ECGResnet(wide_resnet1d101, **kwargs)
    if '1d101' in model_name: return ECGResnet(resnet1d101, **kwargs)
    if '1d152_wide' in model_name: return ECGResnet(wide_resnet1d152, **kwargs)
    if '1d152' in model_name: return ECGResnet(resnet1d152, **kwargs)
    else: 
        exts = [18, 34, 50, 101, 152]
        raise RuntimeError(f"model name should be ecgresnet_[{''.join(exts, '|')}]_[|wide], not {model_name}")


def prepare_kwargs(model_name: str, kwargs):
    if 'embed=flat' in model_name or kwargs.get('in_chans', None) == 1:
        kwargs['patch_embedding'] = Rearrange('B C L -> B 1 (C L)') 
        kwargs['in_chans'] = 1
    else:
        kwargs['patch_embedding'] = nn.Identity() 
        kwargs['in_chans'] = 12