from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
import torchvision

from einops.layers.torch import Rearrange

from src.utils.helper import delete_invalid_kwargs



class ECGResnet(nn.Module):
    def __init__(self, resnet_fn, patch_embedding, **kwargs) -> None:
        super().__init__()
        self.patch = patch_embedding
        kwargs = delete_invalid_kwargs(kwargs, valid_kwargs=['num_classes'])
        resnet = resnet_fn(**kwargs)
        resnet.fc = nn.Identity() # We remove the fully connected classifier
        self.resnet = resnet
        
        
    def forward(self, x):
        x = self.patch(x)
        x = self.resnet(x)
        return x


def resnet_model(model_name: str, **kwargs):
    prepare_kwargs(model_name, kwargs)
    
    if '1d18' in model_name: return ECGResnet(torchvision.models.resnet18, **kwargs)
    if '1d34' in model_name: return ECGResnet(torchvision.models.resnet34, **kwargs)
    if '1d50' in model_name: return ECGResnet(torchvision.models.resnet50, **kwargs)
    if '1d101' in model_name: return ECGResnet(torchvision.models.resnet101, **kwargs)
    if '1d152' in model_name: return ECGResnet(torchvision.models.resnet152, **kwargs)
    else: 
        exts = [18, 34, 50, 101, 152]
        raise RuntimeError(f"model name should be resnet_[{''.join(exts, '|')}]_[|deep|deeper], not {model_name}")




def prepare_kwargs(model_name: str, kwargs):
    kwargs['patch_embedding'] = nn.Identity() 
    