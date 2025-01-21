# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import torch
import  torch.nn.functional as F
import random

import src.transforms.timeseries as transforms

_GLOBAL_SEED = 0
logger = getLogger()



def make_transforms(
    crop_size=1250,
    crop_scale=(0.3, 1.0),
    random_crop=True,
    random_resized_crop=False,
    gaussian_blur=0.5,
    gaussian_noise=False,
    sobel_derivative=False,
    reverse=False,
    invert=False,
    rand_wanderer=False,
    baseline_wanderer=False,
    baseline_shift=False,
    em_noise=False,
    pl_noise=False,
    time_out=False,
    scale=False,
    normalization=None,
    **kwargs
):
    logger.info('making ecgnet data transforms')
    transform_list = []
    if random_crop:
        transform_list += [transforms.RandomCrop(crop_size,)]
    elif random_resized_crop:
        transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if gaussian_blur:
        transform_list += [transforms.RandomApply(transforms.GaussianBlur(), p=gaussian_blur)]
    if gaussian_noise:
        transform_list += [transforms.RandomApply(transforms.GaussianNoise(), p=gaussian_noise)]
    if sobel_derivative:
        transform_list += [transforms.RandomApply(transforms.SobelDerivative(), p=sobel_derivative)]
    if reverse:
        transform_list += [transforms.RandomApply(transforms.Reverse(), p=reverse)]
    if invert:
        transform_list += [transforms.RandomApply(transforms.Invert(), p=invert)]
    if rand_wanderer:
        transform_list += [transforms.RandomApply(transforms.RandWanderer(), p=rand_wanderer)]
    if baseline_wanderer:
        transform_list += [transforms.RandomApply(transforms.BaselineWander(), p=baseline_wanderer)]
    if baseline_shift:
        transform_list += [transforms.RandomApply(transforms.BaselineShift(), p=baseline_shift)]
    if em_noise:
        transform_list += [transforms.RandomApply(transforms.EMNoise(), p=em_noise)]
    if pl_noise:
        transform_list += [transforms.RandomApply(transforms.PowerlineNoise(), p=pl_noise)]
    if time_out:
        transform_list += [transforms.RandomApply(transforms.TimeOut(), p=time_out)]
    if scale:
        transform_list += [transforms.RandomApply(transforms.ChannelResize(), p=scale)]
    if normalization is not None:
        if len(normalization[0]) == 1:
            mean, std = normalization[0][0], normalization[1][0]
            normalization = [[mean for _ in range(12)], [std for _ in range(12)]]
        transform_list += [transforms.Normalize(normalization[0], normalization[1])]
    
    transform = transforms.Compose(transform_list)
    return transform


def make_eval_transforms(normalization=[[0,], [1,]]):
    logger.info('making ecgnet eval data transforms')
    transform_list = []
    if len(normalization[0]) == 1:
        mean, std = normalization[0][0], normalization[1][0]
        normalization = [[mean for _ in range(12)], [std for _ in range(12)]]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]
    transform = transforms.Compose(transform_list)
    return transform

