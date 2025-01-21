# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle
import yaml
import logging

import pydash as py_

from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#W = TypeVar("W", bound=WeightsEnum)
#M = TypeVar("M", bound=nn.Module)
V = TypeVar("V")

def init_local_config(local_config, **kwargs):
        configs = local_config if isinstance(local_config, list) else [local_config]
        for config in configs:    
            for key, val in kwargs.items():
                config[key] = config.get(key, val)
        return local_config


def get_norm_from_pkl(src_path):
    with open(src_path, 'rb') as file:
        scaler = pickle.load(file)
    normalization = [
        [m for m in scaler.mean_],
        [s for s in scaler.scale_]
    ]
    return normalization


def delete_invalid_kwargs(kwargs, valid_kwargs):
    invalid_keys = [key for key, _ in kwargs.items() if key not in valid_kwargs]
    for key in invalid_keys:
        del kwargs[key]
    return kwargs 


def clean_state_dict(model, state_dict):
    def check_parallel(state_dict):
        is_parallel = False
        for k, v in state_dict.items():
            if k.startswith('module.') or k.startswith('backbone.'):
                is_parallel = True
                break
        return is_parallel
    
    parallel_model = check_parallel(model.state_dict())
    parallel_dict = check_parallel(state_dict)

    # -- need to prepend `module.` and `backbone.` on the keys
    if parallel_model and not parallel_dict:
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        #state_dict = {f'backbone.{k}': v for k, v in state_dict.items()}
        
    # -- need to remove `module.` and `backbone.` on the keys
    if not parallel_model and parallel_dict:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}

    for k, v in model.state_dict().items():
        if k not in state_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif state_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            state_dict[k] = v
    return state_dict
    


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value


def load_params(fname:str, update = None):
    def load_value(sval: str):
        sval = sval.strip()
        if sval.endswith('.yaml'):
            with open(sval, 'r') as y_file:
                value = yaml.load(y_file, Loader=yaml.FullLoader)
            return value
        else:
            try:
                return eval(sval)
            except Exception:
                return sval
    
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
    
    if update is None:
        return params
    else:
        for elt in update.split('__'):
            key, sval = elt.strip().split('=')
            sval = load_value(sval)
            current = py_.get(params, key, {})
            if isinstance(sval, dict) and isinstance(current, dict):
                py_.merge(current, sval)
                py_.set_(params, key, current)
            else: 
                py_.set_(params, key, sval)
        print(params)
        return (params)
