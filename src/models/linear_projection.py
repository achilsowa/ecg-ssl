import torch.nn as nn

from src.utils.helper import delete_invalid_kwargs


def get_layers(dims, use_bn=False, use_ln=False):
    layers = [nn.Flatten()]
    for i in range(len(dims) - 1):
        if i != len(dims) - 2:
            if use_ln:
                 layers.append(nn.LayerNorm(dims[i]))
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_bn:
                 layers.append(nn.BatchNorm1d(dims[i+1]))
            if use_ln:
                 layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(nn.ReLU())

        else: #last layer
            if use_ln:
                 layers.append(nn.LayerNorm(dims[i]))
            layers.append(nn.Linear(dims[i], dims[i+1], bias=not use_bn))
            if use_bn: 
                layers.append(nn.BatchNorm1d(dims[i+1]))
            if use_ln:
                 layers.append(nn.LayerNorm(dims[i+1]))
            

    return layers

def get_projection(dims, use_bn=False, use_ln=False):
    layers = get_layers(dims, use_bn, use_ln)
    projection = nn.Sequential(*layers,)
    return projection

def get_simclr_projection(dims):
    layers = get_layers(dims, use_bn=True)
    projection = nn.Sequential(*layers, )
    return projection

def get_byol_projection(dims):
    layers = get_layers(dims, use_bn=True)
    layers = layers[:-1] # remove the last batch norm layer



def cls_proj_model(model_name, **kwargs):
    prepare_kwargs(model_name, kwargs)
    if 'simclr' in model_name:
        model = get_simclr_projection(kwargs['dims'])
    elif 'byol' in model_name:
        model = get_byol_projection(kwargs['dims'])
    else:
        model = get_projection(**kwargs)
    return model


def prepare_kwargs(model_name: str, kwargs):
    """
    Example of model_name format: cls_proj_dims=512,14_bn
    """
    for key in model_name.split("_"):
        if key == "ln":
            kwargs["use_ln"] = True
        if key == "bn":
            kwargs["use_bn"] = True
        if key.startswith("dims"):
            _, dims = key.split("=")
            dims = [int(d) for d in dims.split(",")]
            if 'num_classes' in kwargs:
                dims += [kwargs['num_classes']]
            kwargs['dims'] = dims

    kwargs = delete_invalid_kwargs(kwargs, valid_kwargs=['dims', 'use_bn', 'use_ln'])
    return kwargs

