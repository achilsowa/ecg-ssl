import torch.nn as nn

from einops.layers.torch import Rearrange

from src.utils.helper import delete_invalid_kwargs

def get_adapter(encoder, classifier, in_dim, out_dim, is_resnet=False, use_conv=False,):
    if not is_resnet: return None
    encoder.resnet.avgpool = encoder.resnet.fc = encoder.resnet.flatten = nn.Identity()
    return nn.Sequential(
        Rearrange('B C N -> B N C'), 
        nn.Linear(in_dim, out_dim, bias=False), # try using nn.Conv1d instead of nn.Linear
    )

def adapter_model(model_name: str, encoder, classifier, **kwargs):
    prepare_kwargs(model_name, kwargs)
    adapter = get_adapter(encoder=encoder, classifier=classifier, **kwargs)
    return adapter

def prepare_kwargs(model_name: str, kwargs):
    """
    Example of model_name format: adapter_resnet=2048_cls=768_conv
    """
    for key in model_name.split("_"):
        if key == "conv":
            kwargs["use_conv"] = True
        if key.startswith("resnet"):
            _, embed_dim = key.split("=")
            kwargs["in_dim"] = int(embed_dim)
            kwargs["is_resnet"] = True
        if key.startswith("cls"):
            _, out_dim = key.split("=")
            kwargs["out_dim"] = int(out_dim)
    kwargs = delete_invalid_kwargs(kwargs, valid_kwargs=['in_dim', 'out_dim', 'is_resnet', 'use_conv'])
    return kwargs
