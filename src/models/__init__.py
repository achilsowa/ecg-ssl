from src.models.vision_transformer import vit_model
from src.models.ecg_transformer import ecgt_model
from src.models.text_transformer import textt_model
from src.models.ecg_resnet import ecgresnet_model
from src.models.resnet import resnet_model
from src.models.attentive_pooler import cls_attn_model
from src.models.linear_projection import cls_proj_model
from src.models.adater import adapter_model

def get_model(model_name:str, **kwargs):
    if model_name.startswith("text"):
        return textt_model(model_name, **kwargs)
    
    if model_name.startswith("vit"):
        return vit_model(model_name, **kwargs)
    
    if model_name.startswith("resnet"):
        return resnet_model(model_name, **kwargs)
    
    if model_name.startswith("ecgresnet"):
        return ecgresnet_model(model_name, **kwargs)
    
    if model_name.startswith("ecgt"):
        return ecgt_model(model_name, **kwargs)
    
    if model_name.startswith("cls_attn"):
        return cls_attn_model(model_name, **kwargs)
    
    if model_name.startswith("cls_proj"):
        return cls_proj_model(model_name, **kwargs)
    
    if model_name.startswith("adapter"):
        return adapter_model(model_name, **kwargs)
    
    else: 
        raise RuntimeError(f"model name should start with  [resnet|ecgresnet|ecgt|vit|cls|adapter], not {model_name}")