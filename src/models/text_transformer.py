from collections import namedtuple, UserDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from functools import partial

from src.models.ecg_transformer import ECGTransformer
from src.utils.beamsearch import  BeamSearchScorer
from src.models.utils import get_1d_sincos_pos_embed
from src.utils.tensors import trunc_normal_

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

LARGE_NUMBER = 1e9


class TextEncoder(nn.Module):
    """"""
    def __init__(
        self, 
        vocab_size,
        vocab_idx,
        dmodel=768,
        depth=12,
        nhead=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        word_embedding=nn.Embedding,
        max_len=1000,
        pooling_method='max',
        cls_token=False,        
        ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.vocab_pad_idx = vocab_idx['<pad>']
        self.vocab_eos_idx = vocab_idx['</s>']
        self.vocab_bos_idx = vocab_idx['<s>']

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dmodel, 
            nhead=nhead,
            activation="gelu",
            dim_feedforward=int(mlp_ratio*dmodel),
            norm_first=True, 
            batch_first=True
        )
        # self.pos_embed = PositionalEmbedding(dmodel)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dmodel), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], max_len,cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.word_embed = word_embedding(vocab_size, dmodel, self.vocab_pad_idx )
        self.pooling_method = pooling_method
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            depth, 
            norm=norm_layer(dmodel),
            enable_nested_tensor=False, # because norm_first = True
        )

    def forward(self, x):
        """
        @param x(torch.LongTensor of shape (B, N)) x tensor consisting of indices of words. Ground truth values
        @param tgs(List[List[str]]) list of target sentence tokens, wrapped by `<s>` and `</s>`, and additionnal `<pad>` if needed:
                ground truth values 
        @return (torch.FloatTensor of shape (B, N, dmodel) or (B, dmodel)) depending on the pooling method"""
        
        x = self.word_embed(x)
        x = x + self.pos_embed[:, :x.size(1)]

        x = self.encoder(x)
        if self.pooling_method == 'cls':
            x = x[:, 0]
        elif self.pooling_method == 'mean':
            x = torch.mean(x, dim=1)
        elif self.pooling_method == 'max':
            x, _ = torch.max(x, dim=1)
        return x


def textt_tiny(**kwargs):
    model = TextEncoder(dmodel=192, depth=12, nhead=3, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def textt_xtiny(**kwargs):
    model = TextEncoder(dmodel=192, depth=4, nhead=3, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def textt_model(model_name: str, vocab_size, vocab_idx, **kwargs):
    """Constructs a Transformer encoder for text.
    @param: model_name (str): model name.
    @param: vocab_size: int
    @param: vocab_idx: Dict[str, int] containing the '<pad>', '<s>', '</s>', '</unk>' idx
    @param: **kwargs: Additional arguments to be passed to the Transformer encoder constructor.
    @return:
        ECGTransformer: A 1D Transformer encoder model for processing ECG signals.
    """

    if model_name == 'textt_tiny': return textt_tiny(vocab_size=vocab_size, vocab_idx=vocab_idx, **kwargs)
    elif model_name == 'textt_xtiny': return textt_xtiny(vocab_size=vocab_size, vocab_idx=vocab_idx, **kwargs)
    else: raise RuntimeError("model name should be textt_[large|base|small|tiny], not {}".format(model_name))
