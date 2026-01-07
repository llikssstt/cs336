import torch
import torch.nn as nn
from einops import einsum
from .softmax import Softmax
import math

def scaled_dot_product_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask=None) -> torch.Tensor:
    d_k = k.shape[-1]
    score = einsum(q, k, '... n dk,... m dk -> ... n m') / math.sqrt(d_k)
    a = score.masked_fill(mask==False, float('-inf'))
    b = Softmax(a, dimension=-1)
    out = einsum(b, v, '... n m, ... m dv -> ... n dv')
    return out
