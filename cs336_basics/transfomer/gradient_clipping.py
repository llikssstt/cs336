import torch
import torch.nn as nn
from typing import Iterable
import math

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6) -> None:

    total_norm_sq = sum(p.grad.data.pow(2).sum().item() for p in parameters if p.grad is not None)
    l2_norm = math.sqrt(total_norm_sq)  

    if l2_norm >= max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= max_l2_norm / (l2_norm + eps)
