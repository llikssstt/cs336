import torch.nn as nn
import torch
from einops import einsum, rearrange
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype)).to(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x**2, dim=-1, keepdim=True)
        RMS = torch.sqrt(avg + self.eps)
        x = x / RMS
        x = x * self.weight
        return x