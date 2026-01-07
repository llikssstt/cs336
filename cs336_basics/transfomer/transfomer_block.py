import torch
import torch.nn as nn
from einops import einsum, rearrange
from .multi_head_self_attention import MHA
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
class TransfomerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super(TransfomerBlock, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.mha = MHA(self.d_model, self.num_heads, self.max_seq_len, self.theta)
        self.norm1 = RMSNorm(self.d_model)
        self.norm2 = RMSNorm(self.d_model)

        self.swiglu = SwiGLU(self.d_model, self.d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        y1 = x + self.mha(self.norm1(x), token_positions=token_positions)
        y2 = y1 + self.swiglu(self.norm2(y1))
        return y2


