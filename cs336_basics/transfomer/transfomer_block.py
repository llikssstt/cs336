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

        self.attn = MHA(self.d_model, self.num_heads, self.max_seq_len, self.theta)
        self.ln1 = RMSNorm(self.d_model)
        self.ln2 = RMSNorm(self.d_model)

        self.ffn = SwiGLU(self.d_model, self.d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        y1 = x + self.attn(self.ln1(x), token_positions=token_positions)
        y2 = y1 + self.ffn(self.ln2(y1))
        return y2


