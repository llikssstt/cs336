import torch
import torch.nn as nn
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RoPE, self).__init__()

        if theta is None:
            theta = 10000.0
        self.theta = float(theta)
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        half = d_k // 2
        inv_freq = self.theta ** (-2 * torch.arange(half, dtype=torch.float32, device=device) / d_k)

        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)

        freqs = positions[:, None] * inv_freq[None, :]

        # cos/sin cache: (max_seq_len, half)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # register as buffers so they move with .to(device) and appear in state_dict if needed
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        cos = cos.to(dtype=x.dtype, device=x.device)
        sin = sin.to(dtype=x.dtype, device=x.device)

        x0 = x[..., ::2]
        x1 = x[..., 1::2]

        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos  

        y = torch.empty_like(x)

        y[..., ::2] = y0
        y[..., 1::2] = y1
        return y