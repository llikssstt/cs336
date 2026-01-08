import torch.nn as nn
import torch
from einops import rearrange, einsum
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) :
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype = dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3*std,
            b=3*std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量，形状为 (..., d_in)
               可以是二维 (batch, d_in)，也可以带有更多前置维度

        返回:
            输出张量，形状为 (..., d_out)
        """
        x = einsum(x,self.weight, '... in_features, out_features in_features -> ... out_features')
        return x