import torch.nn as nn
import torch
from einops import rearrange, einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None) -> None:
    
    # class torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, 
    # scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)
        super(Embedding,self).__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim

        self.weight = nn.Parameter(torch.empty((self.vocab_size, self.d_model), dtype=dtype))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]