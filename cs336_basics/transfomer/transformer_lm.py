import torch
import torch.nn as nn
from einops import einsum, rearrange
from .transfomer_block import TransfomerBlock
from .embedding import Embedding
from .rmsnorm import RMSNorm
from .linear import Linear
from .softmax import Softmax
class Transformer_LM(nn.Module):
    def __init__(self,vocab_size: int, context_length: int, d_model: int,num_layers: int, num_heads: int, d_ff: int,rope_theta: float):

        super(Transformer_LM, self).__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.layers = nn.ModuleList([
            TransfomerBlock(self.d_model, self.num_heads, self.d_ff, self.context_length, self.rope_theta)
            for _ in range(self.num_layers)
        ])        
        self.token_embeddings = Embedding(self.vocab_size, self.d_model)
        self.ln_final = RMSNorm(self.d_model)
        self.lm_head = Linear(self.vocab_size, self.d_model)
        self.softmax = Softmax(dimension=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        