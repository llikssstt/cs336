import torch
import torch.nn as nn
from einops import einsum, rearrange
from .linear import Linear
from .rope import RoPE
from .scaled_dot_product_attention import scaled_dot_product_attention

class MHA(nn.Module):
    """
    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len=2048, theta=10000.0):
        super(MHA, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.d_k = d_model // num_heads
        self.q_w = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_w = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_w = Linear(self.d_model, self.num_heads * self.d_k)
        self.rope = RoPE(theta, self.d_k, max_seq_len)
        self.combine = Linear(self.num_heads * self.d_k, self.d_model)

    def forward(self, in_features, token_positions=None):
        batch, seq_len, dim = in_features.shape

        q = self.q_w(in_features)
        k = self.k_w(in_features)
        v = self.v_w(in_features)

        q = rearrange(q, '... (h d) -> ... h d', h=self.num_heads)
        k = rearrange(k, '... (h d) -> ... h d', h=self.num_heads)
        v = rearrange(v, '... (h d) -> ... h d', h=self.num_heads)
        

        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')
        
        # Only apply RoPE when token_positions is explicitly provided
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        

        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))

        score = scaled_dot_product_attention(q, k, v, mask)
        score = rearrange(score, 'b h s d -> b s (h d)')
        out = self.combine(score)

        return out


        






