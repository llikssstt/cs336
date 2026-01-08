#!/usr/bin/env python3
"""
Text generation module for Transformer Language Model.

Supports:
- Prompt completion (generate until <endoftext> or max tokens)
- Temperature scaling
- Top-p (nucleus) sampling
"""

import torch
import torch.nn.functional as F
from typing import Optional


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_tokens: list[int],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    device: str = "cuda",
) -> list[int]:
    """
    Generate text completions from a language model.

    Args:
        model: The Transformer language model.
        prompt_tokens: List of token IDs for the prompt.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Temperature for softmax scaling. Lower = more deterministic.
        top_p: Top-p (nucleus) sampling threshold. 1.0 = no filtering.
        eos_token_id: End-of-sequence token ID. Generation stops when this is produced.
        device: Device to run generation on.

    Returns:
        List of generated token IDs (including the prompt).
    """
    model.eval()
    
    # Initialize with prompt tokens
    tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        # Get model predictions
        # Only use the last context_length tokens if sequence is too long
        context_length = getattr(model, 'context_length', tokens.shape[1])
        input_tokens = tokens[:, -context_length:]
        
        logits = model(input_tokens)  # (batch=1, seq_len, vocab_size)
        
        # Get logits for the last position
        next_token_logits = logits[:, -1, :]  # (batch=1, vocab_size)
        
        # Apply temperature scaling
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Convert to probabilities
        probs = F.softmax(next_token_logits, dim=-1)  # (batch=1, vocab_size)
        
        # Apply top-p (nucleus) sampling
        if top_p < 1.0:
            probs = top_p_filtering(probs, top_p)
        
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)  # (batch=1, 1)
        
        # Append to sequence
        tokens = torch.cat([tokens, next_token], dim=1)
        
        # Check for end-of-sequence
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    
    return tokens[0].tolist()


def top_p_filtering(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) filtering to a probability distribution.

    Args:
        probs: Probability distribution tensor of shape (batch, vocab_size)
        top_p: Cumulative probability threshold

    Returns:
        Filtered probability distribution (renormalized)
    """
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find the cutoff index: smallest set where cumsum >= top_p
    # We keep tokens where cumulative_probs <= top_p (shifted by 1 position)
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Shift to keep at least one token
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False
    
    # Zero out filtered tokens
    sorted_probs[sorted_indices_to_remove] = 0.0
    
    # Scatter back to original order
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(1, sorted_indices, sorted_probs)
    
    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    return filtered_probs


def decode_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token: str = "<|endoftext|>",
    device: str = "cuda",
) -> str:
    """
    High-level function to generate text from a string prompt.

    Args:
        model: The Transformer language model.
        tokenizer: Tokenizer with encode/decode methods.
        prompt: Text prompt to complete.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Temperature for softmax scaling.
        top_p: Top-p sampling threshold.
        eos_token: End-of-sequence token string.
        device: Device to run generation on.

    Returns:
        Generated text (including the prompt).
    """
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    
    # Get EOS token ID
    eos_token_id = None
    if eos_token:
        try:
            eos_tokens = tokenizer.encode(eos_token)
            if len(eos_tokens) == 1:
                eos_token_id = eos_tokens[0]
        except:
            pass
    
    # Generate
    output_tokens = generate(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        device=device,
    )
    
    # Decode
    return tokenizer.decode(output_tokens)
