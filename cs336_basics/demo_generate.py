#!/usr/bin/env python3
"""
Demo script to generate text from a trained checkpoint.

Usage:
    python -m cs336_basics.demo_generate \
        --checkpoint ./checkpoints/checkpoint_5000.pt \
        --tokenizer ./tinystories_bpe.pkl \
        --prompt "Once upon a time" \
        --max_tokens 256 \
        --temperature 0.8 \
        --top_p 0.9
"""

import argparse
import pickle
import torch

from cs336_basics.transfomer.transformer_lm import Transformer_LM
from cs336_basics.transfomer.generate import generate, decode_text
from cs336_basics.tokenizer.BPE_Tokenizer import BPE_Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    
    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer pickle file")
    
    # Model architecture (must match training)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Text prompt to complete")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling threshold")
    parser.add_argument("--eos_token", type=str, default="<|endoftext|>",
                        help="End of sequence token")
    
    # Device
    parser.add_argument("--device", type=str, 
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    with open(args.tokenizer, "rb") as f:
        tok_data = pickle.load(f)
    
    tokenizer = BPE_Tokenizer(
        vocab=tok_data["vocab"],
        merges=tok_data["merges"],
        special_tokens=[args.eos_token] if args.eos_token else None,
    )
    print(f"Tokenizer loaded. Vocab size: {len(tok_data['vocab'])}")
    
    # Create model
    print("Creating model...")
    model = Transformer_LM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    
    iteration = checkpoint.get("iteration", "unknown")
    print(f"Checkpoint loaded (iteration {iteration})")
    
    # Get EOS token ID
    eos_token_id = None
    if args.eos_token:
        try:
            eos_tokens = tokenizer.encode(args.eos_token)
            if len(eos_tokens) == 1:
                eos_token_id = eos_tokens[0]
                print(f"EOS token ID: {eos_token_id}")
        except:
            print(f"Warning: Could not find EOS token '{args.eos_token}'")
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens[:20]}{'...' if len(prompt_tokens) > 20 else ''}")
    
    # Generate
    print(f"\nGenerating with temperature={args.temperature}, top_p={args.top_p}...")
    print("-" * 60)
    
    output_tokens = generate(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        device=args.device,
    )
    
    # Decode and print
    generated_text = tokenizer.decode(output_tokens)
    
    print("\n" + "=" * 60)
    print("GENERATED TEXT:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)
    
    # Stats
    new_tokens = len(output_tokens) - len(prompt_tokens)
    print(f"\nStats:")
    print(f"  Prompt tokens: {len(prompt_tokens)}")
    print(f"  Generated tokens: {new_tokens}")
    print(f"  Total tokens: {len(output_tokens)}")
    
    if eos_token_id is not None and output_tokens[-1] == eos_token_id:
        print(f"  Stopped at: EOS token")
    else:
        print(f"  Stopped at: max tokens")


if __name__ == "__main__":
    main()
