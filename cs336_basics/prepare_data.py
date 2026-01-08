#!/usr/bin/env python3
"""
Data preparation script for training.
Converts text files to tokenized .npy format using a saved BPE tokenizer.

Usage:
    python -m cs336_basics.prepare_data \
        --tokenizer tinystories_bpe.pkl \
        --input data/tinystories.txt \
        --output data/train.npy
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer.BPE_Tokenizer import BPE_Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to BPE tokenizer pickle file (e.g., tinystories_bpe.pkl)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input text file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output .npy file")
    parser.add_argument("--special_tokens", type=str, nargs="*", 
                        default=["<|endoftext|"])
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    with open(args.tokenizer, "rb") as f:
        data = pickle.load(f)
    
    tokenizer = BPE_Tokenizer(
        vocab=data["vocab"],
        merges=data["merges"],
        special_tokens=args.special_tokens,
    )
    vocab_size = len(data["vocab"])
    print(f"Tokenizer loaded. Vocab size: {vocab_size}")
    
    # Read input text
    print(f"Reading input file {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Input file size: {len(text):,} characters")
    
    # Tokenize
    print("Tokenizing... (this may take a while)")
    token_ids = tokenizer.encode(text)
    print(f"Tokenization complete. Total tokens: {len(token_ids):,}")
    
    # Save as numpy array
    print(f"Saving to {args.output}...")
    arr = np.array(token_ids, dtype=np.uint16)
    np.save(args.output, arr)
    print(f"Saved {len(arr):,} tokens to {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
