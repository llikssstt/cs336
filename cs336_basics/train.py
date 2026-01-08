#!/usr/bin/env python3
"""
Training script for Transformer Language Model.

Features:
- Configurable model and optimizer hyperparameters via argparse
- Memory-efficient data loading with np.memmap
- Checkpoint serialization with resume support
- W&B logging for training metrics
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

# Local imports
from cs336_basics.transfomer.transformer_lm import Transformer_LM
from cs336_basics.transfomer.AdamW import AdamW
from cs336_basics.transfomer.cosine import learning_rate_schedule
from cs336_basics.transfomer.gradient_clipping import gradient_clipping
from cs336_basics.transfomer.Cross_entropy import cross_entropy
from cs336_basics.data_loader.data_loading import data_loading
from cs336_basics.data_loader.checkpointing import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    
    # Data paths
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data (.npy memmap file)")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation data (.npy memmap file)")
    
    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=50257,
                        help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256,
                        help="Maximum context length")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0,
                        help="RoPE theta parameter")
    
    # Optimizer hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5,
                        help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95,
                        help="AdamW beta2")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="AdamW epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--max_iters", type=int, default=10000,
                        help="Maximum training iterations")
    parser.add_argument("--warmup_iters", type=int, default=1000,
                        help="Number of warmup iterations")
    
    # Logging and checkpointing
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N iterations")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Evaluate every N iterations")
    parser.add_argument("--eval_iters", type=int, default=100,
                        help="Number of iterations for evaluation")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # W&B configuration
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (disabled if not set)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    
    return parser.parse_args()


def load_data_memmap(path: str) -> np.ndarray:
    """Load data using memory-mapped file for efficiency."""
    return np.memmap(path, dtype=np.uint16, mode='r')


@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, eval_iters):
    """Evaluate model on validation data."""
    model.eval()
    losses = []
    vocab_size = model.vocab_size
    
    for _ in range(eval_iters):
        inputs, targets = data_loading(data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    
    # Device setup
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading training data from {args.train_data}...")
    train_data = load_data_memmap(args.train_data)
    print(f"Training data size: {len(train_data):,} tokens")
    
    val_data = None
    if args.val_data:
        print(f"Loading validation data from {args.val_data}...")
        val_data = load_data_memmap(args.val_data)
        print(f"Validation data size: {len(val_data):,} tokens")
    
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
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    
    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed at iteration {start_iter}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize W&B if configured
    wandb = None
    if args.wandb_project:
        try:
            import wandb as wb
            wandb = wb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
            print(f"W&B initialized: {args.wandb_project}")
        except ImportError:
            print("Warning: wandb not installed, logging disabled")
    
    # Training loop
    print(f"\nStarting training from iteration {start_iter} to {args.max_iters}")
    print("-" * 60)
    
    model.train()
    t0 = time.time()
    
    for iteration in range(start_iter, args.max_iters):
        # Update learning rate
        lr = learning_rate_schedule(
            iteration,
            args.lr,
            args.min_lr,
            args.warmup_iters,
            args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch
        inputs, targets = data_loading(train_data, args.batch_size, args.context_length, device)
        
        # Forward pass
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, args.vocab_size), targets.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()
        
        # Logging
        if iteration % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tokens_per_sec = args.batch_size * args.context_length * args.log_interval / dt if dt > 0 else 0
            
            print(f"iter {iteration:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {tokens_per_sec:.0f} tok/s")
            
            if wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                }, step=iteration)
        
        # Validation
        if val_data is not None and iteration % args.eval_interval == 0 and iteration > 0:
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
            print(f"iter {iteration:6d} | val_loss {val_loss:.4f}")
            
            if wandb:
                wandb.log({"val/loss": val_loss}, step=iteration)
        
        # Checkpointing
        if iteration % args.checkpoint_interval == 0 and iteration > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Final checkpoint
    final_checkpoint = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint)
    print(f"Saved final checkpoint to {final_checkpoint}")
    
    if wandb:
        wandb.finish()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
