import time
import tracemalloc
import pickle
import sys
import os

# Add parent directory to sys.path to allow importing from cs336 package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from BPE_Tokenizer_trainer import BPE_Tokenizer_Trainer

def train_tinystories():
    input_path = r"d:\cs336\data\TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    print(f"Initializing BPE Trainer with input: {input_path}")
    print(f"Target Vocab Size: {vocab_size}")
    
    trainer = BPE_Tokenizer_Trainer(input_path, vocab_size, special_tokens)
    
    # Start measuring time and memory
    tracemalloc.start()
    start_time = time.time()
    
    try:
        vocab, merges = trainer.train()
    finally:
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
    duration = end_time - start_time
    peak_mb = peak / (1024 * 1024)
    
    print("-" * 40)
    print(f"Training Completed!")
    print(f"Time Taken: {duration:.2f} seconds ({duration/3600:.2f} hours)")
    print(f"Peak Memory Usage: {peak_mb:.2f} MB")
    
    # Analyze Vocabulary
    longest_token_bytes = b""
    longest_token_str = ""
    
    for token_id, token_bytes in vocab.items():
        if len(token_bytes) > len(longest_token_bytes):
            longest_token_bytes = token_bytes
            
    # Decode for display
    try:
        longest_token_str = longest_token_bytes.decode('utf-8', errors='replace')
    except:
        longest_token_str = str(longest_token_bytes)

    print(f"Vocab Size: {len(vocab)}")
    print(f"Longest Token (bytes): {longest_token_bytes}")
    print(f"Longest Token (decoded): {repr(longest_token_str)}")
    
    # Serialize results
    output_file = "tinystories_bpe.pkl"
    with open(output_file, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    train_tinystories()
