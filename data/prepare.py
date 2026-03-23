import os
import numpy as np
import tiktoken
import time
from datasets import load_dataset

start_time = time.time()

def t():
    return f"[{time.time() - start_time:.2f}s]"

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = "AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0"
TEXT_COLUMN = "assistant"
SPLIT_RATIO = 0.9

print(f"{t()} - Using dataset: {DATASET_NAME}")
ds = load_dataset(DATASET_NAME, split="train")

print(f"{t()} - Concatenating data...")
full_text = "\n\n".join(row[TEXT_COLUMN] for row in ds if row[TEXT_COLUMN])
print(f"Total characters: {len(full_text):,}")

print(f"{t()} - Tokenizing")
enc = tiktoken.get_encoding("gpt2")
token_ids = enc.encode(full_text, allowed_special={"<|endoftext|>"})
print(f"{t()} - Total tokens: {len(token_ids):,}")

tokens = np.array(token_ids, dtype=np.uint16)

n = int(SPLIT_RATIO * len(tokens))
train_data = tokens[:n]
val_data = tokens[n:]
print(f"{t()} - Train tokens: {len(train_data):,} | Validation tokens: {len(val_data):,}")

train_path = os.path.join(OUTPUT_DIR, "train.bin")
val_path = os.path.join(OUTPUT_DIR, "val.bin")

train_data.tofile(train_path)
val_data.tofile(val_path)

print(f"\n{t()} - Saved {train_path}  ({os.path.getsize(train_path) / 1e6:.1f} MB)")
print(f"{t()} Saved {val_path}  ({os.path.getsize(val_path) / 1e6:.1f} MB)")
print(f"\n{t()} - Vocab size (gpt2):", enc.n_vocab)
