import os
import time
import numpy as np
import torch
from tqdm import tqdm
from model.transformer import GPT
from model.config import ModelConfig
from training.lr_schedule import get_lr

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR      = "data"
OUT_DIR       = "checkpoints"
MAX_ITERS     = 10000
EVAL_INTERVAL = 500
EVAL_ITERS    = 20       # how many batches to average for eval loss
BATCH_SIZE    = 8
LEARNING_RATE = 3e-4
MIN_LR        = 3e-5     # cosine decays down to this
WARMUP_STEPS  = 300
GRAD_CLIP     = 1.0
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

def load_data(split):
    path = os.path.join(DATA_DIR, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode='r')

train_data = load_data("train")
val_data   = load_data("val")

def get_batch(data, config):
    ix = torch.randint(len(data) - config.context_length, (BATCH_SIZE,))

    x = torch.stack([
        torch.from_numpy(data[i : i + config.context_length].astype(np.int64))
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(data[i + 1 : i + 1 + config.context_length].astype(np.int64))
        for i in ix
    ])

    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, config):
    model.eval()
    losses = {}

    for split, data in [("train", train_data), ("val", val_data)]:
        split_losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, config)
            _, loss = model(x, y)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean().item()

    model.train()
    return losses

def train():
    config = ModelConfig()
    model  = GPT(config).to(DEVICE)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Training on: {DEVICE}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    last_time = time.time()
    losses = {"train": float("nan"), "val": float("nan")}
    pbar = tqdm(range(MAX_ITERS), desc="Training", unit="step")

    for step in pbar:

        lr = get_lr(step, WARMUP_STEPS, MAX_ITERS, LEARNING_RATE, MIN_LR)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, config)
            now = time.time()
            steps_per_sec = EVAL_INTERVAL / (now - last_time) if step > 0 else 0
            eta_seconds = (MAX_ITERS - step) / steps_per_sec if steps_per_sec > 0 else 0
            last_time = now

            tqdm.write(
                f"step {step:>5} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"lr {lr:.2e} | "
                f"eta {eta_seconds/60:.0f}min"
            )

            checkpoint = {
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config":    config,
                "step":      step,
                "val_loss":  losses["val"],
            }
            torch.save(checkpoint, os.path.join(OUT_DIR, f"ckpt_step{step}.pt"))

        x, y = get_batch(train_data, config)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        pbar.set_postfix({
            "loss":  f"{loss.item():.4f}",
            "val":   f"{losses['val']:.4f}",
            "lr":    f"{lr:.2e}",
        })

    checkpoint = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config":    config,
        "step":      MAX_ITERS,
        "val_loss":  losses["val"],
    }
    torch.save(checkpoint, os.path.join(OUT_DIR, "ckpt_final.pt"))
    print("\nTraining complete. Final checkpoint saved.")

if __name__ == "__main__":
    train()
