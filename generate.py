import torch
import tiktoken
import argparse
from model.transformer import GPT
from model.config import ModelConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_path):
    from model.config import ModelConfig
    torch.serialization.add_safe_globals([ModelConfig])
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    config = checkpoint["config"]
    model = GPT(config).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

@torch.no_grad()
def generate(model, prompt, max_tokens=200, temperature=0.8, top_k=40):
    enc = tiktoken.get_encoding("gpt2")

    # Encode the prompt into token ids
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, T)

    for _ in range(max_tokens):
        # Crop to context length if we've grown too long
        tokens_cropped = tokens[:, -model.config.context_length:]

        # Forward pass — we only care about the last position's logits
        logits, _ = model(tokens_cropped)
        logits = logits[:, -1, :]  # (1, vocab_size)

        # Temperature scaling — higher = more random, lower = more focused
        logits = logits / temperature

        # Top-k filtering — zero out everything outside the top k options
        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = float('-inf')

        # Sample from the distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # Append and continue
        tokens = torch.cat([tokens, next_token], dim=1)

    # Decode back to text
    generated = tokens[0].tolist()
    return enc.decode(generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The vulnerability allows")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    print(f"Model loaded — generating...\n")
    output = generate(model, args.prompt, args.max_tokens, args.temperature, args.top_k)
    print(output)
