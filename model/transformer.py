import torch
import torch.nn as nn
from model.attention import CausalSelfAttention

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu  = nn.GELU()
        self.fc2   = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)    # expand
        x = self.gelu(x)   # non-linearity (like ReLU but smoother)
        x = self.fc2(x)    # contract back
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))

        x = x + self.mlp(self.ln2(x))

        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.context_length, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        device = idx.device

        tok_emb = self.token_emb(idx)

        pos = torch.arange(T, device=device)
        pos_emb = self.pos_emb(pos)

        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            return logits, loss

        return logits, None

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
