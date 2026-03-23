import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0

        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))

        return out
