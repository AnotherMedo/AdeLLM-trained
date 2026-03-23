from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 512
    n_embd: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.1
