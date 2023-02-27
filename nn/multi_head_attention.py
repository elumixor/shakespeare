import torch
import torch.nn as nn
from .attention_head import AttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, block_size: int, embedding_size: int, dropout: float):
        super().__init__()

        head_size = embedding_size // num_heads
        self.heads = nn.ModuleList([AttentionHead(embedding_size, block_size, head_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(embedding_size, embedding_size)  # Why do we need this?
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)  # Why do we need this?
        out = self.dropout(out)

        return out
