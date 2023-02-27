import torch.nn as nn
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention


class Block(nn.Module):
    def __init__(self, block_size, embedding_size, num_heads, dropout):
        super().__init__()

        self.heads = MultiHeadAttention(num_heads, block_size, embedding_size, dropout)
        self.ff = FeedForward(embedding_size, dropout)

        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self.heads(self.layer_norm1(x))
        x = x + self.ff(self.layer_norm2(x))
        return x
