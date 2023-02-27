import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, embedding_size: int, block_size: int, head_size: int, dropout: float):
        super().__init__()

        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)

        self.triu = torch.triu(torch.ones(block_size, block_size), diagonal=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        _, block_size, head_size = v.shape

        weights = q @ k.transpose(-2, -1)
        weights = weights / head_size**0.5
        weights = torch.masked_fill(weights, self.triu, -torch.inf)
        weights = weights.softmax(dim=-1)

        weights = self.dropout(weights)

        return weights @ v

