import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(size, 4 * size),
            nn.ReLU(),
            nn.Linear(4 * size, size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)
