import torch
import torch.nn as nn

from .model import Model


class Transformer(Model):
    def __init__(self, vocab: int, block_size: int, embed_size=32, num_heads=4, num_blocks=4, dropout=0.2):
        super().__init__(vocab)

        self.block_size = block_size
        self.embed_size = embed_size

        self.embed = nn.Embedding(vocab.size(), embed_size)
        self.pos_embed = nn.Embedding(block_size, embed_size)

        self.blocks = nn.Sequential(
            *[Block(embed_size, num_heads, dropout) for _ in range(num_blocks)]
        )

        self.logits = nn.Linear(embed_size, vocab.size())

    def get_logits(self, x):
        device = x.device

        # Positional embedding
        positions = torch.arange(x.shape[1], device=device)
        pos_embed = self.pos_embed(positions)

        # Token embedding
        embed = self.embed(x)

        x = embed + pos_embed  # Add them together, (B, T, E)

        x = self.blocks(x)  # Apply the head

        # Apply the final transformation to obtain the logits
        x = self.logits(x)

        return x

    # We override the generate method because we need to limit the length of the source
    # sequence to the block size
    @torch.no_grad()
    def generate(self, start_sequence: str, max_length=100):
        device = self.embed.weight.device

        self.to("cpu")

        x = self.vocab.encode(start_sequence).view(1, -1)

        for _ in range(max_length):
            x_limited = x[:, -self.block_size:]

            # Get the logits
            logits = self(x_limited)

            # We are only interested in the last character
            logits = logits[:, -1, :]

            # Sample from the logits
            sampled = torch.multinomial(logits.softmax(dim=-1), num_samples=1)

            # Append the sampled character to the input
            x = torch.cat([x, sampled], dim=1)

        self.to(device)
        return self.vocab.decode(x.view(-1).tolist())

    @torch.no_grad()
    def generate_continuous(self, start_sequence: str):
        print(start_sequence, end="", flush=True)

        self.to("cpu")

        x = self.vocab.encode(start_sequence).view(1, -1)

        while True:
            x = x[:, -self.block_size:]

            # Get the logits
            logits = self(x)

            # We are only interested in the last character
            logits = logits[:, -1, :]

            # Sample from the logits
            sampled = torch.multinomial(logits.softmax(dim=-1), num_samples=1)

            character = self.vocab.decode(sampled.view(-1).tolist())[0]
            print(character, end="", flush=True)

            # Append the sampled character to the input
            x = torch.cat([x, sampled], dim=1)


class Head(nn.Module):
    def __init__(self, embed_size, head_size, dropout=0.2):
        super().__init__()

        self.keys = nn.Linear(embed_size, head_size, bias=False)
        self.queries = nn.Linear(embed_size, head_size, bias=False)
        self.values = nn.Linear(embed_size, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        device = x.device
        B, T, E = x.shape  # batch size, sequence length

        # Now add the attention heads
        keys = self.keys(x)        # (B, T, E)
        queries = self.queries(x)  # (B, T, E)
        values = self.values(x)    # (B, T, E)

        w = queries @ keys.transpose(-2, -1)  # (B, T, T)
        w = torch.masked_fill(w, torch.tril(torch.ones(T, T, device=device)) == 0, -torch.inf)  # Forbid communication between future tokens
        w = w / (E ** 0.5)  # Normalization to keep the variance
        w = w.softmax(dim=-1)  # Get the final weights, (B, T, T)
        w = self.dropout(w)  # Apply dropout

        x = w @ values  # (B, T, E)

        return x


class MultiHead(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.2):
        super().__init__()

        head_size = embed_size // num_heads

        self.heads = nn.ModuleList([
            Head(embed_size, head_size, dropout)
            for _ in range(num_heads)
        ])

        # self.project = nn.Linear(embed_size, embed_size)  # Doesn't look like it's needed

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        # x = self.project(x)
        x = self.dropout(x)  # Apply dropout
        return x


class FF(nn.Module):
    def __init__(self, embed_size, dropout=0.2):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)


class Block(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.2):
        super().__init__()

        self.heads = MultiHead(embed_size, num_heads, dropout)
        self.ff = FF(embed_size, dropout)

        self.norm_pre_heads = nn.LayerNorm(embed_size)
        self.norm_pre_ff = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.heads(self.norm_pre_heads(x))
        x = x + self.ff(self.norm_pre_ff(x))
        return x
