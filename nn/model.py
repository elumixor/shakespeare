import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab: int):
        super().__init__()

        self.vocab = vocab
        self.embed = nn.Embedding(vocab.size(), vocab.size())

    def forward(self, batch, return_loss=False):
        if return_loss:
            assert len(batch) == 2, "Batch must contain two elements - input and target"
            x, target = batch

            assert x.shape == target.shape, "Input and target must have the same shape"
        else:
            x = batch
            target = None

        assert x.ndim == 2, f"Input must be a batch of sequences. Shape should be (batch_size, sequence_length) but received {x.shape}"

        logits = self.get_logits(x)

        if not return_loss:
            return logits

        batch_size, block_size = x.shape
        logits = logits.view(batch_size * block_size, -1)
        target = target.view(-1)

        loss = F.cross_entropy(logits, target)

        return loss

    def get_logits(self, x):
        return self.embed(x)

    @torch.no_grad()
    def generate(self, start_sequence: str, max_length=100):
        device = self.embed.weight.device
        x = self.vocab.encode(start_sequence).view(1, -1).to(device)

        for _ in range(max_length):
            # Get the logits
            logits = self(x)

            # We are only interested in the last character
            logits = logits[:, -1, :]

            # Sample from the logits
            sampled = torch.multinomial(logits.softmax(dim=-1), num_samples=1)

            # Append the sampled character to the input
            x = torch.cat([x, sampled], dim=1)

        return self.vocab.decode(x.view(-1).cpu().tolist())

    def generate_string(self, length=100):
        device = self.embedding.weight.device
        input_token = torch.zeros((1, 1), dtype=torch.long, device=device)
        return self.generate(input_token, length)[0].tolist()
