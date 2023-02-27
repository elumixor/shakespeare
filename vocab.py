import torch


class Vocab:
    def __init__(self, text: str):
        self.vocab = sorted(set(text))

        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for i, s in enumerate(self.vocab)}

    def encode(self, seq: str):
        return torch.tensor([self.stoi[s] for s in seq])

    def decode(self, seq: list[int]):
        return "".join([self.itos[i] for i in seq])

    def __len__(self):
        return len(self.vocab)

    def size(self):
        return len(self)

    def __getitem__(self, idx):
        return self.vocab[idx]

    def __iter__(self):
        return iter(self.vocab)