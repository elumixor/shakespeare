import os
import requests
import torch
from typing import Literal
from vocab import Vocab


def load_data(val=0.2, device: Literal["cpu", "cuda"] = "cpu", block_size=8, batch_size=32, text_file_path="input.txt"):
    # Load the text
    file_path = os.path.join(os.path.dirname(__file__), text_file_path)

    # Download the file if not done so already
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            text_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

            print("Downloading text from {}".format(text_url))
            content = requests.get(text_url).text
            file.write(content)

    # If it is, just read the content
    else:
        with open(file_path, "r") as file:
            text = file.read()

    vocab = Vocab(text)
    data = torch.tensor(vocab.encode(text), dtype=torch.long, device=device)

    val = int(len(data) * (1 - val))

    return Dataset(data[:val]), Dataset(data[val:]), vocab


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        offsets, block_size = idx

        # For each offset, we create a block of size block_size
        x = torch.stack([self.data[i:i + block_size] for i in offsets])
        y = torch.stack([self.data[i + 1:i + 1 + block_size] for i in offsets])

        return DeviceTuple(x, y)

class DeviceTuple:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to(self, device):
        return DeviceTuple(self.x.to(device), self.y.to(device))

    def __getitem__(self, idx):
        return self.x if idx == 0 else self.y

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        return 2


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: Dataset, block_size: int, batch_size: int, num_workers=0, *args, **kwargs):
        sampler = BlockSampler(len(dataset), block_size=block_size, batch_size=batch_size)
        super().__init__(dataset, sampler=sampler, batch_size=None, num_workers=num_workers, *args, **kwargs)


class BlockSampler(torch.utils.data.Sampler):
    def __init__(self, data_size, block_size, batch_size):
        self.data_size = data_size
        self.block_size = block_size
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(len(self)):
            # Generate batch_size number of random offsets
            offsets = torch.randint(0, self.data_size - self.block_size, (self.batch_size,))

            yield offsets, self.block_size

    def __len__(self):
        return self.data_size // (self.batch_size * self.block_size)
