from data import Data
from nn import Model
from training import train
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    data = Data(block_size=8, batch_size=4, device=device)

    model = Model(data.vocab_size)
    model = model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses_trn, losses_val = train(model, data, optim, epochs=10000, batch_size=32)

    plt.plot(losses_trn, label="Train")
    plt.plot(losses_val, label="Validation")
    plt.legend()
    plt.show()

    print(data.decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 1000)[0].tolist()))
