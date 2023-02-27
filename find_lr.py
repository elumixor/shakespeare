import torch
import itertools


def find_lr(model, data, Optim, lre_min=-7, lre_max=1, size=1000, return_info=False, device="cpu"):
    model.to(device)
    model.train()

    lres = torch.linspace(lre_min, lre_max, size)
    lrs = 10 ** lres
    losses = torch.zeros_like(lrs)

    init_loss = None
    for i, (lr, batch) in enumerate(zip(lrs, itertools.cycle(data))):
        optim = Optim(lr)

        optim.zero_grad()
        loss = model(batch.to(device), return_loss=True)
        loss.backward()
        optim.step()

        loss = loss.item()

        if init_loss is None:
            init_loss = loss

        if loss > 4 * init_loss:
            print("Stopping early because the loss has exploded")
            break

        losses[i] = loss

    losses = losses[:i]
    lres = lres[:i]
    lrs = lrs[:i]

    # Smooth the losses - take the average of the previous 5 and next 5 losses around the current one
    smoothed_losses = torch.zeros_like(losses)

    for i, loss in enumerate(losses):
        selected = losses[max(0, i - 5):min(len(losses), i + 5)]
        smoothed = selected.mean()
        smoothed_losses[i] = smoothed

    # Find the minimum of the smoothed losses
    min_loss = smoothed_losses.min().item()
    optimal_lr = lrs[smoothed_losses.argmin()].item()

    if not return_info:
        return optimal_lr

    return optimal_lr, min_loss, lres, losses, smoothed_losses
