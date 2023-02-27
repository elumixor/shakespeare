import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from typing import Optional
from nn import Model

from .checkpoints import load_checkpoint, save_checkpoint, remove_checkpoints


def train(model: nn.Module, optim: torch.optim.Optimizer,
          trn: torch.utils.data.DataLoader, val: torch.utils.data.DataLoader,
          epochs: int, validate_freq: int, 
          lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
          checkpoint_freq: Optional[int] = None, checkpoints_dir="checkpoints",
          restart=False, project=None, run_name=None, use_wandb=False, num_evaluations=100,
          device="cpu"):
    """
    Trains the model on the data
    """
    if run_name is None and checkpoint_freq is not None:
        raise ValueError("run_name must be specified if checkpoint_freq is specified")

    # Load checkpoint if it exists
    used_checkpoint_for_init = False
    if run_name is not None:
        if restart:
            remove_checkpoints(run_name, checkpoints_dir=checkpoints_dir)
        else:
            used_checkpoint_for_init = load_checkpoint(model, optim, run_name, checkpoints_dir=checkpoints_dir)

    run = wandb.init(project=project, name=run_name, resume=used_checkpoint_for_init) if use_wandb else None

    model.to(device)

    best_loss_val = evaluate(model, val, device=device)

    print(f"Initial model has loss {best_loss_val}")

    # Sometimes during the checkpoint the validation loss does not improve
    # then we will save it during the next successful validation
    needs_save = False

    losses = []

    model.train()
    for epoch in range(epochs):
        for batch in tqdm(trn):
            # Training
            optim.zero_grad()
            model(batch.to(device), return_loss=True).backward()
            optim.step()

        # Validation
        loss_val = None
        if epoch % validate_freq == 0 or epoch == epochs - 1:
            loss_trn, loss_val = evaluate(model, val, trn, num_evaluations=num_evaluations, device=device)
            losses.append((loss_trn, loss_val))

            if lr_scheduler is not None:
                lr_scheduler.step(loss_val)

            if run is not None:
                run.log({"loss_trn": loss_trn, "loss_val": loss_val}, step=epoch)

            print(f"Epoch {epoch}: training: {loss_trn} validation: {loss_val}")

            if loss_val < best_loss_val:
                best_loss_val = loss_val

            if needs_save and loss_val < best_loss_val:
                file_path = save_checkpoint(model, optim, run_name, epoch=epoch, checkpoints_dir=checkpoints_dir)
                print(f"Saving checkpoint to {file_path}. The model improved from {best_loss_val} to {loss_val}")
                best_loss_val = loss_val
                needs_save = False

        # Checkpoint
        if run_name is not None and checkpoint_freq is not None and \
                (epoch > 0 and epoch % checkpoint_freq == 0 or epoch == epochs - 1):
            if loss_val is None:
                loss_val = evaluate(model, val, num_evaluations=num_evaluations, device=device)

            if loss_val > best_loss_val:
                print("Skipping checkpoint because the model did not improve")
                needs_save = True

                if epoch < epochs - 1:
                    print("Will save on the next better validation")

                continue

            file_path = save_checkpoint(model, optim, run_name, epoch=epoch, checkpoints_dir=checkpoints_dir)
            print(f"Saving checkpoint to {file_path}. The model improved from {best_loss_val} to {loss_val}")
            best_loss_val = loss_val
            needs_save = False

    print("Best achieved validation loss:", best_loss_val)

    losses = list(zip(*losses))
    return losses


@torch.no_grad()
def evaluate(model: Model, val: torch.utils.data.DataLoader, trn: Optional[torch.utils.data.DataLoader] = None, num_evaluations=100, device="cpu"):
    model.to(device)
    model.eval()

    loss_trn = None
    if trn is not None:
        loss_trn = torch.tensor([
            model(batch.to(device), return_loss=True) for batch, _ in zip(trn, range(num_evaluations))
        ]).mean().item()

    loss_val = torch.tensor([
        model(batch.to(device), return_loss=True) for batch, _ in zip(val, range(num_evaluations))
    ]).mean().item()

    model.train()

    if loss_trn is not None:
        return loss_trn, loss_val

    return loss_val
