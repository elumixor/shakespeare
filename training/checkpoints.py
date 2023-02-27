import os
import re
import torch
import torch.nn as nn
from typing import Optional


def load_checkpoint(model: nn.Module, optim: torch.optim.Optimizer,
                    run_name: str, epoch: Optional[int] = None, checkpoints_dir="checkpoints"):
    # Load the **last** checkpoint that matches the pattern checkpoint-(epoch).pt
    checkpoint = None

    # Get all checkpoints in the checkpoints directory
    try:
        checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]

        if epoch is None:
            max_epoch = -1

            for c in checkpoints:
                if re.match(f"{run_name}(-\d+)?.pt", c):
                    if re.match(f"{run_name}-\d+.pt", c):
                        epoch = int(c.split("-")[1].split(".")[0])
                    else:
                        epoch = 0

                    if epoch > max_epoch:
                        max_epoch = epoch
                        checkpoint = c
        else:
            # Filter matching checkpoints by regex
            for c in checkpoints:
                if re.match(f"{run_name}-{epoch}.pt", c):
                    checkpoint = c
                    break

    except FileNotFoundError:
        pass

    if checkpoint is None:
        print(f"Checkpoints for {run_name} not found, starting from scratch")
        return False

    checkpoint = os.path.join(checkpoints_dir, checkpoint)

    checkpoint_data = torch.load(checkpoint)

    model.load_state_dict(checkpoint_data["model"])
    optim.load_state_dict(checkpoint_data["optim"])

    print(f"Loaded checkpoint from {checkpoint}")

    return True


def save_checkpoint(model: nn.Module, optim: torch.optim.Optimizer, run_name: str, epoch: int,
                    checkpoints_dir="checkpoints", keep_old=False):
    os.makedirs(checkpoints_dir, exist_ok=True)  # Make sure the checkpoint directory exists

    if not keep_old:
        remove_checkpoints(run_name, checkpoints_dir=checkpoints_dir)

    path = os.path.join(checkpoints_dir, f"{run_name}-{epoch}.pt")
    torch.save({ "model": model.state_dict(), "optim": optim.state_dict(), }, path)

    return path


def remove_checkpoints(run_name: str, checkpoints_dir="checkpoints"):
    # Remove all the checkpoints that match the pattern checkpoint-(epoch).pt
    # under the checkpoints directory
    if not os.path.exists(checkpoints_dir):
        return

    for f in os.listdir(checkpoints_dir):
        if re.match(f"{run_name}(-\d+)?.pt", f):
            os.remove(os.path.join(checkpoints_dir, f))
