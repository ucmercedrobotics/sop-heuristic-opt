from typing import Any, Tuple
from dataclasses import dataclass
import os

import torch
import torch.nn as nn
from torch.optim import Optimizer


@dataclass
class TrainState:
    cfg: Any
    num_steps: int = 0


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    state: TrainState,
    experiment_dir: str,
    checkpoint_name: str,
):
    checkpoint = {
        "state": state,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    os.makedirs(experiment_dir, exist_ok=True)
    path = os.path.join(experiment_dir, checkpoint_name)
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    experiment_dir: str,
    checkpoint_name: str,
) -> Tuple[nn.Module, Optimizer, TrainState]:
    path = os.path.join(experiment_dir, checkpoint_name)
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    state = checkpoint["state"]
    return state
