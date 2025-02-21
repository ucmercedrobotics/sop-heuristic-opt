from typing import Optional, Callable
from functools import partial
from dataclasses import dataclass
import os
from tqdm import tqdm

import hydra
from hydra.core.config_store import ConfigStore

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sop.utils.dataset import DataLoader
from sop.utils.checkpoint import TrainState, save_checkpoint, load_checkpoint
from sop.utils.graph import TorchGraph, generate_sop_graphs
from sop.models.egt import EGT
from sop.inference.rollout import (
    categorical_action_selection,
    reward_failure_scoring_fn,
)
from sop.train.preprocess import preprocess_graph_mean
from sop.train.reinforce import heuristic_walk, reinforce_loss, reinforce_loss_ER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    # General
    device: str = DEVICE

    # Data
    data_dir: str = "data"
    eval_data_name: str = "dataset_32_50_0"

    # Graph Generation
    num_nodes: int = 50
    budget: int = 2
    start_node: int = 0
    goal_node: int = 49
    num_samples: int = 100
    kappa: float = 0.5

    # Training
    batch_size: int = 32
    num_epochs: int = 100
    num_steps: int = 50

    # checkpoint
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "EGT"
    checkpoint_name: Optional[str] = None

    # Reinforce
    num_rollouts: int = 100
    p_f: float = 0.1
    entropy_coef: float = 0.1

    # EGT
    node_dim: int = 5
    edge_dim: int = 5
    node_hidden_dim: int = 64
    edge_hidden_dim: int = 16
    num_heads: int = 4
    num_layers: int = 3

    # Optimizer
    lr: float = 1e-3


cs = ConfigStore.instance()
cs.store(name="train_egt", node=Config)


def train_model(
    model: nn.Module,
    optimizer: Optimizer,
    graphs: TorchGraph,
    num_rollouts: int,
    p_f: float,
    preprocess_fn: Callable = preprocess_graph_mean,
    loss_fn: Callable = reinforce_loss,
    action_selection_fn: Callable = categorical_action_selection,
):
    model.train()

    # Predict heuristic
    node_features, edge_features, adj = preprocess_fn(graphs)
    heuristic = model(node_features, edge_features, adj)

    # Generate Trajectories
    output = heuristic_walk(graphs, heuristic, num_rollouts, p_f, action_selection_fn)
    scores = reward_failure_scoring_fn(output, p_f)
    loss = loss_fn(scores, output.log_probs)

    # Update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, scores


def eval_model(
    model: nn.Module,
    graphs: TorchGraph,
    num_rollouts: int,
    p_f: float,
    preprocess_fn: Callable = preprocess_graph_mean,
    action_selection_fn: Callable = categorical_action_selection,
):
    model.eval()
    node_features, edge_features, adj = preprocess_fn(graphs)
    heuristic = model(node_features, edge_features, adj)
    output = heuristic_walk(graphs, heuristic, num_rollouts, p_f, action_selection_fn)
    scores = reward_failure_scoring_fn(output, p_f)
    return scores


@hydra.main(version_base=None, config_name="train_egt")
def main(cfg: Config) -> None:
    torch.set_default_device(cfg.device)

    # Experiment
    experiment_dir = os.path.join(cfg.checkpoint_dir, cfg.experiment_name)
    log_dir = os.path.join(experiment_dir, "tensorboard")
    writer = SummaryWriter(log_dir)

    # Load Eval Dataset
    print("Loading Eval Dataset...")
    dataloader = DataLoader(cfg.data_dir)
    eval_graphs, sop_cfg = dataloader.load(cfg.eval_data_name)

    # Load Model and Optimizer
    print("Loading model...")
    model = EGT(
        cfg.node_dim,
        cfg.edge_dim,
        cfg.node_hidden_dim,
        cfg.edge_hidden_dim,
        cfg.num_heads,
        cfg.num_layers,
    )
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    # Model Summary
    summary(model, [(1, 1, cfg.node_dim), (1, 1, 1, cfg.edge_dim), (1, 1, 1)])

    # Load Checkpoint
    if cfg.checkpoint_name is not None:
        print(f"Resuming from checkpoint: `{cfg.checkpoint_name}` ...")
        state = load_checkpoint(model, optimizer, experiment_dir, cfg.checkpoint_name)
    else:
        state = TrainState(cfg=cfg)

    # Entroy regularization loss
    loss_fn = partial(reinforce_loss_ER, entropy_coef=cfg.entropy_coef)

    # Training Loop
    for epoch in range(cfg.num_epochs):
        print(f"{epoch}: Starting Training...")
        for _ in tqdm(range(cfg.num_steps)):
            # 1. Generate random graphs
            graphs = generate_sop_graphs(
                cfg.batch_size,
                cfg.num_nodes,
                cfg.start_node,
                cfg.goal_node,
                cfg.budget,
                cfg.num_samples,
                cfg.kappa,
            )

            # 2. Train
            loss, scores = train_model(
                model, optimizer, graphs, cfg.num_rollouts, cfg.p_f, loss_fn=loss_fn
            )

            # 3. Log metrics
            writer.add_scalar("train/loss", loss, state.num_steps)
            writer.add_scalar("train/avg_score", scores.mean(), state.num_steps)
            state.num_steps += 1

        # 4. Evaluate
        print(f"{epoch}: Evaluating...")
        scores = eval_model(model, eval_graphs, cfg.num_rollouts, cfg.p_f)
        writer.add_scalar(
            "eval/avg_score", scores.mean(), state.num_steps // cfg.num_steps
        )

        # 5. Save checkpoint
        checkpoint_name = f"{state.num_steps}_checkpoint.pth"
        save_checkpoint(model, optimizer, state, experiment_dir, checkpoint_name)


if __name__ == "__main__":
    main()
