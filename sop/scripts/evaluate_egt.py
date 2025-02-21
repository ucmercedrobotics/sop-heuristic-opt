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
from sop.utils.path import Path
from sop.utils.evaluation import evaluate_paths
from sop.utils.visualization import plot_statistics, plot_heuristics
from sop.models.egt import EGT
from sop.inference.rollout import (
    categorical_action_selection,
    reward_failure_scoring_fn,
)
from sop.inference.aco import sop_aco_solver, aco_search, vanilla_search
from sop.train.preprocess import (
    preprocess_graph_mean,
    preprocess_graph_normalize_budget,
)
from sop.train.reinforce import heuristic_walk, reinforce_loss, reinforce_loss_ER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    # General
    device: str = DEVICE

    # Data
    data_dir: str = "data"
    eval_data_name: str = "dataset_32_50_0"

    # checkpoint
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "EGT"
    checkpoint_name: Optional[str] = None

    # Evaluation Params
    p_f: float = 0.1
    num_rollouts: int = 100
    num_iterations: int = 5
    num_runs: int = 100
    # Samples
    num_samples: int = 1000
    kappa: float = 0.5

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
cs.store(name="evaluate_egt", node=Config)


def evaluate_search(
    cfg: Config,
    graphs: TorchGraph,
    heuristic: Tensor,
    search_fn: Callable,
    action_selection_fn: Callable = categorical_action_selection,
):
    batch_size, N = graphs.size()
    p_f = torch.full((graphs.shape[0],), cfg.p_f, dtype=torch.float32)
    # Expand graphs and heuristic
    graphs = graphs.unsqueeze(-1).expand(batch_size, cfg.num_runs).flatten()
    heuristic = (
        heuristic.unsqueeze(-3)
        .expand(batch_size, cfg.num_runs, N, N)
        .reshape(batch_size * cfg.num_runs, N, N)
    )
    p_f = p_f.unsqueeze(-1).expand(batch_size, cfg.num_runs).flatten()
    # Run search
    paths, _ = sop_aco_solver(
        graphs,
        heuristic,
        cfg.num_rollouts,
        cfg.num_iterations,
        p_f,
        cfg.kappa,
        search_fn,
        action_selection_fn,
    )

    return paths.reshape(batch_size, cfg.num_runs)


@hydra.main(version_base=None, config_name="evaluate_egt")
def main(cfg: Config) -> None:
    torch.set_default_device(cfg.device)
    # Experiment
    experiment_dir = os.path.join(cfg.checkpoint_dir, cfg.experiment_name)

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

    # Load Checkpoint
    if cfg.checkpoint_name is not None:
        load_checkpoint(model, optimizer, experiment_dir, cfg.checkpoint_name)

    # Load Eval Dataset
    print("Loading Eval Dataset...")
    dataloader = DataLoader(cfg.data_dir)
    graphs, sop_cfg = dataloader.load(cfg.eval_data_name)

    # Compute heuristic
    model.eval()
    node_features, edge_features, adj = preprocess_graph_normalize_budget(graphs)
    heuristic = model(node_features, edge_features, adj)

    # Walk
    p_f = torch.full((graphs.shape[0],), cfg.p_f, dtype=torch.float32)
    output = heuristic_walk(
        graphs, heuristic, cfg.num_rollouts, p_f, categorical_action_selection
    )
    walk_paths = output.path

    # Vanilla
    vanilla_paths = evaluate_search(cfg, graphs, heuristic, search_fn=vanilla_search)
    # ACO
    aco_paths = evaluate_search(cfg, graphs, heuristic, search_fn=aco_search)

    # MILP
    milp_paths = dataloader.load_solutions(cfg.eval_data_name, prefix="milp")
    milp_paths = milp_paths.unsqueeze(-1)

    labels = ["Walk", "Vanilla", "ACO", "MILP"]
    # labels = ["Walk", "Vanilla", "MILP"]
    data_labels = ["Min", "Avg", "Max", "p_f"]
    paths = [walk_paths, vanilla_paths, aco_paths, milp_paths]
    # paths = [walk_paths, vanilla_paths, milp_paths]
    data = {}

    for label, path in zip(labels, paths):
        # Compute statistics
        reward = path.reward.sum(-1)
        min_reward = torch.min(reward, dim=-1).values.mean()
        avg_reward = torch.mean(reward, dim=-1).mean()
        max_reward = torch.max(reward, dim=-1).values.mean()
        avg_cost, F = evaluate_paths(graphs, path, cfg.num_samples, cfg.kappa)

        # Add to data dict
        data[label] = [
            float(min_reward),
            float(avg_reward),
            float(max_reward),
            float(F.mean()),
        ]

    plot_statistics(data, title="EGT Evaluation", data_labels=data_labels)
    plot_heuristics(heuristics=[heuristic[0].detach().numpy()], titles=["EGT"])


if __name__ == "__main__":
    main()
