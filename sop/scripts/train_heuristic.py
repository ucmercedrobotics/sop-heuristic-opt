from typing import Optional
from dataclasses import dataclass
import time
import os
from datetime import datetime
from tqdm import tqdm

import hydra
from hydra.core.config_store import ConfigStore
import torch
import torch.nn.functional as F
from torch import Tensor
import rootutils
import optuna
import numpy as np

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sop.utils.graph_torch import TorchGraph, generate_sop_graphs
from sop.utils.visualization import plot_solutions, plot_heuristics
from sop.utils.path import evaluate_path, path_to_heatmap, Path
from sop.utils.seed import random_seed, set_seed

from sop.milp.pulp_milp_sop import sop_milp_solver
from sop.mcts.aco import (
    ACOParams,
    sop_aco_solver,
    mcts_sopcc_heuristic,
    mcts_sopcc_norm_heuristic,
    random_heuristic,
    small_heuristic,
    aco_rollout,
    reinforce_action_selection,
    RolloutOutput,
    carpin_scoring_fn,
    carpin_alpha_scoring_fn,
)
from sop.models.nar_encoder import GATEmbedding, NAREncoder
from sop.scripts.run_milp import run_milp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -- Config
@dataclass
class Config:
    # Data
    dataset_dir: str = "data"
    visual_dir: str = "viz"
    seed: Optional[int] = None
    dataset_name: Optional[str] = None
    # Batch
    batch_size: int = 1
    device: str = DEVICE
    # Graph
    num_nodes: int = 50
    budget: int = 2
    start_node: int = 0
    goal_node: int = 19
    # Sampling
    num_samples: int = 100
    kappa: float = 0.5
    # ACO
    num_rollouts: int = 100
    num_iterations: int = 5
    p_f: float = 0.1
    # Training
    num_epochs: int = 5
    num_steps: int = 100
    lr: float = 1e-3
    topk: int = 10
    # MILP
    milp_time_limit: Optional[int] = None


cs = ConfigStore.instance()
cs.store(name="train_heuristic", node=Config)


def generate_data(cfg):
    # Set seed
    if cfg.seed is None:
        cfg.seed = random_seed()
    set_seed(cfg.seed)

    cfg.start_node = 0
    cfg.goal_node = cfg.num_nodes - 1

    graphs = generate_sop_graphs(
        cfg.batch_size,
        cfg.num_nodes,
        cfg.start_node,
        cfg.goal_node,
        cfg.budget,
        cfg.num_samples,
        cfg.kappa,
    )

    # Reset seed
    set_seed(random_seed())

    return graphs


def rollout(
    params: ACOParams,
    heuristic: Tensor,
    graph: TorchGraph,
    num_rollouts: int,
    p_f: float,
):
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    start_node = graph.extra["start_node"]
    budget = graph.extra["budget"]
    goal_node = graph.extra["goal_node"]

    current_node = start_node.clone()
    current_budget = budget.clone()

    # Initialize Path
    path = Path.empty(batch_size, num_nodes)
    path.append(indices, current_node)
    path.mask[indices, goal_node] = 1

    output = aco_rollout(
        params,
        heuristic,
        graph,
        path,
        current_node,
        current_budget,
        num_rollouts,
        p_f,
        action_selection_fn=reinforce_action_selection,
        store_log_probs=True,
    )

    return output


def reinforce_loss(cfg: Config, reward: Tensor, log_probs: Tensor):
    baseline = reward.mean(-1)

    A = reward - baseline.unsqueeze(-1)
    sum_probs = log_probs.sum(-1)

    loss = (A * sum_probs).sum(-1) / cfg.num_rollouts
    loss = loss.sum(-1)

    return loss


def train_step(loss: Tensor, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_step_manual(loss: Tensor, heuristic: Tensor, lr: float):
    loss.backward()
    heuristic.data -= lr * heuristic.grad
    heuristic.grad = None


def preprocess_graph(graph: TorchGraph):
    # node features: reward, is_start, is_goal
    # edge features: normalized mean samples, is_start_edge, is_goal_edge
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    reward = graph.nodes["reward"]
    start_node = graph.extra["start_node"]
    goal_node = graph.extra["goal_node"]
    samples = graph.edges["samples"]

    # Adjacency Matrix
    adj = graph.edges["adj"]

    # Node Features

    is_start_node = torch.zeros((batch_size, num_nodes), dtype=torch.long)
    is_goal_node = torch.zeros((batch_size, num_nodes), dtype=torch.long)
    is_start_node[indices, start_node] = 1
    is_goal_node[indices, goal_node] = 1

    node_features = torch.cat(
        [
            reward.unsqueeze(-1),
            F.one_hot(is_start_node, num_classes=2),
            F.one_hot(is_goal_node, num_classes=2),
        ],
        dim=-1,
    )

    # Edge Features

    sample_mean = samples.mean(-1)

    is_start_edge = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.long)
    is_goal_edge = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.long)
    is_start_edge[indices, start_node] = 1
    is_goal_edge[indices, :, goal_node] = 1

    edge_features = torch.cat(
        [
            sample_mean.unsqueeze(-1),
            F.one_hot(is_start_edge, num_classes=2),
            F.one_hot(is_goal_edge, num_classes=2),
        ],
        dim=-1,
    )

    return node_features, adj, edge_features


@hydra.main(version_base=None, config_name="train_heuristic")
def main(cfg: Config) -> None:
    torch.set_default_device(cfg.device)

    # Load Dataset/Generate Data
    # TODO

    # TODO: Normalize sampled distances with budget

    # Initialize Model
    node_embedder = GATEmbedding(node_dim=5, out_dim=32, edge_dim=5)
    model = NAREncoder(node_embedder)

    # Params
    params = ACOParams()

    # Generate Dataset
    print("Generating Data...")
    graphs = generate_data(cfg)

    # Dirs
    viz_path = cfg.dataset_dir + "/" + cfg.visual_dir
    viz_name = f"{cfg.seed}_{cfg.batch_size}_{cfg.num_nodes}"
    print(f"Creating vizualization folder {viz_path}...")
    os.makedirs(viz_path, exist_ok=True)
    viz_prefix = f"{viz_path}/{viz_name}"

    # Generate "model"
    comp_heuristic = mcts_sopcc_norm_heuristic(
        graphs.nodes["reward"], graphs.edges["samples"]
    )

    # Broadcast...
    n = 10
    graph = graphs[0]
    broadcasted_graph = graph.unsqueeze(0).expand(n).clone()

    score = (
        comp_heuristic[0]
        .clone()
        .detach()
        .unsqueeze(0)
        .expand(n, cfg.num_nodes, cfg.num_nodes)
    )

    # -- MILP
    # run_milp(cfg, graph, viz_prefix)

    # -- Initial Evaluation
    path, is_success = sop_aco_solver(
        params,
        broadcasted_graph,
        score,
        num_rollouts=cfg.num_rollouts,
        num_iterations=cfg.num_iterations,
        p_f=cfg.p_f,
        kappa=cfg.kappa,
    )
    reward = path.reward.sum(-1).mean(-1).mean(-1)
    print(f"Initial Reward: {reward}")
    failure_prob, avg_cost = evaluate_path(
        path[0].unsqueeze(0), graph.unsqueeze(0), cfg.num_samples, cfg.kappa
    )
    aco_info = (
        "ACO; "
        + f"R: {path[0].reward.sum(-1):.5f}, "
        + f"B: {float(graph.extra['budget'])}, "
        + f"C: {float(avg_cost):.5f}, "
        + f"F: {float(failure_prob):.3f} "
        + f"N: {int(path[0].length)}"
    )
    print(aco_info)
    plot_solutions(
        graph,
        paths=[path[0]],
        titles=[aco_info],
        out_path=viz_prefix + "_aco",
        rows=1,
        cols=1,
    )

    # -- Training Loop
    pbar = tqdm(maxinterval=cfg.num_epochs * cfg.num_steps)
    indices = torch.arange(cfg.batch_size)
    b_indices = indices.unsqueeze(-1).expand((-1, cfg.topk))

    heuristic = small_heuristic(cfg.batch_size, cfg.num_nodes)
    # heuristic = comp_heuristic
    heuristic.requires_grad_(True)

    for epoch in range(cfg.num_epochs):
        for step in range(cfg.num_steps):
            # print("Generating Data...")
            # graphs = generate_data(cfg)

            # print("Creating Heuristic with GNN...")
            # node_features, adj, edge_features = preprocess_graph(graphs)
            # heuristic = model(node_features, adj, edge_features)
            # heuristic = mcts_sopcc_norm_heuristic(
            #     graphs.nodes["reward"], graphs.edges["samples"]
            # )
            # heuristic.requires_grad_(True)

            output = rollout(params, heuristic, graphs, cfg.num_rollouts, cfg.p_f)

            reward = carpin_alpha_scoring_fn(params, graphs, output, cfg.p_f)
            _, topk_indices = torch.topk(reward, k=cfg.topk, dim=-1)
            topk_reward = reward[b_indices, topk_indices]
            topk_log_probs = output.log_probs[b_indices, topk_indices]

            loss = reinforce_loss(cfg, topk_reward, topk_log_probs)
            train_step_manual(loss, heuristic, cfg.lr)

            avg_reward = topk_reward.mean(-1).mean(-1)

            pbar.set_description(f"[{epoch}:{step}] Loss: {loss} Reward: {avg_reward}")
            pbar.update()

        score = (
            heuristic[0]
            .clone()
            .detach()
            .unsqueeze(0)
            .expand(n, cfg.num_nodes, cfg.num_nodes)
        )
        path, is_success = sop_aco_solver(
            params,
            broadcasted_graph,
            score,
            num_rollouts=cfg.num_rollouts,
            num_iterations=cfg.num_iterations,
            p_f=cfg.p_f,
            kappa=cfg.kappa,
        )
        reward = path.reward.sum(-1).mean(-1).mean(-1)

        failure_prob, avg_cost = evaluate_path(
            path[0].unsqueeze(0), graph.unsqueeze(0), cfg.num_samples, cfg.kappa
        )
        aco_info = (
            "ACO; "
            + f"R: {path[0].reward.sum(-1):.5f}, "
            + f"B: {float(graph.extra['budget'])}, "
            + f"C: {float(avg_cost):.5f}, "
            + f"F: {float(failure_prob):.3f} "
            + f"N: {int(path[0].length)}"
        )
        reward = path.reward.sum(-1).mean(-1).mean(-1)
        print(f"{epoch} Reward: {reward}")
        print(aco_info)
        plot_solutions(
            graph,
            paths=[path[0]],
            titles=[aco_info],
            out_path=viz_prefix + f"_aco_{epoch}",
            rows=1,
            cols=1,
        )

    pbar.close()


if __name__ == "__main__":
    main()
