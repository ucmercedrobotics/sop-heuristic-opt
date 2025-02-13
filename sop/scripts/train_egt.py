from typing import Optional, Callable
from dataclasses import dataclass
import time
import os
from tqdm import tqdm

import hydra
from hydra.core.config_store import ConfigStore

import torch
from torch import Tensor
from torch.optim import Adam
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sop.utils.graph_torch import TorchGraph, generate_sop_graphs, preprocess_graph
from sop.utils.visualization import (
    plot_solutions,
    plot_heuristics,
    Stats,
    plot_statistics,
)
from sop.utils.path import evaluate_path, path_to_heatmap, Path
from sop.utils.seed import random_seed, set_seed

from sop.mcts.aco2 import (
    sop_aco_solver,
    rollout,
    categorical_action_selection,
    eps_greedy_action_selection,
    reward_failure_scoring_fn,
    RolloutOutput,
    aco_search,
    search,
)

from sop.models.egt import EGT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    # Data
    dataset_dir: str = "data"
    visual_dir: str = "viz"
    seed: Optional[int] = None
    dataset_name: Optional[str] = None
    # Batch
    batch_size: int = 32
    device: str = DEVICE
    # Graph
    num_nodes: int = 50
    budget: int = 2
    start_node: int = 0
    goal_node: int = 49
    # Sampling
    num_samples: int = 100
    kappa: float = 0.5
    # ACO
    num_rollouts: int = 100
    num_iterations: int = 10
    p_f: float = 0.1
    # EGT
    node_hidden_dim: int = 64
    edge_hidden_dim: int = 16
    num_heads: int = 4
    num_layers: int = 3
    # Training
    lr: float = 1e-3
    entropy_coef: float = 0.1
    num_epochs: int = 10
    num_steps: int = 25
    # Evaluation
    num_runs: int = 10
    # MILP
    milp_time_limit: Optional[int] = None


cs = ConfigStore.instance()
cs.store(name="aco_sop2", node=Config)


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


# --- DATASET


def load_dataset(cfg: Config) -> TorchGraph:
    # Checks
    start_node = clamp(cfg.start_node, 0, cfg.num_nodes - 1)
    goal_node = clamp(cfg.goal_node, 0, cfg.num_nodes - 1)

    # Set seed if given
    if cfg.seed is not None:
        set_seed(cfg.seed)

    graphs = generate_sop_graphs(
        cfg.batch_size,
        cfg.num_nodes,
        start_node,
        goal_node,
        cfg.budget,
        cfg.num_samples,
        cfg.kappa,
    )

    # Reset seed
    set_seed(random_seed())

    return graphs


# --- VISUALIZATION


def create_viz_prefix(cfg: Config):
    viz_path = os.path.join(cfg.dataset_dir, cfg.visual_dir)
    viz_name = f"{cfg.seed}_{cfg.batch_size}_{cfg.num_nodes}"
    print(f"Creating vizualization folder {viz_path}...")
    os.makedirs(viz_path, exist_ok=True)
    viz_prefix = os.path.join(viz_path, viz_name)
    return viz_prefix


def save_path(
    cfg: Config,
    paths: Path,
    graphs: TorchGraph,
    label: str,
    index: int = 0,
    prefix: Optional[str] = None,
):
    path, graph = paths[index].unsqueeze(0), graphs[index].unsqueeze(0)

    failure_prob, avg_cost = evaluate_path(path, graph, cfg.num_samples, cfg.kappa)
    info = f"""\
    {label};
    R: {float(path.reward.sum(-1)):.5f},
    """
    info = (
        f"{label}; "
        f"R: {float(path.reward.sum(-1)):.5f}, "
        f"B: {float(graph.extra['budget'])}, "
        f"C: {float(avg_cost):.5f}, "
        f"F: {float(failure_prob):.3f}, "
        f"N: {float(int(path.length))}, "
    )
    print(info)

    out_path = prefix + f"_{label}" if prefix is not None else prefix
    plot_solutions(
        graph.squeeze(0),
        paths=[path.squeeze(0)],
        titles=[info],
        out_path=out_path,
        rows=1,
        cols=1,
    )


# --- HEURISTICS


def mcts_sopcc_heuristic(rewards: Tensor, sampled_costs: Tensor):
    average_cost = sampled_costs.mean(dim=-1)
    return rewards.unsqueeze(-1) / average_cost


def mcts_sopcc_norm_heuristic(rewards: Tensor, sampled_costs: Tensor):
    # Average and normalize costs
    s = sampled_costs.mean(dim=-1)
    s_norm = s / s.sum(dim=-1, keepdim=True)
    # Average and normalize rewards
    r_norm = rewards / rewards.sum(-1, keepdim=True)
    return (r_norm.unsqueeze(-1) + 1e-5) / s_norm


# --- REINFORCE


def generate_paths(
    cfg: Config,
    graphs: TorchGraph,
    heuristic: Tensor,
    action_selection_fn: Callable = categorical_action_selection,
):
    start_node = graphs.extra["start_node"]
    budget = graphs.extra["budget"]
    goal_node = graphs.extra["goal_node"]

    batch_size, num_nodes = graphs.size()
    indices = torch.arange(batch_size)

    # Loop State
    current_node = start_node.clone()
    current_budget = budget.clone()

    # Initialize Path
    path = Path.empty(batch_size, num_nodes)
    path.append(indices, current_node)
    # Mask goal node
    path.mask[indices, goal_node] = 1

    # Rollout
    output = rollout(
        heuristic,
        graphs,
        current_node,
        current_budget,
        path,
        cfg.num_rollouts,
        cfg.p_f,
        action_selection_fn,
        store_log_probs=True,
    )

    return output


def temperature_scaled_softmax(logits: Tensor, temperature: Tensor, dim=1):
    logits = logits / temperature
    return torch.softmax(logits, dim)


def reinforce_loss(cfg: Config, scores: Tensor, log_probs: Tensor):
    A = (scores - scores.mean(-1, keepdim=True)) / (scores.std(-1, keepdim=True) + 1e-9)
    sum_probs = log_probs.sum(-1)
    loss = (A * sum_probs).sum(-1)
    loss = loss.mean(-1)

    return -loss


def reinforce_loss_entropy_regularization(
    cfg: Config, scores: Tensor, log_probs: Tensor
):
    A = (scores - scores.mean(-1, keepdim=True)) / (scores.std(-1, keepdim=True) + 1e-9)
    sum_probs = log_probs.sum(-1)
    reinforce_loss = (A * sum_probs).sum(-1)

    entropy_loss = -(log_probs.exp() * log_probs).sum(-1).sum(-1)
    loss = reinforce_loss.mean(-1) + cfg.entropy_coef * entropy_loss.mean(-1)

    return -loss


def reinforce_loss_entropy_regularization_manual(
    cfg: Config, scores: Tensor, log_probs: Tensor
):
    A = (scores - scores.mean(-1, keepdim=True)) / (scores.std(-1, keepdim=True) + 1e-9)
    sum_probs = log_probs.sum(-1)
    reinforce_loss = (A * sum_probs).sum(-1)

    entropy_loss = -(log_probs.exp() * log_probs).sum(-1).sum(-1)
    loss = reinforce_loss.sum(-1) + cfg.entropy_coef * entropy_loss.sum(-1)

    return -loss


def train_reinforce_manual(
    cfg: Config,
    graphs: TorchGraph,
    heuristic: Tensor,
    lr: float = 1e-2,
    loss_fn: Callable = reinforce_loss,
    action_selection_fn: Callable = categorical_action_selection,
):
    def train_step(scores: Tensor, log_probs: Tensor):
        loss = loss_fn(cfg, scores, log_probs)
        loss.backward()
        heuristic.data -= lr * heuristic.grad
        heuristic.grad = None

        return loss

    output = generate_paths(cfg, graphs, heuristic, action_selection_fn)
    scores = reward_failure_scoring_fn(output, cfg.p_f)
    loss = train_step(scores, output.log_probs)

    return loss, scores


def train_reinforce(
    cfg: Config,
    graphs: TorchGraph,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable = reinforce_loss,
    action_selection_fn: Callable = categorical_action_selection,
):
    def train_step(scores: Tensor, log_probs: Tensor):
        loss = loss_fn(cfg, scores, log_probs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    # Predict Heuristic
    node_features, edge_features, adj = preprocess_graph(graphs)
    heuristic = model(node_features, edge_features, adj)

    output = generate_paths(cfg, graphs, heuristic, action_selection_fn)
    scores = reward_failure_scoring_fn(output, cfg.p_f)
    loss = train_step(scores, output.log_probs)

    return loss, scores


# --- EVALUATION
def eval_model(
    cfg: Config,
    graphs: TorchGraph,
    model: torch.nn.Module,
    action_selection_fn: Callable = categorical_action_selection,
):
    node_features, edge_features, adj = preprocess_graph(graphs)
    heuristic = model(node_features, edge_features, adj)
    output = generate_paths(cfg, graphs, heuristic, action_selection_fn)
    scores = reward_failure_scoring_fn(output, cfg.p_f)

    return output, scores


def eval_heuristic(
    cfg: Config,
    graphs: TorchGraph,
    heuristic: Tensor,
    action_selection_fn: Callable = categorical_action_selection,
):
    output = generate_paths(cfg, graphs, heuristic, action_selection_fn)
    scores = reward_failure_scoring_fn(output, cfg.p_f)

    return output, scores


# --- TEST


def test_search(
    cfg: Config,
    graphs: TorchGraph,
    heuristic: Tensor,
    index: int = 0,
    search_fn: Callable = search,
):
    graph = graphs[index]
    heuristic = heuristic[index]

    # Expand graph
    graph = graph.unsqueeze(0).expand(cfg.num_runs)
    heuristic = heuristic.unsqueeze(0).expand(
        (cfg.num_runs, cfg.num_nodes, cfg.num_nodes)
    )

    paths, is_success = sop_aco_solver(
        graph,
        heuristic,
        cfg.num_rollouts,
        cfg.num_iterations,
        cfg.p_f,
        cfg.kappa,
        search_fn=search_fn,
    )
    reward = paths.reward.sum(-1)

    min_reward = float(torch.min(reward))
    avg_reward = float(reward.sum(-1) / cfg.num_runs)
    max_reward = float(torch.max(reward))
    failure_prob = float(1 - (is_success.sum(-1) / cfg.num_runs))

    print(f"Results averaged over {cfg.num_runs} runs:")
    print(
        "Params:\n"
        + f"- num_nodes: {cfg.num_nodes}\n"
        + f"- p_f: {cfg.p_f}\n"
        + f"- num_iterations: {cfg.num_iterations}\n"
        + f"- num_rollouts: {cfg.num_rollouts}\n"
        + f"- num_samples: {cfg.num_samples}"
    )
    print(f"Min Reward: {min_reward}")
    print(f"Avg Reward: {avg_reward}")
    print(f"Max Reward: {max_reward}")
    print(f"Failure Prob: {failure_prob:0.3f}")

    return Stats(
        paths=paths,
        min_reward=min_reward,
        avg_reward=avg_reward,
        max_reward=max_reward,
        failure_prob=failure_prob,
    )


def test_rollout(cfg: Config, graphs: TorchGraph, heuristic: Tensor, index: int = 0):
    graph = graphs[index].unsqueeze(0)
    heuristic = heuristic[index].unsqueeze(0)
    output = generate_paths(cfg, graph, heuristic)

    reward = output.path.reward.sum(-1)
    min_reward = float(torch.min(reward))
    avg_reward = float(reward.sum(-1) / cfg.num_rollouts)
    max_reward = float(torch.max(reward))
    failure_prob = (output.residual < 0).sum(-1) / output.residual.shape[-1]
    failure_prob = float(failure_prob.mean(-1))

    print(f"Results averaged over {cfg.num_rollouts} runs:")
    print(
        "Params:\n"
        + f"- num_nodes: {cfg.num_nodes}\n"
        + f"- p_f: {cfg.p_f}\n"
        + f"- num_samples: {cfg.num_samples}"
    )
    print(f"Min Path: {min_reward}")
    print(f"Avg Reward: {avg_reward}")
    print(f"Max Reward: {max_reward}")
    print(f"Failure Prob: {failure_prob:0.3f}")

    return Stats(
        paths=output.path.squeeze(0),
        min_reward=min_reward,
        avg_reward=avg_reward,
        max_reward=max_reward,
        failure_prob=failure_prob,
    )


# -- MAIN
def main_egt(cfg: Config, eval_graphs: TorchGraph, writer: SummaryWriter):
    # Load sample data
    node_features, edge_features, adj = preprocess_graph(eval_graphs)
    node_dim = node_features.shape[-1]
    edge_dim = edge_features.shape[-1]

    # Load model
    egt = EGT(
        node_dim,
        edge_dim,
        cfg.node_hidden_dim,
        cfg.edge_hidden_dim,
        cfg.num_heads,
        cfg.num_layers,
    )
    # Summary
    N = cfg.num_nodes
    summary(egt, [(1, N, node_dim), (1, N, N, edge_dim), (1, N, N)])
    # Load optimizer
    optimizer = Adam(egt.parameters(), lr=cfg.lr)
    # Initial Test
    heuristic = egt(node_features, edge_features, adj)
    old_heuristic = heuristic.clone().detach()

    # Initial Eval
    _, scores = eval_model(cfg, eval_graphs, egt)
    writer.add_scalar("EGT/eval", scores.mean(), 0)

    # Train Model
    i = 0
    for epoch in range(cfg.num_epochs):
        print(f"Starting epoch: {epoch}")
        for step in tqdm(range(cfg.num_steps)):
            # 1. Generate Graphs
            graphs = load_dataset(cfg)

            # 2. Train Reinforce
            loss, scores = train_reinforce(
                cfg,
                graphs,
                egt,
                optimizer,
                loss_fn=reinforce_loss_entropy_regularization_manual,
                action_selection_fn=categorical_action_selection,
            )

            # 3. Log
            writer.add_scalar("EGT/train", scores.mean(), i)
            writer.add_scalar("EGT/loss", loss, i)
            i += 1

        # 4. Evaluation
        print(f"[{epoch}] Running evaluation")
        _, scores = eval_model(cfg, eval_graphs, egt)
        writer.add_scalar("EGT/eval", scores.mean(), i)

    # For comparisons
    heuristic = egt(node_features, edge_features, adj)
    return heuristic, old_heuristic


def main_heuristic(cfg: Config, graphs: TorchGraph, writer: SummaryWriter):
    heuristic = mcts_sopcc_heuristic(graphs.nodes["reward"], graphs.edges["samples"])
    heuristic.requires_grad_(True)

    # Initial Eval
    old_heuristic = heuristic.clone().detach()
    _, scores = eval_heuristic(cfg, graphs, old_heuristic)
    writer.add_scalar("SOPCC/eval", scores.mean(), 0)

    # Optimize Heuristic
    i = 0
    for epoch in range(cfg.num_epochs):
        print(f"Starting epoch: {epoch}")
        for step in tqdm(range(cfg.num_steps)):
            loss, scores = train_reinforce_manual(
                cfg,
                graphs,
                heuristic,
                loss_fn=reinforce_loss_entropy_regularization_manual,
            )

            # 3. Log
            writer.add_scalar("SOPCC/train", scores.mean(), i)
            writer.add_scalar("SOPCC/loss", loss, i)
            i += 1

        print(f"[{epoch}] Running evaluation")
        _, scores = eval_heuristic(cfg, graphs, heuristic)
        writer.add_scalar("SOPCC/eval", scores.mean(), i)

    # For comparisons
    return heuristic, old_heuristic


@hydra.main(version_base=None, config_name="aco_sop2")
def main(cfg: Config) -> None:
    torch.set_default_device(cfg.device)

    # Directories
    viz_prefix = create_viz_prefix(cfg)

    # Generate Eval Dataset
    print("Loading Eval Dataset...")
    eval_graphs = load_dataset(cfg)
    cfg.seed = None

    # Tensorboard
    writer = SummaryWriter()

    egt_heuristic, egt_old_heuristic = main_egt(cfg, eval_graphs, writer)
    sopcc_heuristic, sopcc_old_heuristic = main_heuristic(cfg, eval_graphs, writer)

    heuristics = [egt_heuristic, sopcc_heuristic]
    old_heuristics = [egt_old_heuristic, sopcc_old_heuristic]
    titles = ["EGT", "SOPCC"]
    for title, heuristic, old_heuristic in zip(titles, heuristics, old_heuristics):
        print(f"{title} Old ACO Search")
        o_stats = test_search(
            cfg, eval_graphs, old_heuristic, index=0, search_fn=aco_search
        )
        print("---------")
        print(f"{title} Rollout")
        r_stats = test_rollout(cfg, eval_graphs, heuristic, index=0)
        print("---------")
        print(f"{title} Search")
        s_stats = test_search(cfg, eval_graphs, heuristic, index=0, search_fn=search)
        print("---------")
        print(f"{title} Trained ACO Search")
        a_stats = test_search(
            cfg, eval_graphs, heuristic, index=0, search_fn=aco_search
        )
        print("---------")
        print("Visualizing...")
        save_path(
            cfg, a_stats.paths, eval_graphs, index=0, label=title, prefix=viz_prefix
        )
        plot_heuristics(
            heuristics=[old_heuristic[0], heuristic[0].detach()],
            titles=["Old_H", "New_H"],
            out_path=viz_prefix + f"_{title}_heatmap",
            cols=2,
        )
        plot_statistics(
            stats=[o_stats, r_stats, s_stats, a_stats],
            titles=["Old ACO", "Rollout", "Search", "ACO"],
            out_path=viz_prefix + f"_{title}_stats",
            rows=2,
            cols=2,
        )


if __name__ == "__main__":
    main()
