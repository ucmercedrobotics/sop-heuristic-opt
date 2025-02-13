from typing import Optional
from dataclasses import dataclass
import time
import os
from tqdm import tqdm

import hydra
from hydra.core.config_store import ConfigStore
import torch
from torch import Tensor
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sop.utils.graph_torch import TorchGraph, generate_sop_graphs
from sop.utils.visualization import plot_solutions, plot_heuristics
from sop.utils.path import evaluate_path, path_to_heatmap, Path
from sop.utils.seed import random_seed, set_seed

from sop.mcts.aco2 import (
    sop_aco_solver,
    rollout,
    categorical_action_selection,
    reward_failure_scoring_fn,
    RolloutOutput,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    goal_node: int = 49
    # Sampling
    num_samples: int = 100
    kappa: float = 0.5
    # ACO
    num_rollouts: int = 100
    num_iterations: int = 10
    p_f: float = 0.1
    num_runs: int = 10
    # Training
    lr: float = 1e-2
    # MILP
    milp_time_limit: Optional[int] = None


cs = ConfigStore.instance()
cs.store(name="aco_sop2", node=Config)


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


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


def mcts_sopcc_heuristic(rewards: Tensor, sampled_costs: Tensor):
    average_cost = sampled_costs.mean(dim=-1)
    return rewards.unsqueeze(-1) / average_cost


def generate_paths(cfg: Config, graphs: TorchGraph, heuristic: Tensor):
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
        action_selection_fn=categorical_action_selection,
        store_log_probs=True,
    )

    return output


def temperature_scaled_softmax(logits: Tensor, temperature: Tensor, dim=1):
    logits = logits / temperature
    return torch.softmax(logits, dim)


def reinforce_loss(cfg: Config, scores: Tensor, log_probs: Tensor):
    # A = scores - scores.mean(-1, keepdim=True)
    A = (scores - scores.mean(-1, keepdim=True)) / (scores.std(-1, keepdim=True) + 1e-9)

    sum_probs = log_probs.sum(-1)

    loss = (A * sum_probs).sum(-1)
    loss = loss.mean(-1)

    return loss


def reinforce_loss_entropy_regularization(
    cfg: Config, scores: Tensor, log_probs: Tensor, entropy_coef: float = 0.1
):
    # A = scores - scores.mean(-1, keepdim=True)
    A = (scores - scores.mean(-1, keepdim=True)) / (scores.std(-1, keepdim=True) + 1e-9)

    sum_probs = log_probs.sum(-1)
    # sum_probs = (sum_probs - sum_probs.mean()) / (sum_probs.std() + 1e-9)

    reinforce_loss = (A * sum_probs).sum(-1)

    entropy_loss = -(log_probs.exp() * log_probs).sum(-1)
    loss = reinforce_loss.mean(-1) + entropy_coef * entropy_loss.mean(-1)

    return loss


def train_step_manual(loss: Tensor, heuristic: Tensor, lr: float):
    loss.backward()
    heuristic.data += lr * heuristic.grad
    heuristic.grad = None


def train_reinforce(cfg: Config, graphs: TorchGraph, heuristic: Tensor, lr: float):
    output = generate_paths(cfg, graphs, heuristic)
    scores = reward_failure_scoring_fn(output, cfg.p_f)
    # loss = reinforce_loss(cfg, scores, output.log_probs)
    loss = reinforce_loss_entropy_regularization(cfg, scores, output.log_probs)
    train_step_manual(loss, heuristic, lr)

    return loss, scores


def lr_scheduler(epoch, base_lr=0.1, decay=0.5, step_size=5):
    return base_lr * (decay ** (epoch // step_size))


def ppo_loss(
    cfg: Config, pi_old: Tensor, pi_current: Tensor, scores: Tensor, clip: float
):
    A = (scores - scores.mean(-1, keepdim=True)) / scores.std(-1, keepdim=True)

    # 1. I think
    ratio = pi_current.sum(-1) / pi_old.sum(-1)
    # 2. This guy thinks
    # ratio = torch.exp(pi_current.sum(-1) - pi_old.sum(-1))

    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * A

    # 1. I think
    loss = (-torch.min(surr1, surr2)).sum(-1)
    # 2. This guy thinks
    # loss = (-torch.min(surr1, surr2)).mean(-1)

    loss = loss.mean(-1)

    return loss


def train_ppo(
    cfg: Config,
    graphs: TorchGraph,
    heuristic: Tensor,
    lr: float,
    clip: float = 0.2,
    n_updates: int = 2,
):
    output = generate_paths(cfg, graphs, heuristic)
    scores = reward_failure_scoring_fn(output, cfg.p_f)

    # Old policy
    pi_old = output.log_probs.clone().detach() + 1e-5

    for _ in range(n_updates):
        pi_current = get_current_log_probs(output, heuristic)
        loss = ppo_loss(cfg, pi_old, pi_current, scores, clip)
        train_step_manual(loss, heuristic, lr)

    return loss, scores


# TODO: Make some kind of wrapper to automatically iterate over paths
def get_current_log_probs(output: RolloutOutput, heuristic: Tensor):
    batch_size, num_rollouts, max_length = output.path.nodes.shape

    # Batch and sim indices
    b_indices = (
        torch.arange(batch_size).unsqueeze(-1).expand((-1, num_rollouts)).flatten()
    )
    s_indices = torch.arange(batch_size * num_rollouts)
    output = output.flatten()

    # Log Prob buffer
    current_log_probs = torch.zeros_like(output.log_probs)

    path_index = 1
    while path_index < max_length:
        current_node = output.path.nodes[s_indices, path_index]

        is_continuing = current_node != -1
        b_indices, s_indices = b_indices[is_continuing], s_indices[is_continuing]
        if s_indices.numel() == 0:
            break

        current_node = current_node[is_continuing]
        prev_node = output.path.nodes[s_indices, path_index - 1]

        scores = heuristic[b_indices, prev_node]
        dist = torch.distributions.Categorical(logits=scores)
        log_prob = dist.log_prob(current_node)
        current_log_probs[s_indices, path_index - 1] = log_prob

        path_index += 1

    # Reshape and return
    return current_log_probs.reshape(batch_size, num_rollouts, -1)


def test_heuristic(
    cfg: Config,
    graphs: TorchGraph,
    heuristic: Tensor,
    index: int = 0,
):
    graph = graphs[index]
    heuristic = heuristic[index]

    # Expand graph
    graph = graph.unsqueeze(0).expand(cfg.num_runs)
    heuristic = heuristic.unsqueeze(0).expand(
        (cfg.num_runs, cfg.num_nodes, cfg.num_nodes)
    )

    paths, is_success = sop_aco_solver(
        graph, heuristic, cfg.num_rollouts, cfg.num_iterations, cfg.p_f, cfg.kappa
    )

    # Statistics
    reward = paths.reward.sum(-1)
    avg_reward = reward.sum(-1) / cfg.num_runs
    min_reward = torch.min(reward)
    max_reward = torch.max(reward)

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
    print(f"Failure Prob: {1 - (is_success.sum(-1) / cfg.num_runs)}")

    return paths


@hydra.main(version_base=None, config_name="aco_sop2")
def main(cfg: Config) -> None:
    torch.set_default_device(cfg.device)

    # Directories
    viz_prefix = create_viz_prefix(cfg)

    # Generate Data
    print("Loading Dataset...")
    graphs = load_dataset(cfg)

    # Create heuristic
    heuristic = torch.ones((cfg.batch_size, cfg.num_nodes, cfg.num_nodes))
    heuristic = mcts_sopcc_heuristic(graphs.nodes["reward"], graphs.edges["samples"])
    heuristic.requires_grad_(True)

    coded_heuristic = mcts_sopcc_heuristic(
        graphs.nodes["reward"], graphs.edges["samples"]
    )
    old_heuristic = heuristic.clone().detach()

    # Test
    print("Testing ACO Coded")
    paths = test_heuristic(cfg, graphs, coded_heuristic, index=0)

    print("Testing Random Heuristic")
    paths = test_heuristic(cfg, graphs, old_heuristic, index=0)

    # Generate Paths w/ Heuristic
    print("Training...")
    pbar = tqdm()
    base_lr = cfg.lr
    for i in range(1000):
        lr = base_lr
        loss, scores = train_reinforce(cfg, graphs, heuristic, lr)
        # loss, scores = train_ppo(cfg, graphs, heuristic, lr, clip=0.9, n_updates=10)

        pbar.set_description(
            f"[{i}] Loss: {float(loss)}, Avg Scores: {float(scores.mean(-1).mean(-1))}"
        )
        pbar.update()
    pbar.close()

    # Test
    print("Testing ACO after")
    paths = test_heuristic(cfg, graphs, heuristic, index=0)

    # Visualize
    print("Visualizing...")
    save_path(cfg, paths, graphs, index=0, label="ACO", prefix=viz_prefix)
    plot_heuristics(
        heuristics=[old_heuristic[0], heuristic[0].detach(), coded_heuristic[0]],
        titles=["Old_H", "New_H", "SOPCC_H"],
        out_path=viz_prefix + "_ACO_heatmap",
        cols=3,
    )


if __name__ == "__main__":
    main()
