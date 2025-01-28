from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore

import torch

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sop.utils.graph_torch import TorchGraph, generate_random_graph_batch
from sop.utils.sample import sample_costs
from sop.mcts.sop2 import sop_mcts_solver
from sop.milp.pulp_milp_sop import sop_milp_solver
from sop.utils.visualization import plot_solutions
from sop.utils.path import Path


# -- Config
@dataclass
class Config:
    # Batch
    batch_size: int = 64
    device: str = "cpu"
    # Graph
    num_nodes: int = 20
    budget: int = 2
    start_node: int = 1
    goal_node: int = 2
    # Sampling
    num_samples: int = 100
    kappa: float = 0.5
    # MCTS
    num_simulations: int = 100
    z: float = 0.1


cs = ConfigStore.instance()
cs.store(name="improve_heuristic", node=Config)


# -- Graph Generation
def generate_sop_graphs(cfg: Config) -> TorchGraph:
    graphs = generate_random_graph_batch(cfg.batch_size, cfg.num_nodes, cfg.device)
    graphs.extra["start_node"] = torch.full(
        (cfg.batch_size,), cfg.start_node, device=cfg.device
    )
    graphs.extra["goal_node"] = torch.full(
        (cfg.batch_size,), cfg.goal_node, device=cfg.device
    )
    graphs.extra["budget"] = torch.full(
        (cfg.batch_size,), cfg.budget, dtype=torch.float32, device=cfg.device
    )
    graphs.edges["samples"] = sample_costs(
        graphs.edges["distance"], cfg.num_samples, cfg.kappa, cfg.device
    )
    return graphs


# -- Heuristics
def random_heuristic(batch_size: int, num_nodes: int, device: str = "cpu"):
    return torch.rand((batch_size, num_nodes, num_nodes), device=device).softmax(-1)


def mcts_sopcc_heuristic(
    rewards: torch.Tensor, sampled_costs: torch.Tensor, device: str = "cpu"
):
    average_cost = sampled_costs.mean(dim=-1)
    return rewards.unsqueeze(-1) / average_cost


# -- Evaluate Path
def evaluate_path(
    path: Path, graph: TorchGraph, num_samples: int, kappa: float
) -> torch.Tensor:
    batch_size, max_length = path.size()
    indices = torch.arange(batch_size)

    budget = graph.extra["budget"].clone()
    current_budget = budget.unsqueeze(-1).expand((-1, num_samples))

    path_index = 1
    while path_index < max_length:
        prev_node = path.nodes[indices, path_index - 1]
        current_node = path.nodes[indices, path_index]
        weight = graph.edges["distance"][indices, prev_node, current_node]
        # samples = graph.edges["samples"][indices, prev_node, current_node]

        is_continuing = current_node != -1
        indices = indices[is_continuing]
        if indices.numel() == 0:
            break

        sampled_cost = sample_costs(weight[is_continuing], num_samples, kappa)
        current_budget[indices] -= sampled_cost
        # current_budget[indices] -= samples[is_continuing]
        path_index += 1

    return (current_budget < 0).sum(-1) / num_samples, current_budget.mean(-1)


# -- Main Script
@hydra.main(version_base=None, config_name="improve_heuristic")
def main(cfg: Config) -> None:
    torch.set_default_device(cfg.device)

    # -- Generate Data
    print("Generating graphs...")
    graphs = generate_sop_graphs(cfg)

    # -- Heuristic Creation
    # -- TODO: GNN heuristic
    # -- TODO: Gflownet heuristic
    # -- TODO: bayesian heuristic
    print("Computing Heuristics...")
    random_H = random_heuristic(cfg.batch_size, cfg.num_nodes, cfg.device)
    computed_H = mcts_sopcc_heuristic(
        graphs.nodes["reward"], graphs.edges["samples"], cfg.device
    )

    # -- MCTS Improvement
    # -- TODO: pUCT
    # -- TODO: Gumbel Muzero
    # -- TODO: Thompson Sampling
    print("Generating Random Paths...")
    random_paths, is_success = sop_mcts_solver(
        graph=graphs,
        heuristic=random_H,
        num_simulations=cfg.num_simulations,
        num_rollouts=cfg.num_samples,
        z=cfg.z,
        device=cfg.device,
    )

    print("Generating Hardcoded Paths...")
    hardcoded_paths, is_success = sop_mcts_solver(
        graph=graphs,
        heuristic=computed_H,
        num_simulations=cfg.num_simulations,
        num_rollouts=cfg.num_samples,
        z=cfg.z,
        device=cfg.device,
    )

    # -- Test against MILP
    graph = graphs[0].squeeze().cpu()
    milp_path = sop_milp_solver(graph, time_limit=180, num_samples=cfg.num_samples)

    # print(
    #     "hardcoded failure_prob",
    #     evaluate_path(
    #         hardcoded_paths[0].unsqueeze(0),
    #         graphs[0].unsqueeze(0),
    #         cfg.num_samples,
    #         cfg.kappa,
    #     ),
    # )
    # print(
    #     "random failure_prob",
    #     evaluate_path(
    #         random_paths[0].unsqueeze(0),
    #         graphs[0].unsqueeze(0),
    #         cfg.num_samples,
    #         cfg.kappa,
    #     ),
    # )
    # print(
    #     "milp failure_prob",
    #     evaluate_path(milp_path, graph.unsqueeze(0), cfg.num_samples, cfg.kappa),
    # )

    plot_solutions(
        graphs[0],
        paths=[
            random_paths[0],
            hardcoded_paths[0],
            milp_path[0],
        ],
        titles=[
            f"Random Reward: {random_paths.reward.sum(-1)[0]:.5f}",
            f"Hardcoded Reward: {hardcoded_paths.reward.sum(-1)[0]:.5f}",
            f"Milp Reward: {milp_path.reward.sum(-1)[0]:.5f}",
        ],
        rows=1,
        cols=3,
    )


if __name__ == "__main__":
    main()
