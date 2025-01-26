from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore

import time
import torch

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sop.utils.graph_torch import TorchGraph, generate_random_graph_batch
from sop.utils.sample import sample_costs
from sop.mcts.sop2 import sop_mcts_solver
from sop.milp.pulp_milp_sop import sop_milp_solver


# -- Config
@dataclass
class Config:
    # Batch
    batch_size: int = 1024
    device: str = "cpu"
    # Graph
    num_nodes: int = 100
    budget: int = 4
    start_node: int = 1
    goal_node: int = 2
    # Sampling
    num_samples: int = 100
    kappa: float = 0.5
    # MCTS
    num_simulations: int = 100
    z: float = 100


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
        (cfg.batch_size,), cfg.budget, device=cfg.device
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


# -- Main Script
@hydra.main(version_base=None, config_name="improve_heuristic")
def main(cfg: Config) -> None:
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
    paths, is_success = sop_mcts_solver(
        graph=graphs,
        heuristic=random_H,
        num_simulations=cfg.num_simulations,
        num_rollouts=cfg.num_samples,
        z=cfg.z,
        device=cfg.device,
    )
    print(torch.sum(is_success) / cfg.batch_size)
    print(torch.mean(paths.length.float()))
    print(paths.reward.sum(-1).mean())

    print("Generating Hardcoded Paths...")
    paths, is_success = sop_mcts_solver(
        graph=graphs,
        heuristic=computed_H,
        num_simulations=cfg.num_simulations,
        num_rollouts=cfg.num_samples,
        z=cfg.z,
        device=cfg.device,
    )

    print(torch.sum(is_success) / cfg.batch_size)
    print(torch.mean(paths.length.float()))
    print(paths.reward.sum(-1).mean())

    # -- Test against MILP
    # TODO: MILP is bugged.....
    # graph = graphs[0].squeeze()
    # edge_list = sop_milp_solver(graph, num_samples=cfg.num_samples)
    # print(edge_list)


if __name__ == "__main__":
    main()
