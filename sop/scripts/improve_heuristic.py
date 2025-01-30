from dataclasses import dataclass
import os
from datetime import datetime

import hydra
from hydra.core.config_store import ConfigStore
import torch
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sop.utils.graph_torch import TorchGraph, generate_sop_graphs
from sop.utils.sample import sample_costs
from sop.mcts.sop2 import sop_mcts_solver, random_heuristic, mcts_sopcc_heuristic
from sop.milp.pulp_milp_sop import sop_milp_solver
from sop.utils.visualization import plot_solutions
from sop.utils.path import Path

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# -- Config
@dataclass
class Config:
    # Data
    dataset_dir: str = "data"
    visual_dir: str = "viz"
    timestamp: str = "2025-01-29_01-11-08"
    # Batch
    batch_size: int = 1
    device: str = DEVICE
    # Graph
    num_nodes: int = 20
    budget: int = 2
    start_node: int = 1
    goal_node: int = 19
    # Sampling
    num_samples: int = 100
    kappa: float = 0.5
    # MCTS
    num_simulations: int = 200
    z: float = 0.1


cs = ConfigStore.instance()
cs.store(name="improve_heuristic", node=Config)


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
    # TODO: if file exists, import
    # TODO: if we begin to have multiple experiments of the same time, we can add more flags to the path
    expected_graph_tensor_path: str = (
        cfg.dataset_dir
        + "/"
        + cfg.timestamp
        + "_graphs_"
        + str(cfg.batch_size)
        + "_"
        + str(cfg.num_nodes)
    )
    if os.path.isfile(expected_graph_tensor_path):
        print(f"Loading graphs from {expected_graph_tensor_path}...")
        graphs: TorchGraph = TorchGraph.load(expected_graph_tensor_path)
        timestamp: str = cfg.timestamp
    else:
        print(f"Generating {cfg.batch_size} graphs...")
        graphs: TorchGraph = generate_sop_graphs(
            cfg.batch_size,
            cfg.num_nodes,
            cfg.start_node,
            cfg.goal_node,
            cfg.budget,
            cfg.num_samples,
            cfg.kappa,
        )
        timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # -- Heuristic Creation
    # -- TODO: GNN heuristic
    # -- TODO: Gflownet heuristic
    # -- TODO: bayesian heuristic
    print("Computing Heuristics...")
    random_H = random_heuristic(cfg.batch_size, cfg.num_nodes)
    computed_H = mcts_sopcc_heuristic(graphs.nodes["reward"], graphs.edges["samples"])

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
    )

    print("Generating Hardcoded Paths...")
    hardcoded_paths, is_success = sop_mcts_solver(
        graph=graphs,
        heuristic=computed_H,
        num_simulations=cfg.num_simulations,
        num_rollouts=cfg.num_samples,
        z=cfg.z,
    )

    # -- Test against MILP
    graph = graphs[0].squeeze().cpu()
    milp_path = sop_milp_solver(graph, time_limit=45, num_samples=cfg.num_samples)

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

    viz_path: str = cfg.dataset_dir + "/" + cfg.visual_dir + "/"
    print(f"Creating vizualization folder {viz_path}...")
    os.makedirs(viz_path, exist_ok=True)

    plot_solutions(
        viz_path + timestamp + "_",
        graphs[0].cpu(),
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
