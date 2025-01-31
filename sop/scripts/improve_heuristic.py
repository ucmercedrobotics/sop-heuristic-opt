from typing import Optional
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
from sop.mcts.sop2 import (
    sop_mcts_aco_solver,
    random_heuristic,
    mcts_sopcc_heuristic,
)
from sop.milp.pulp_milp_sop import sop_milp_solver
from sop.utils.visualization import plot_solutions, plot_heuristics
from sop.utils.path import Path, evaluate_path, path_to_heatmap
from sop.utils.seed import set_seed, random_seed

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# -- Config
@dataclass
class Config:
    # Data
    dataset_dir: str = "data"
    visual_dir: str = "viz"
    seed: Optional[int] = None
    # Batch
    batch_size: int = 8
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
    num_simulations: int = 100
    z: float = 0.15
    epsilon: float = 0.1
    p_f: float = 0.1


cs = ConfigStore.instance()
cs.store(name="improve_heuristic", node=Config)


# -- Main Script
@hydra.main(version_base=None, config_name="improve_heuristic")
def main(cfg: Config) -> None:
    torch.set_default_device(cfg.device)

    # -- Set seed
    if cfg.seed is None:
        cfg.seed = random_seed()

    # -- Generate Data
    # TODO: if file exists, import
    # TODO: if we begin to have multiple experiments of the same time, we can add more flags to the path
    expected_graph_tensor_path = (
        cfg.dataset_dir
        + "/"
        + str(cfg.seed)
        + "_graphs_"
        + str(cfg.batch_size)
        + "_"
        + str(cfg.num_nodes)
    )
    if os.path.isfile(expected_graph_tensor_path):
        print(f"Loading graphs from {expected_graph_tensor_path}...")
        graphs = TorchGraph.load(expected_graph_tensor_path)
    else:
        print(f"Generating {cfg.batch_size} graphs...")
        graphs = generate_sop_graphs(
            cfg.batch_size,
            cfg.num_nodes,
            cfg.start_node,
            cfg.goal_node,
            cfg.budget,
            cfg.num_samples,
            cfg.kappa,
        )

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
    print("Generating MCTS+ACO Paths...")
    aco_paths, is_success, new_P = sop_mcts_aco_solver(
        graph=graphs,
        heuristic=computed_H,
        num_simulations=cfg.num_simulations,
        num_rollouts=cfg.num_samples,
        z=cfg.z,
        p_f=cfg.p_f,
        epsilon=cfg.epsilon,
        kappa=cfg.kappa,
    )
    failure_prob, avg_cost = evaluate_path(
        aco_paths[0].unsqueeze(0),
        graphs[0].unsqueeze(0),
        cfg.num_samples,
        cfg.kappa,
    )
    aco_info = (
        "MCTS+ACO; "
        + f"R: {aco_paths.reward.sum(-1)[0]:.5f}, "
        + f"B: {cfg.budget}, "
        + f"C: {float(avg_cost):.5f}, "
        + f"F: {float(failure_prob):.3f} "
        + f"N: {int(aco_paths.length[0])}"
    )
    print(aco_info)
    # new_H = random_H

    # -- Test against MILP
    # milp_path = sop_milp_solver(
    #     graphs[0].cpu(), time_limit=180, num_samples=cfg.num_samples
    # )
    # failure_prob, avg_cost = evaluate_path(
    #     milp_path,
    #     graphs[0].unsqueeze(0),
    #     cfg.num_samples,
    #     cfg.kappa,
    # )
    # milp_info = (
    #     "MILP; "
    #     + f"R: {milp_path.reward.sum(-1)[0]:.5f}, "
    #     + f"B: {cfg.budget}, "
    #     + f"C: {float(avg_cost):.5f}, "
    #     + f"F: {float(failure_prob):.3f} "
    #     + f"N: {int(milp_path.length[0])}"
    # )
    # print(milp_info)

    viz_path = cfg.dataset_dir + "/" + cfg.visual_dir
    print(f"Creating vizualization folder {viz_path}...")
    os.makedirs(viz_path, exist_ok=True)
    viz_prefix = f"{viz_path}/{cfg.seed}_{cfg.batch_size}_{cfg.num_nodes}"

    plot_solutions(
        graphs[0].cpu(),
        paths=[
            aco_paths[0],
            # milp_path[0],
        ],
        titles=[
            aco_info,
            # milp_info,
        ],
        out_path=viz_prefix + "_aco",
        rows=1,
        cols=1,
    )

    plot_heuristics(
        heuristics=[
            computed_H[0],
            new_P[0],
            path_to_heatmap(aco_paths)[0],
        ],
        titles=["Random_H", "New_H", "Path_H"],
        out_path=viz_prefix + "_aco_heatmap",
        rows=1,
        cols=3,
    )


if __name__ == "__main__":
    main()
