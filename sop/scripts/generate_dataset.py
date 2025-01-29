from dataclasses import dataclass
from datetime import datetime
import os

import hydra
from hydra.core.config_store import ConfigStore
import torch
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sop.utils.graph_torch import TorchGraph, generate_random_graph_batch
from sop.utils.sample import sample_costs
from sop.mcts.sop2 import sop_mcts_solver, random_heuristic, mcts_sopcc_heuristic
from sop.milp.pulp_milp_sop import sop_milp_solver
from sop.utils.visualization import plot_solutions
from sop.utils.path import Path

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# -- Config
@dataclass
class Config:
    # Dataset
    dataset_dir: str = "data"
    # Batch
    batch_size: int = 32
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
    z: float = 0.1


cs = ConfigStore.instance()
cs.store(name="generate_dataset", node=Config)


# -- Graph Generation
def generate_sop_graphs(cfg: Config) -> TorchGraph:
    graphs = generate_random_graph_batch(cfg.batch_size, cfg.num_nodes)
    graphs.extra["start_node"] = torch.full((cfg.batch_size,), cfg.start_node)
    graphs.extra["goal_node"] = torch.full((cfg.batch_size,), cfg.goal_node)
    graphs.extra["budget"] = torch.full(
        (cfg.batch_size,), cfg.budget, dtype=torch.float32
    )
    graphs.edges["samples"] = sample_costs(
        graphs.edges["distance"], cfg.num_samples, cfg.kappa
    )
    return graphs


# -- Main Script
@hydra.main(version_base=None, config_name="generate_dataset")
def main(cfg: Config) -> None:
    # -- Device selection
    print(f"Using device {DEVICE}...")
    torch.set_default_device(DEVICE)

    # -- Creating data folder
    print(f"Creating data folder {cfg.dataset_dir}...")
    os.makedirs(cfg.dataset_dir, exist_ok=True)

    # -- Generate Data
    print("Generating graphs...")
    graphs: TorchGraph = generate_sop_graphs(cfg)
    # get time for stamping graph export file
    current_datetime: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # NOTE: current format of path is <time>_dataset_<batch_size>_<graph_size>
    graph_path: str = (
        cfg.dataset_dir
        + "/"
        + current_datetime
        + "_"
        + "graphs"
        + "_"
        + str(cfg.batch_size)
        + "_"
        + str(cfg.num_nodes)
    )
    graphs.export(graph_path)
    print(f"Graphs generated at {graph_path}...")

    # -- Heuristic Creation
    # -- TODO: GNN heuristic
    # -- TODO: Gflownet heuristic
    # -- TODO: bayesian heuristic
    print("Computing edge utility map...")
    random_H = random_heuristic(cfg.batch_size, cfg.num_nodes)
    computed_H = mcts_sopcc_heuristic(graphs.nodes["reward"], graphs.edges["samples"])

    # -- MCTS Improvement
    # -- TODO: pUCT
    # -- TODO: Gumbel Muzero
    # -- TODO: Thompson Sampling
    print("Generating Random Heuristic Solution Paths...")
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

    # # -- Test against MILP
    graph = graphs[0].squeeze().cpu()
    milp_path = sop_milp_solver(graph, time_limit=60, num_samples=cfg.num_samples)

    viz_path: str = cfg.dataset_dir + "/viz/"
    print(f"Creating vizualization folder {viz_path}...")
    os.makedirs(viz_path, exist_ok=True)

    plot_solutions(
        viz_path + current_datetime + "_",
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
