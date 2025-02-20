from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import os

import hydra
from hydra.core.config_store import ConfigStore
import torch
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sop.utils.graph_torch import (
    TorchGraph,
    generate_random_graph_batch,
    generate_sop_graphs,
)
from sop.utils.sample import sample_costs
from sop.inference.milp import sop_milp_solver
from sop.utils.visualization import plot_solutions
from sop.utils.path import Path
from sop.utils.seed import set_seed, random_seed

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# -- Config
@dataclass
class Config:
    # Dataset
    dataset_dir: str = "data"
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
    z: float = 0.1


cs = ConfigStore.instance()
cs.store(name="generate_dataset", node=Config)


# -- Main Script
@hydra.main(version_base=None, config_name="generate_dataset")
def main(cfg: Config) -> None:
    # -- Device selection
    print(f"Using device {DEVICE}...")
    torch.set_default_device(DEVICE)

    # -- Creating data folder
    print(f"Creating data folder {cfg.dataset_dir}...")
    os.makedirs(cfg.dataset_dir, exist_ok=True)

    # -- Set seed
    if cfg.seed is None:
        cfg.seed = random_seed()
    set_seed(cfg.seed)

    # -- Generate Data
    print("Generating graphs...")
    graphs: TorchGraph = generate_sop_graphs(
        cfg.batch_size,
        cfg.num_nodes,
        cfg.start_node,
        cfg.goal_node,
        cfg.budget,
        cfg.num_samples,
        cfg.kappa,
    )
    # NOTE: current format of graph dataset path is <time>_dataset_<batch_size>_<graph_size>
    # TODO: if we begin to have multiple experiments of the same time, we can add more flags to the path
    graph_path: str = (
        cfg.dataset_dir
        + "/"
        + str(cfg.seed)
        + "_"
        + "graphs"
        + "_"
        + str(cfg.batch_size)
        + "_"
        + str(cfg.num_nodes)
    )
    graphs.export(graph_path)
    print(f"Graphs Tensors generated at {graph_path}...")


if __name__ == "__main__":
    main()
