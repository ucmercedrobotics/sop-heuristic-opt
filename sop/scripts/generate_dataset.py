from typing import Optional
from dataclasses import dataclass
import os

import hydra
from hydra.core.config_store import ConfigStore
import torch
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sop.utils.sample import random_seed
from sop.utils.dataset import SOPConfig, DataLoader

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# -- Config
@dataclass
class Config:
    # Dataset
    data_dir: str = "data"

    # SOP Config
    name: str = "dataset"
    seed: Optional[int] = None
    num_graphs: int = 32
    num_nodes: int = 50
    start_node: Optional[int] = None
    goal_node: Optional[int] = None
    budget: float = 2.0
    num_samples: int = 100
    kappa: float = 0.5

    # MILP Config
    add_milp_solutions: bool = False
    milp_time_limit: int = 180
    milp_num_samples: int = 100
    milp_p_f: float = 0.075


cs = ConfigStore.instance()
cs.store(name="generate_dataset", node=Config)


# -- Main Script
@hydra.main(version_base=None, config_name="generate_dataset")
def main(cfg: Config) -> None:
    # -- Device selection
    print(f"Using device {DEVICE}...")
    torch.set_default_device(DEVICE)

    # -- Parse Config
    if cfg.seed is None:
        cfg.seed = random_seed()
    if cfg.start_node is None:
        cfg.start_node = 0
    if cfg.goal_node is None:
        cfg.goal_node = cfg.num_nodes - 1

    # -- Create dataset config
    sop_config = SOPConfig(
        name=cfg.name,
        seed=cfg.seed,
        num_graphs=cfg.num_graphs,
        num_nodes=cfg.num_nodes,
        start_node=cfg.start_node,
        goal_node=cfg.goal_node,
        budget=cfg.budget,
        num_samples=cfg.num_samples,
        kappa=cfg.kappa,
    )

    # -- Generate graphs
    dataloader = DataLoader(cfg.data_dir)
    dataloader.generate(sop_config)


if __name__ == "__main__":
    main()
