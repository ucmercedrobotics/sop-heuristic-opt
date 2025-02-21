from typing import Optional
from functools import partial
from dataclasses import dataclass
import os
from multiprocessing import Pool

import hydra
from hydra.core.config_store import ConfigStore
import torch
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sop.utils.dataset import DataLoader
from sop.utils.visualization import plot_solutions
from sop.inference.milp import solve_sop_multithread


# -- MILP will always run on CPU
DEVICE = "cpu"


# -- Config
@dataclass
class Config:
    # Data
    data_dir: str = "data"
    data_name: str = ""

    # MILP
    num_workers: int = 1
    num_samples: int = 100
    p_f: float = 0.1
    time_limit: Optional[int] = None


cs = ConfigStore.instance()
cs.store(name="generate_milp_solutions", node=Config)


@hydra.main(version_base=None, config_name="generate_milp_solutions")
def main(cfg: Config) -> None:
    torch.set_default_device(DEVICE)

    # Load Dataset
    dataloader = DataLoader(cfg.data_dir)
    graphs, sop_cfg = dataloader.load(cfg.data_name)
    print(sop_cfg)

    # TODO: Add progress persistence
    paths = solve_sop_multithread(
        graphs,
        cfg.num_workers,
        cfg.p_f,
        cfg.time_limit,
        cfg.num_samples,
        sop_cfg.kappa,
    )
    dataloader.save_solutions(cfg.data_name, paths, "milp")
    paths = dataloader.load_solutions(cfg.data_name, "milp")
    plot_solutions(graphs[0], paths=[paths[0]], titles=["milp"])


if __name__ == "__main__":
    main()
