from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import os

import hydra
from hydra.core.config_store import ConfigStore
import torch
import rootutils
import scipy
from tensordict import TensorDict

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sop.utils.graph_torch import (
    TorchGraph,
    complete_adjacency_matrix,
    compute_distances,
)
from sop.utils.sample import sample_costs
from sop.utils.seed import set_seed, random_seed

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    # Dataset
    dataset_dir: str = "data"
    benchmark_dir: str = "data/benchmark/"
    benchmark_name: str = "att48"
    # Batch
    device: str = DEVICE
    # Graph
    budget: int = 25000
    # Sampling
    num_samples: int = 100
    kappa: float = 0.5


cs = ConfigStore.instance()
cs.store(name="benchmark", node=Config)


def get_benchmark_path(cfg: Config):
    return os.path.join(cfg.benchmark_dir, cfg.benchmark_name + ".mat")


def get_graph_path(cfg: Config): ...


def mat_to_torchgraph(cfg: Config):
    mat = scipy.io.loadmat(get_benchmark_path(cfg))

    # Nodes
    positions = torch.tensor(mat["xy"]).unsqueeze(0).float()
    rewards = torch.tensor(mat["rewards"]).squeeze(-1).unsqueeze(0).float()
    batch_size, num_nodes = rewards.shape

    nodes = TensorDict(
        {"position": positions, "reward": rewards},
        batch_size=[batch_size],
    )

    # Edges
    adj = complete_adjacency_matrix(batch_size, num_nodes)
    edge_distances = compute_distances(positions, p=2)
    samples = sample_costs(edge_distances, cfg.num_samples, cfg.kappa)

    edges = TensorDict(
        {"adj": adj, "distance": edge_distances, "samples": samples},
        batch_size=[batch_size],
    )

    # Extra
    start_node = 0
    goal_node = num_nodes - 1

    extra = TensorDict(
        {
            "start_node": torch.full((batch_size,), start_node),
            "goal_node": torch.full((batch_size,), goal_node),
            "budget": torch.full((batch_size,), cfg.budget, dtype=torch.float32),
        },
        batch_size=[batch_size],
    )

    return TorchGraph(nodes=nodes, edges=edges, extra=extra, batch_size=[batch_size])


@hydra.main(version_base=None, config_name="benchmark")
def main(cfg: Config) -> None:
    graph = mat_to_torchgraph(cfg)
    save_path = os.path.join(cfg.dataset_dir, cfg.benchmark_name)
    graph.export(save_path)
    print(f"Benchmark graph saved at '{save_path}'...")


if __name__ == "__main__":
    main()
