from dataclasses import dataclass
import os

import torch

from sop.utils.graph import TorchGraph, generate_sop_graphs
from sop.utils.path import Path
from sop.utils.sample import set_seed, random_seed


@dataclass
class SOPConfig:
    name: str = "dataset"
    seed: int = 0
    num_graphs: int = 32
    num_nodes: int = 50
    start_node: int = 0
    goal_node: int = 49
    budget: float = 2.0
    num_samples: int = 100
    kappa: float = 0.5

    def get_name(self):
        return f"{self.name}_{self.num_graphs}_{self.num_nodes}_{self.seed}"


# TODO: Make this cleaner
class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def generate(self, cfg: SOPConfig) -> TorchGraph:
        data_path = self._is_new(cfg)
        graph_path = self._graph_path(data_path)
        print(f"Saving graphs to {graph_path} ...")

        graphs = generate_dataset(cfg)

        data = {"graphs": graphs, "config": cfg}
        torch.save(data, graph_path)
        return graphs

    def load(self, data_name: str):
        data_path = self._exists(data_name)
        graph_path = self._graph_path(data_path)
        print(f"Loading graphs from {graph_path} ...")
        data = torch.load(graph_path, weights_only=False)
        return data["graphs"], data["config"]

    def save_solutions(self, data_name: str, paths: Path, prefix: str):
        data_path = self._exists(data_name)
        sol_path = self._solution_path(data_path, prefix)
        print(f"Saving solutions to {sol_path} ...")
        torch.save(paths, sol_path)

    def load_solutions(self, data_name: str, prefix: str) -> Path:
        data_path = self._exists(data_name)
        sol_path = self._solution_path(data_path, prefix)
        print(f"Loading solutions from {sol_path} ...")
        paths = torch.load(sol_path, weights_only=False)
        return paths

    # -- Utilities
    def _get_data_path(self, cfg: SOPConfig | str) -> str:
        name = cfg if type(cfg) is str else cfg.get_name()
        return os.path.join(self.data_dir, name)

    def _exists(self, cfg: SOPConfig | str) -> str:
        data_path = self._get_data_path(cfg)
        assert os.path.exists(data_path), (
            f"Data path: {data_path} does not exist! Please `generate` the dataset."
        )
        return data_path

    def _is_new(self, cfg: SOPConfig | str) -> str:
        data_path = self._get_data_path(cfg)
        assert not os.path.exists(data_path), (
            f"Data path: {data_path} already exists! Please `load` the dataset."
        )
        os.makedirs(data_path, exist_ok=True)
        return data_path

    def _graph_path(self, data_path: str):
        return os.path.join(data_path, "graphs.pth")

    def _solution_path(self, data_path: str, prefix: str):
        return os.path.join(data_path, f"{prefix}_solutions.pth")


# -- Generation
def generate_dataset(cfg: SOPConfig) -> TorchGraph:
    set_seed(cfg.seed)
    graphs = generate_sop_graphs(
        cfg.num_graphs,
        cfg.num_nodes,
        cfg.start_node,
        cfg.goal_node,
        cfg.budget,
        cfg.num_samples,
        cfg.kappa,
    )
    set_seed(random_seed())
    return graphs
