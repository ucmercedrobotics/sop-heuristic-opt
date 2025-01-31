from typing import Optional
from typing_extensions import Self

import torch
from torch import Tensor
from tensordict import tensorclass

from sop.utils.graph_torch import TorchGraph
from sop.utils.sample import sample_costs

UNVISITED = -1


@tensorclass
class Path:
    """Path for SOP."""

    nodes: Tensor  # [B, num_nodes + 1]
    reward: Tensor  # [B, num_nodes + 1]
    mask: Tensor  # [B, num_nodes]
    length: Tensor  # [B,]

    @classmethod
    def empty(cls, batch_size: int, num_nodes: int):
        max_size = (batch_size, num_nodes + 1)
        mask_size = (batch_size, num_nodes)
        return cls(
            nodes=torch.full(max_size, UNVISITED, dtype=torch.long),
            reward=torch.zeros(max_size, dtype=torch.float32),
            mask=torch.zeros(mask_size, dtype=torch.bool),
            length=torch.zeros((batch_size,), dtype=torch.long),
            batch_size=[batch_size],
        )

    def size(self):
        return self.nodes.shape

    def append(
        self,
        indices: Tensor,
        node: Tensor,
        reward: Optional[Tensor] = None,
    ):
        index = self.length[indices]

        self.nodes[indices, index] = node

        if reward is not None:
            self.reward[indices, index] = reward

        self.mask[indices, node] = 1
        self.length[indices] += 1

    def export(self, path: str) -> None:
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> Self:
        tg = torch.load(path, weights_only=False)

        return tg


# -- Utilities
def evaluate_path(
    path: Path, graph: TorchGraph, num_samples: int, kappa: float
) -> torch.Tensor:
    batch_size, max_length = path.size()
    indices = torch.arange(batch_size)

    budget = graph.extra["budget"].clone()
    total_sampled_cost = torch.zeros((batch_size, num_samples))

    path_index = 1
    while path_index < max_length:
        prev_node = path.nodes[indices, path_index - 1]
        current_node = path.nodes[indices, path_index]
        weight = graph.edges["distance"][indices, prev_node, current_node]

        is_continuing = current_node != -1
        indices = indices[is_continuing]
        if indices.numel() == 0:
            break

        sampled_cost = sample_costs(weight[is_continuing], num_samples, kappa)
        total_sampled_cost[indices] += sampled_cost
        path_index += 1

    residual_budget = budget - total_sampled_cost

    return (residual_budget < 0).sum(-1) / num_samples, total_sampled_cost.mean(-1)


def path_to_heatmap(path: Path) -> torch.Tensor:
    batch_size, max_length = path.size()
    indices = torch.arange(batch_size)

    num_nodes = max_length - 1
    heatmap = torch.zeros((batch_size, num_nodes, num_nodes))

    path_index = 1
    while path_index < max_length:
        prev_node = path.nodes[indices, path_index - 1]
        current_node = path.nodes[indices, path_index]
        heatmap[indices, prev_node, current_node] = 1

        is_continuing = current_node != -1
        indices = indices[is_continuing]
        if indices.numel() == 0:
            break

        path_index += 1

    return heatmap
