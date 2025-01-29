from typing import Optional
import torch
from torch import Tensor
from tensordict import tensorclass

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
    def load(path: str, device) -> Self:
        tg = torch.load(path, map_location=device)

        return tg
