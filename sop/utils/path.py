import torch
from torch import Tensor
from tensordict import tensorclass

UNVISITED = -1


@tensorclass
class Path:
    nodes: Tensor  # [B, num_nodes + 1]
    rewards: Tensor  # [B, num_nodes + 1]
    mask: Tensor  # [B, num_nodes]
    length: Tensor  # [B,]

    @classmethod
    def empty(cls, batch_size: int, num_nodes: int, device: str = "cpu"):
        max_size = (batch_size, num_nodes + 1)
        mask_size = (batch_size, num_nodes)
        return cls(
            nodes=torch.full(max_size, UNVISITED, dtype=torch.long, device=device),
            rewards=torch.zeros(max_size, dtype=torch.float32, device=device),
            mask=torch.zeros(mask_size, dtype=torch.bool, device=device),
            length=torch.zeros((batch_size,), dtype=torch.long, device=device),
            batch_size=[batch_size],
            device=device,
        )

    def size(self):
        return self.nodes.shape

    def append(
        self,
        indices: Tensor,
        node: Tensor,
        reward: Tensor,
    ):
        index = self.length[indices]
        self.nodes[indices, index] = node
        self.rewards[indices, index] = reward
        self.mask[indices, node] = 1
        self.length[indices] += 1


@tensorclass
class TreeStats:
    root_values: Tensor  # [B, num_nodes]
    root_qvalues: Tensor  # [B, num_nodes, num_nodes]
    length: Tensor  # [B,]

    @classmethod
    def empty(cls, batch_size: int, num_nodes: int, device: str = "cpu"):
        max_size = (batch_size, num_nodes)
        max_children_size = (batch_size, num_nodes, num_nodes)

        return cls(
            root_values=torch.full(
                max_size, -torch.inf, dtype=torch.float32, device=device
            ),
            root_qvalues=torch.full(
                max_children_size, -torch.inf, dtype=torch.float32, device=device
            ),
            length=torch.zeros((batch_size,), dtype=torch.long, device=device),
            batch_size=[batch_size],
            device=device,
        )

    def size(self):
        return self.root_qvalues.shape

    def append(self, indices: Tensor, value: Tensor, qvalues: Tensor):
        index = self.length[indices]
        self.root_values[indices, index] = value
        self.root_qvalues[indices, index] = qvalues
        self.length[indices] += 1
