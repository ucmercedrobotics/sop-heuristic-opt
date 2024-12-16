import torch
from tensordict import tensorclass

UNVISITED = -1


@tensorclass
class Path:
    nodes: torch.Tensor  # [B, num_nodes + 2]
    costs: torch.Tensor  # [B, num_nodes + 2]
    mask: torch.Tensor  # [B, num_nodes]
    length: torch.Tensor  # [B,]

    @classmethod
    def empty(cls, batch_size: int, num_nodes: int, device: str = "cpu"):
        max_size = (batch_size, num_nodes + 1)
        mask_size = (batch_size, num_nodes)
        return cls(
            nodes=torch.full(max_size, UNVISITED, dtype=torch.long, device=device),
            costs=torch.zeros(max_size, dtype=torch.float32, device=device),
            mask=torch.zeros(mask_size, dtype=torch.bool, device=device),
            length=torch.zeros((batch_size,), dtype=torch.long, device=device),
            batch_size=[batch_size],
            device=device,
        )

    def size(self):
        return self.nodes.shape

    def append(self, indices: torch.Tensor, node: torch.Tensor, cost: torch.Tensor):
        index = self.length[indices]
        self.nodes[indices, index] = node
        self.costs[indices, index] = cost
        self.mask[indices, node] = 1
        self.length[indices] += 1

    def pop(self, indices):
        self.length -= 1

        # pop values
        node = self.nodes[indices, self.length]
        cost = self.costs[indices, self.length]

        # remove values
        self.nodes[indices, self.length] = UNVISITED
        self.costs[indices, self.length] = 0
        self.mask[indices, node] = 0

        return node, cost
