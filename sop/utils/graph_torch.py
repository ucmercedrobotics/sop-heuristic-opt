import torch
from tensordict import tensorclass, TensorDict


@tensorclass
class TorchGraph:
    nodes: TensorDict
    edge_matrix: torch.Tensor

    def size(self):
        "Returns size of graph as (batch_size, num_nodes)"
        return self.edge_matrix.size(0), self.edge_matrix.size(-1)


def generate_random_graph_batch(
    batch_size: int, num_nodes: int, device: str = "cpu"
) -> TorchGraph:
    # Create Nodes
    positions = generate_uniform_positions(size=(batch_size, num_nodes), device=device)
    rewards = generate_uniform_reward(size=(batch_size, num_nodes), device=device)

    # Convert from (B, 2, N) -> (B, N, 2)
    positions = positions.permute(0, 2, 1)

    nodes = TensorDict(
        {"position": positions, "reward": rewards},
        batch_size=[batch_size],
        device=device,
    )

    # Create edges
    # p=2 is L2 norm, or euclidean distance
    edge_matrices = torch.cdist(positions, positions, p=2)

    return TorchGraph(
        nodes=nodes,
        edge_matrix=edge_matrices,
        batch_size=[batch_size],
        device=device,
    )


# -- Position
def generate_uniform_positions(size: tuple[int], device: str = "cpu") -> torch.Tensor:
    """Creates 2D positions for each node in the graph."""
    xs = torch.rand(size, device=device)  # (...size,)
    ys = torch.rand(size, device=device)  # (...size,)
    return torch.stack([xs, ys], dim=-2)  # (2, size...)


# -- Reward
def generate_uniform_reward(size: tuple, device: str = "cpu") -> torch.Tensor:
    """Computes random reward for each node in the graph."""
    return torch.rand(size, device=device)


if __name__ == "__main__":
    import time

    N = 100  # num_nodes
    B = 256  # batch_size

    start = time.time()
    G = generate_random_graph_batch(B, N)
    batch_time = time.time() - start
    print(f"Time elapsed batched2: {batch_time}")
