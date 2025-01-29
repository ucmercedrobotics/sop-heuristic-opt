from typing_extensions import Self
import torch
from tensordict import tensorclass, TensorDict


@tensorclass
class TorchGraph:
    nodes: TensorDict
    edges: TensorDict
    extra: TensorDict

    def size(self, key="adj"):
        "Returns size of graph as (batch_size, num_nodes)"
        return self.edges[key].size(0), self.edges[key].size(-1)

    def export(self, path: str) -> None:
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> Self:
        tg = torch.load(path)

        return tg


def generate_random_graph_batch(batch_size: int, num_nodes: int) -> TorchGraph:
    # Create Nodes
    positions = generate_uniform_positions(size=(batch_size, num_nodes))
    rewards = generate_uniform_reward(size=(batch_size, num_nodes))

    # Convert from (B, 2, N) -> (B, N, 2)
    positions = positions.permute(0, 2, 1)

    # Create edges
    # p=2 is L2 norm, or euclidean distance
    adj = complete_adjacency_matrix(batch_size, num_nodes)
    edge_distances = compute_distances(positions, p=2)

    nodes = TensorDict(
        {"position": positions, "reward": rewards},
        batch_size=[batch_size],
    )

    edges = TensorDict(
        {"adj": adj, "distance": edge_distances},
        batch_size=[batch_size],
    )

    # Extra dict for information like start_node and budget
    extra = TensorDict({}, batch_size=[batch_size])

    return TorchGraph(
        nodes=nodes,
        edges=edges,
        extra=extra,
        batch_size=[batch_size],
    )


# -- Position
def generate_uniform_positions(size: tuple[int]) -> torch.Tensor:
    """Creates 2D positions for each node in the graph."""
    xs = torch.rand(size)  # (...size,)
    ys = torch.rand(size)  # (...size,)
    return torch.stack([xs, ys], dim=-2)  # (2, size...)


# -- Reward
def generate_uniform_reward(size: tuple) -> torch.Tensor:
    """Computes random reward for each node in the graph."""
    return torch.rand(size)


# -- Edges
def compute_distances(x: torch.Tensor, p: int = 2):
    return torch.cdist(x, x, p=p)


def complete_adjacency_matrix(batch_size: int, num_nodes: int):
    adj = torch.ones((num_nodes, num_nodes))
    adj.fill_diagonal_(0)
    adj = adj.unsqueeze(0).expand((batch_size, num_nodes, num_nodes))
    return adj


if __name__ == "__main__":
    import time

    N = 100  # num_nodes
    B = 1024  # batch_size

    _ = generate_random_graph_batch(B, N)

    start = time.time()
    G = generate_random_graph_batch(B, N)
    batch_time = time.time() - start
    print(f"Time elapsed batched2: {batch_time}")

    print(G.edges["adj"].shape)
