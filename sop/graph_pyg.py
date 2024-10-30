import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T


# -- Main
def generate_random_graph(num_nodes: int) -> Data:
    """Generates a random complete graph."""
    # Create graph with only node attributes
    positions = generate_uniform_positions(size=(num_nodes,))
    rewards = generate_uniform_reward(size=(num_nodes,))
    # [2, N] to [N, 2]
    positions = positions.T

    data = Data(pos=positions)
    data.reward = rewards

    # For a normalized graph, the maximum distance is sqrt(2) (when points are (0,0) and (1,1))
    max_distance = (2**0.5) + 1e-5

    # 1. RadiusGraph to create a complete graph
    # 2. Distance to compute euclidean distance for every edge
    transform = T.Compose([T.RadiusGraph(r=max_distance), T.Distance()])
    data = transform(data)
    return data


def generate_random_graph_batch(num_nodes: int, batch_size: int) -> Batch:
    """Generates a batch of random complete graphs."""
    batch_shape = (batch_size, num_nodes)
    # Batch sample values
    positions = generate_uniform_positions(size=batch_shape)
    rewards = generate_uniform_reward(size=batch_shape)
    # (2, B, N) -> (B, N, 2)
    positions = positions.permute(1, 2, 0)

    # Create batched graphs
    data_list = []
    for i in range(batch_size):
        data = Data(pos=positions[i, :])
        data.reward = rewards[i, :]
        data_list.append(data)

    batch = Batch.from_data_list(data_list)

    # For a normalized graph, the maximum distance is sqrt(2) (when points are (0,0) and (1,1))
    max_distance = (2**0.5) + 1e-5

    # 1. RadiusGraph to create a complete graph
    # 2. Distance to compute euclidean distance for every edge
    transform = T.Compose([T.RadiusGraph(r=max_distance), T.Distance()])
    batch = transform(batch)
    return batch


# -- Position
def generate_uniform_positions(size: tuple[int]) -> torch.Tensor:
    """Creates 2D positions for each node in the graph."""
    xs = torch.rand(size)  # (...size,)
    ys = torch.rand(size)  # (...size,)
    return torch.stack([xs, ys])  # (2, size...)


# -- Reward
def generate_uniform_reward(size: tuple) -> torch.Tensor:
    """Computes random reward for each node in the graph."""
    return torch.rand(size)


if __name__ == "__main__":
    import time

    N = 100  # num_nodes
    B = 1024  # batch_size

    # Single
    _ = generate_random_graph(N)  # warmup
    start = time.time()
    G = generate_random_graph(N)
    single_time = time.time() - start
    print(G)

    # Batch
    _ = generate_random_graph_batch(N, B)  # warmup
    start = time.time()
    G = generate_random_graph_batch(N, B)
    batch_time = time.time() - start
    print(G)

    print(f"Single graph: {single_time}")
    print(f"Batch graph: {batch_time}")
    print(f"Improvement Ratio: {(single_time * B) / batch_time}")
