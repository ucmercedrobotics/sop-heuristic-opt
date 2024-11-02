from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T


# TODO: Make this cleaner
@dataclass
class Graph:
    data: Batch  # PyG compatible structure
    dists: torch.Tensor  # Adjacency matrix
    rewards: torch.Tensor  # rewards for each node
    positions: torch.Tensor  # x,y positions for each node


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
    transform = T.Compose([T.RadiusGraph(r=max_distance), T.Distance(norm=False)])
    data = transform(data)
    return data, rewards, positions


def generate_random_graph_batch(num_nodes: int, batch_size: int) -> Graph:
    """Generates a batch of random complete graphs."""
    # TODO: Make this cleaner..

    # Create graphs
    data_list = []
    dist_list = []
    reward_list = []
    position_list = []
    for _ in range(batch_size):
        data, rewards, positions = generate_random_graph(num_nodes)
        adj = get_adjacency_matrix(data)
        data_list.append(data)
        dist_list.append(adj)
        reward_list.append(rewards)
        position_list.append(positions)

    return Graph(
        data=Batch.from_data_list(data_list),
        dists=torch.stack(dist_list, dim=0),
        rewards=torch.stack(reward_list, dim=0),
        positions=torch.stack(position_list, dim=0),
    )


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


# -- Utilities
def get_bytes(graph: Data | Batch):
    """Returns the size of the graph(s) in bytes."""
    return sum([v.element_size() * v.numel() for k, v in graph])


def get_adjacency_matrix(data: Data) -> torch.Tensor:
    """Returns dense adjacency matrix from Data object."""
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    adj_matrix = torch.sparse_coo_tensor(
        edge_index, edge_attr.squeeze(), size=(num_nodes, num_nodes)
    )

    return adj_matrix.to_dense()


def infer_graph_shape(graph: Graph):
    batch_size = graph.data.batch_size
    num_nodes = graph.data.num_nodes // batch_size
    return (batch_size, num_nodes)


if __name__ == "__main__":
    import time

    N = 20  # num_nodes
    B = 256  # batch_size

    # Single
    _, _, _ = generate_random_graph(N)  # warmup
    start = time.time()
    G, R, P = generate_random_graph(N)
    single_time = time.time() - start
    print(f"Time elapsed single: {single_time}")

    start = time.time()
    G = generate_random_graph_batch(N, B)
    batch_time = time.time() - start
    print(f"Time elapsed batched: {batch_time}")
