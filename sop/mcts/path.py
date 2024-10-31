from typing import ClassVar
from dataclasses import dataclass
import torch
from sop.utils.graph_pyg import Graph, infer_graph_shape


@dataclass
class Path:
    """Stores information about paths.

    The largest possible path can be from start_node -> every other node -> to end_node.
    Therefore, the max_size of the path is num_nodes + 1, assuming start_node = end_node

    - nodes: list of node index from first visited to last
    - dists: distances of edges traversed. dists[i] maps to the edge of nodes[i] and nodes[i+1]
    - rewards: reward of node traversed. rewards[i] maps to nodes[i]
    - length: counter to keep track of where to input new node
    """

    nodes: torch.Tensor  # [B, num_nodes + 1]
    dists: torch.Tensor  # [B, num_nodes + 1]
    rewards: torch.Tensor  # [B, num_nodes + 1]
    length: torch.Tensor  # [B,]

    EMPTY: ClassVar[int] = -1


def infer_path_shape(path: Path):
    batch_size, max_length = path.nodes.shape
    return batch_size, max_length


def create_path_from_start(
    graph: Graph, start_node: torch.Tensor, max_length: int = None
):
    batch_size, num_nodes = infer_graph_shape(graph)
    max_length = max_length if max_length is not None else num_nodes + 1
    path_shape = (batch_size, max_length)

    path = Path(
        nodes=torch.full(path_shape, Path.EMPTY),
        dists=torch.full(path_shape, Path.EMPTY, dtype=torch.float32),
        rewards=torch.zeros(path_shape, dtype=torch.float32),
        length=torch.zeros((batch_size,), dtype=torch.int32),
    )

    # Add root to path
    path.nodes[:, 0] = start_node
    path.length += 1

    return path


def add_node(path: Path, graph: Graph, node: torch.Tensor, indices: torch.Tensor):
    index = path.length[indices]
    path.nodes[indices, index] = node

    # add dist using adj matrix
    from_idx = path.nodes[indices, index - 1]
    to_idx = node
    dist = graph.dists[indices, from_idx, to_idx]
    path.dists[indices, index - 1] = dist

    # add reward
    reward = graph.rewards[indices, node]
    path.rewards[indices, index] = reward

    path.length += 1
