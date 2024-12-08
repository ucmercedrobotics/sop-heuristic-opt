from typing import ClassVar
import time
import torch
from tensordict import tensorclass, TensorDict

from sop.utils.graph_torch import TorchGraph
from sop.utils.path2 import Path

# TODO: Figure out how to add this to tree class
# Static class variables
ROOT_INDEX: int = 0
NO_PARENT: int = -1
UNVISITED: int = -1


@tensorclass
class Tree:
    node_mapping: torch.Tensor  # [B, N], index of node in original graph
    raw_values: torch.Tensor  # [B, N], raw computed value for each node
    node_values: torch.Tensor  # [B, N], cumulative search value for each node
    node_visits: torch.Tensor  # [B, N], visit counts for each node
    parents: torch.Tensor  # [B, N], node index for the parents of each node
    neighbor_from_parent: (
        torch.Tensor  # [B, N], the neighbor index to take from the parent to each node
    )
    children_index: (
        torch.Tensor  # [B, N, num_neighbors], the node index of the neighbor if visited
    )
    children_values: (
        torch.Tensor  # [B, N, num_neighbors], the value of traveling to neighbor from node
    )
    children_visits: (
        torch.Tensor
    )  # [B, N, num_nodes], the visit counts for each neighbor

    num_nodes: torch.Tensor  # [B,], amount of visited nodes in the tree

    @classmethod
    def instantiate_from_root(
        cls,
        root_graph_node: torch.Tensor,
        graph: TorchGraph,
        num_simulations: int,
        device: str = "cpu",
    ):
        batch_size, num_nodes = graph.size()
        max_nodes = (batch_size, num_simulations + 1)
        max_children = (batch_size, num_simulations + 1, num_nodes)

        def full(size: torch.Tensor, value: int, dtype: torch.dtype = torch.long):
            return torch.full(size, fill_value=value, dtype=dtype, device=device)

        def zeros(size: torch.Tensor, dtype: torch.dtype = torch.float32):
            return torch.zeros(size, dtype=dtype, device=device)

        tree = cls(
            node_mapping=full(max_nodes, UNVISITED),
            raw_values=zeros(max_nodes),
            node_values=zeros(max_nodes),
            node_visits=zeros(max_nodes),
            parents=full(max_nodes, NO_PARENT),
            neighbor_from_parent=full(max_nodes, UNVISITED),
            children_index=full(max_children, UNVISITED),
            children_values=zeros(max_children),
            children_visits=zeros(max_children),
            num_nodes=torch.ones((batch_size,), dtype=torch.long),
            batch_size=[batch_size],
            device=device,
        )

        tree.node_mapping[:, ROOT_INDEX] = root_graph_node

        return tree


def increase_visit_count(
    indices: torch.Tensor,
    tree: Tree,
    current_tree_node: torch.Tensor,
    depth: torch.Tensor,
):
    tree.node_visits[indices, current_tree_node] += 1

    # Update parent stats as well
    has_parent = depth != 0
    indices = indices[has_parent]
    current_tree_node = current_tree_node[has_parent]
    if current_tree_node.numel() > 0:
        parent = tree.parents[indices, current_tree_node]
        child_graph_node = tree.neighbor_from_parent[indices, current_tree_node]
        tree.children_visits[indices, parent, child_graph_node] += 1


def compute_uct(
    indices: torch.Tensor, tree: Tree, current_tree_node: torch.Tensor, z: float
):
    Q = tree.children_values[indices, current_tree_node]
    N = tree.children_visits[indices, current_tree_node]
    t = tree.node_visits[indices, ROOT_INDEX]

    return Q + z * torch.sqrt(torch.log(t + 1e-5).unsqueeze(-1) / N)


def select(tree: Tree, graph: TorchGraph, current_path: Path, z: float = 0.1):
    batch_size, num_nodes = graph.size()
    batch_shape = (batch_size,)
    device = graph.device

    # need copied path for simulation
    tree_path = current_path.clone()

    # Loop State
    parent_tree_node = torch.full(batch_shape, ROOT_INDEX, device=device)
    leaf_graph_node = torch.full(batch_shape, UNVISITED, device=device)
    depth = torch.zeros(batch_shape, dtype=torch.int32)
    indices = torch.arange(batch_size, device=device)

    while indices.numel() > 0:
        # Index only continuing state
        current_tree_node = parent_tree_node[indices]
        active_depth = depth[indices]

        # Compute Scores
        increase_visit_count(indices, tree, current_tree_node, active_depth)
        scores = compute_uct(indices, tree, current_tree_node, z)
        scores = torch.masked_fill(scores, tree_path.mask[indices], 0)

        # Choose best next node
        next_graph_node = torch.argmax(scores, axis=-1)
        next_tree_node = tree.children_index[
            indices, current_tree_node, next_graph_node
        ]

        # Add to path
        current_graph_node = tree.node_mapping[indices, current_tree_node]
        cost = graph.edge_matrix[indices, current_graph_node, next_graph_node]
        tree_path.append(indices, next_graph_node, cost)

        # From this point there are two cases:
        # 1. node is a leaf -> return and expand node
        # 2. node has been expanded -> traverse find best neighbor
        is_visited = next_tree_node != UNVISITED
        is_path_not_complete = tree_path.length < num_nodes
        is_continuing = torch.logical_and(is_visited, is_path_not_complete)

        # Update loop state
        active_depth += 1
        leaf_graph_node[indices] = next_graph_node
        indices = indices[is_continuing]
        parent_tree_node[is_continuing] = next_tree_node[is_continuing]

    return parent_tree_node, leaf_graph_node, depth, tree_path


def run_tsp_solver(
    graph: TorchGraph,
    start_graph_node: torch.Tensor,
    num_simulations: int,
    device: str = "cpu",
):
    batch_size, num_nodes = graph.size()

    # Loop State
    current_graph_node = start_graph_node
    indices = torch.arange(batch_size)

    # Create path buffer
    path = Path.empty(batch_size, num_nodes, device)
    path.append(
        indices,
        current_graph_node,
        cost=torch.zeros(batch_size, device=device),
    )

    while indices.numel() > 0:
        next_graph_node = MCTS_TSP(
            graph[indices], path[indices], current_graph_node[indices], num_simulations
        )

        break


def MCTS_TSP(
    graph: TorchGraph,
    current_path: Path,
    current_graph_node: torch.Tensor,
    num_simulations: int,
    z: float = 0.1,
):
    tree = Tree.instantiate_from_root(current_graph_node, graph, num_simulations)

    for k in range(num_simulations):
        start = time.time()
        nodes = select(tree, graph, current_path, z=z)
        print(f"Time elapsed select: {time.time() - start}")
        break
