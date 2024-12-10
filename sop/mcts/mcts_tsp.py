from typing import ClassVar
import time
import torch
import torch.nn.functional as F
from tensordict import tensorclass, TensorDict

from sop.utils.graph_torch import TorchGraph
from sop.utils.path2 import Path


def run_tsp_solver(
    graph: TorchGraph,
    start_graph_node: torch.Tensor,
    num_simulations: int,
    device: str = "cpu",
):
    batch_size, num_nodes = graph.size()

    # Loop State
    current_graph_node = start_graph_node.clone()
    indices = torch.arange(batch_size)

    # Create path buffer
    path = Path.empty(batch_size, num_nodes, device)
    path.append(
        indices,
        current_graph_node,
        cost=torch.zeros(batch_size, device=device),
    )
    for step in range(num_nodes + 1):
        if step < num_nodes:
            next_graph_node = MCTS_TSP(
                graph,
                path,
                current_graph_node,
                start_graph_node,
                num_simulations,
            )
        else:
            next_graph_node = start_graph_node

        cost = graph.edge_matrix[indices, current_graph_node, next_graph_node]
        path.append(indices, next_graph_node, cost)

        current_graph_node = next_graph_node

    return path


def MCTS_TSP(
    graph: TorchGraph,
    current_path: Path,
    current_graph_node: torch.Tensor,
    goal_graph_node: torch.Tensor,
    num_simulations: int,
    z: float = 0.1,
):
    tree = Tree.instantiate_from_root(current_graph_node, graph, num_simulations)

    for k in range(num_simulations):
        # start = time.time()
        parent_tree_node, parent_graph_node, tree_path, is_expanded, depth = select_gnn(
            tree, graph, current_path, z=z
        )
        # print(f"Select: {time.time() - start}")
        # start = time.time()
        Q = policy_gnn(graph, tree_path, parent_graph_node, goal_graph_node)
        # print(f"Policy: {time.time() - start}")
        # start = time.time()
        expand_gnn(tree, tree_path, parent_tree_node, Q, is_expanded, depth)
        # print(f"Expand: {time.time() - start}")
        # start = time.time()
        backup_gnn(tree, tree_path, graph, parent_tree_node)
        # print(f"Backup: {time.time() - start}")

    return select_action(tree, tree_path)


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
        max_nodes = (batch_size, (num_simulations * num_nodes) + 1)
        max_children = (batch_size, (num_simulations * num_nodes) + 1, num_nodes)

        def full(size: torch.Tensor, value: int, dtype: torch.dtype = torch.long):
            return torch.full(size, fill_value=value, dtype=dtype, device=device)

        def zeros(size: torch.Tensor, dtype: torch.dtype = torch.float32):
            return torch.zeros(size, dtype=dtype, device=device)

        tree = cls(
            node_mapping=full(max_nodes, UNVISITED),
            raw_values=full(max_nodes, torch.inf, dtype=torch.float32),
            node_values=full(max_nodes, torch.inf, dtype=torch.float32),
            node_visits=zeros(max_nodes),
            parents=full(max_nodes, NO_PARENT),
            neighbor_from_parent=full(max_nodes, UNVISITED),
            children_index=full(max_children, UNVISITED),
            children_values=full(max_children, torch.inf, dtype=torch.float32),
            children_visits=zeros(max_children),
            num_nodes=torch.ones((batch_size,), dtype=torch.long, device=device),
            batch_size=[batch_size],
            device=device,
        )

        tree.node_mapping[:, ROOT_INDEX] = root_graph_node

        return tree

    def size(self):
        # [batch_size, num_tree_nodes, num_graph_nodes]
        return self.children_index.shape


# TODO: Figure out how to get rid of depth
def update_visit_count(
    indices: torch.Tensor,
    tree: Tree,
    current_tree_node: torch.Tensor,
    depth: torch.Tensor,
    value: int = 1,
):
    tree.node_visits[indices, current_tree_node] += value

    # Update parent stats as well
    has_parent = depth != 0
    indices = indices[has_parent]
    current_tree_node = current_tree_node[has_parent]
    if current_tree_node.numel() > 0:
        parent = tree.parents[indices, current_tree_node]
        child_graph_node = tree.neighbor_from_parent[indices, current_tree_node]
        tree.children_visits[indices, parent, child_graph_node] += value


def compute_uct(
    indices: torch.Tensor, tree: Tree, current_tree_node: torch.Tensor, z: float
):
    Q = tree.children_values[indices, current_tree_node]
    N = tree.children_visits[indices, current_tree_node]
    t = tree.node_visits[indices, ROOT_INDEX]

    return Q - z * torch.sqrt(torch.log(t + 1e-5).unsqueeze(-1) / N)


def select_gnn(tree: Tree, graph: TorchGraph, current_path: Path, z: float = 0.1):
    """Select function using tree policy to pick best node to expand.

    In regular MCTS select, you choose one leaf node to expand,
    but because the GNN gives a value for every child node,
    we instead need to choose a parent node to expand all its children."""
    batch_size, num_nodes = graph.size()
    batch_shape = (batch_size,)
    device = graph.device

    # need copied path for simulation
    tree_path = current_path.clone()

    # Loop State
    indices = torch.arange(batch_size, device=device)
    parent_tree_node = torch.full(batch_shape, ROOT_INDEX, device=device)
    parent_graph_node = tree.node_mapping[indices, ROOT_INDEX]
    depth = torch.zeros(batch_shape, dtype=torch.int32, device=device)
    is_expanded = torch.zeros(batch_shape, dtype=torch.bool, device=device)

    while indices.numel() > 0:
        # Get active state
        current_tree_node = parent_tree_node[indices]
        active_depth = depth[indices]

        # Compute Scores
        update_visit_count(indices, tree, current_tree_node, active_depth)
        scores = compute_uct(indices, tree, current_tree_node, z)
        scores = torch.masked_fill(scores, tree_path.mask[indices], torch.inf)

        # Choose best next node
        next_graph_node = torch.argmin(scores, axis=-1)
        next_tree_node = tree.children_index[
            indices, current_tree_node, next_graph_node
        ]

        # From this point there are two cases:
        # 1. next node is a leaf -> return parent node
        # 2. next node has been expanded -> continue with next node
        is_visited = next_tree_node != UNVISITED
        is_path_not_complete = tree_path.length[indices] < num_nodes
        is_continuing = torch.logical_and(is_visited, is_path_not_complete)

        # Update global is_leaf
        is_expanded[indices] = is_visited

        # Mask finished indices
        indices = indices[is_continuing]

        if indices.numel() > 0:
            next_graph_node = next_graph_node[is_continuing]
            next_tree_node = next_tree_node[is_continuing]

            # Add to path
            current_graph_node = parent_graph_node[indices]
            cost = graph.edge_matrix[indices, current_graph_node, next_graph_node]
            tree_path.append(indices, next_graph_node, cost)

            # Update Loop State
            depth[indices] += 1
            parent_tree_node[indices] = next_tree_node
            parent_graph_node[indices] = next_graph_node

    return parent_tree_node, parent_graph_node, tree_path, is_expanded, depth


def preprocess_features(
    graph: TorchGraph, tree_path: Path, current_graph_node: torch.Tensor
):
    batch_size, _ = graph.size()

    # -- Node features
    position = graph.nodes["position"]  # [B, N, 2]
    reward = graph.nodes["reward"]  # [B, N]

    visited = tree_path.mask.long()
    visited_one_hot = F.one_hot(visited, num_classes=2)  # [B, N, 2]

    current = torch.zeros_like(visited)
    current[torch.arange(batch_size), current_graph_node] = 1
    current_one_hot = F.one_hot(current, num_classes=2)  # [B, N, 2]

    node_features = torch.cat(
        [position, reward.unsqueeze(-1), visited_one_hot, current_one_hot], dim=-1
    )

    # -- Edge features
    edge_matrix = graph.edge_matrix

    return node_features, edge_matrix


def policy_gnn(
    graph: TorchGraph,
    tree_path: Path,
    current_graph_node: torch.Tensor,
    goal_graph_node: torch.Tensor,
):
    node_features, edge_matrix = preprocess_features(
        graph, tree_path, current_graph_node
    )

    # TODO: Make the GNN policy lmao...
    Q = torch.rand(graph.size(), device=graph.device)

    return Q


def add_node(
    indices: torch.Tensor,
    tree: Tree,
    parent_tree_node: torch.Tensor,
    new_graph_node: torch.Tensor,
    Q: torch.Tensor,
):
    nt = tree.num_nodes[indices]  # new tree node
    pt = parent_tree_node[indices]
    ng = new_graph_node
    q = Q[indices, new_graph_node]

    tree.node_mapping[indices, nt] = ng
    tree.raw_values[indices, nt] = q
    tree.node_visits[indices, nt] = 1
    tree.neighbor_from_parent[indices, nt] = ng
    tree.parents[indices, nt] = pt

    tree.children_index[indices, pt, ng] = nt
    tree.children_values[indices, pt, ng] = q
    tree.children_visits[indices, pt, ng] = 1

    tree.num_nodes[indices] += 1


def update_node(
    indices: torch.Tensor,
    tree: Tree,
    parent_tree_node: torch.Tensor,
    graph_node: torch.Tensor,
    Q: torch.Tensor,
):
    pt = parent_tree_node[indices]
    g = graph_node
    q = Q[indices, graph_node]
    ct = tree.children_index[indices, pt, g]  # child tree node

    # TODO: Create running average instead of replacing
    tree.raw_values[indices, ct] = q
    tree.node_visits[indices, ct] += 1
    tree.children_values[indices, pt, g] = q
    tree.children_visits[indices, pt, g] += 1


# TODO: This scales linearly with num nodes. Figure out how to batch this.
# TODO: Make sure the count is updated correctly
def expand_gnn(
    tree: Tree,
    tree_path: Path,
    parent_tree_node: torch.Tensor,
    Q: torch.Tensor,
    is_expanded: torch.Tensor,
    depth: torch.Tensor,
):
    batch_size, _, num_graph_nodes = tree.size()
    indices = torch.arange(batch_size)

    mask = ~tree_path.mask
    are_leaves = torch.logical_and(mask, ~is_expanded.unsqueeze(-1))
    are_expanded = torch.logical_and(mask, is_expanded.unsqueeze(-1))

    # Add each valid node to the graph
    for graph_node in range(num_graph_nodes):
        leaf_mask = are_leaves[indices, graph_node]
        expanded_mask = are_expanded[indices, graph_node]

        # Case 1. Parent node is leaf
        leaf_i = indices[leaf_mask]
        if leaf_i.numel() > 0:
            add_node(
                indices=leaf_i,
                tree=tree,
                parent_tree_node=parent_tree_node,
                new_graph_node=torch.full_like(leaf_i, graph_node),
                Q=Q,
            )

        # Case 2. Parent node has been expanded
        expanded_i = indices[expanded_mask]
        if expanded_i.numel() > 0:
            update_node(
                indices=expanded_i,
                tree=tree,
                parent_tree_node=parent_tree_node,
                graph_node=torch.full_like(expanded_i, graph_node),
                Q=Q,
            )

    # Increment parent node visit counts
    num_visits = torch.sum(mask, axis=-1)
    update_visit_count(indices, tree, parent_tree_node, depth, num_visits)


def backup_gnn(
    tree: Tree,
    tree_path: Path,
    graph: TorchGraph,
    current_tree_node: torch.Tensor,
):
    batch_size, _, num_graph_nodes = tree.size()
    indices = torch.arange(batch_size)

    # -- Get values
    parent_Q = tree.raw_values[indices, current_tree_node]
    children_Q = tree.children_values[indices, current_tree_node]

    # -- Find smallest cost child
    masked_Q = torch.masked_fill(children_Q, tree_path.mask, torch.inf)
    min_child = torch.argmin(masked_Q, axis=-1)
    min_Q = tree.children_values[indices, current_tree_node, min_child]

    # -- Loop State
    parent_tree_node = current_tree_node
    parent_graph_node = tree.node_mapping[indices, parent_tree_node]
    parent_Q = parent_Q
    sec_parent_tree_node = tree.parents[indices, parent_tree_node]
    child_graph_node = min_child
    child_Q = min_Q

    # -- Backpropogate lowest value
    while indices.numel() > 0:
        # Calculate new estimate
        distance = graph.edge_matrix[indices, parent_graph_node, child_graph_node]
        new_estimate = child_Q + distance

        # If the predicted is a better value
        is_lower = new_estimate < parent_Q
        has_parent = sec_parent_tree_node != NO_PARENT
        is_continuing = torch.logical_and(is_lower, has_parent)

        # Update parent regardless of continuing
        lower_i = indices[is_lower]
        if lower_i.numel() > 0:
            tree.node_values[lower_i, parent_tree_node[is_lower]] = new_estimate[
                is_lower
            ]

        indices = indices[is_continuing]
        if indices.numel() > 0:
            # Mask state
            parent_tree_node = parent_tree_node[is_continuing]
            parent_graph_node = parent_graph_node[is_continuing]
            sec_parent_tree_node = sec_parent_tree_node[is_continuing]
            new_estimate = new_estimate[is_continuing]

            # Update second parent
            tree.children_values[indices, sec_parent_tree_node, parent_graph_node] = (
                new_estimate
            )

            # Update Loop State
            parent_tree_node = sec_parent_tree_node
            parent_Q = tree.raw_values[indices, parent_tree_node]
            sec_parent_tree_node = tree.parents[indices, sec_parent_tree_node]
            child_graph_node = parent_graph_node
            child_Q = new_estimate


def select_action(tree: Tree, tree_path: Path):
    scores = tree.children_values[:, ROOT_INDEX]
    scores = torch.masked_fill(scores, tree_path.mask, torch.inf)
    next_graph_node = torch.argmin(scores, axis=-1)

    return next_graph_node
