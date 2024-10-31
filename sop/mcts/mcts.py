import torch
from torch_geometric.data import Data, Batch

import sop.utils.graph_pyg as graph_lib
import sop.mcts.tree as tree_lib
import sop.mcts.path as path_lib

Graph = graph_lib.Graph
Tree = tree_lib.Tree
Path = path_lib.Path


def run(
    graph: Graph,
    start_node: torch.Tensor,
    end_node: torch.Tensor,
    budget: torch.Tensor,
    num_simulations: int,
):
    """Main SOP function.
    Takes in original Graph, starting node id, ending node id, and budget.
    """
    batch_size, num_nodes = graph_lib.infer_graph_shape(graph)

    # Create path buffer
    path = path_lib.create_path_from_start(graph, start_node)
    print("--here--")

    # TODO: Add indices to this function
    indices = torch.arange(batch_size)
    current_node = start_node

    # the first action
    # while budget > 0 and current_node != end_node:
    # {
    result = MCTS_SOPCC(graph, path, current_node, budget, num_simulations)
    # move to vertex next_v in graph state
    # sampled_cost = incurred cost
    # budget = budget - sampled_cost
    # current_node = next_node
    # }

    # if budget > 0:
    # return True  # Success
    # else:
    # return False  # Failure

    return None


def MCTS_SOPCC(
    graph: Graph,
    path: Path,
    current_node: torch.Tensor,
    budget: torch.Tensor,
    num_simulations: int,
):
    # Create new MCTS tree
    tree = instantiate_tree_from_root(current_node, graph, num_simulations)
    # For K iterations
    # Select new child vertex with UCTF
    parent_node, leaf_node_index, depth = simulate(tree, max_depth=num_simulations)
    path = create_path_from_root(tree, graph, parent_node, leaf_node_index, depth)
    print(path)
    ts = sample_traverse_cost(path, samples=2, kappa=0.5)
    new_budgets = budget.unsqueeze(-1) - ts
    # try on first sample before batching
    new_budgets = new_budgets[:, 0].squeeze()
    paths = batch_rollout(leaf_node_index, new_budgets)
    # TODO: Compute Q[v_j] and F[v_j] based on the S rollouts
    return None


def instantiate_tree_from_root(
    root_index: torch.Tensor, graph: Graph, num_simulations: int
):
    batch_size, num_nodes = graph_lib.infer_graph_shape(graph)

    # Need nodes for root + N simulations
    batch_node = (batch_size, num_simulations + 1)
    # Each graph is complete, so every node has N nodes as neighbors
    batch_node_neighbors = (batch_size, num_simulations + 1, num_nodes)

    # Create empty tree
    tree = Tree(
        node_mapping=torch.full(batch_node, Tree.UNVISITED),
        raw_values=torch.zeros(batch_node),
        node_values=torch.zeros(batch_node),
        failure_probs=torch.zeros(batch_node),
        node_visits=torch.zeros(batch_node),
        parents=torch.full(batch_node, Tree.NO_PARENT),
        neighbor_from_parent=torch.full(batch_node, Tree.UNVISITED),
        children_index=torch.full(batch_node_neighbors, Tree.UNVISITED),
        children_values=torch.zeros(batch_node_neighbors),
        children_failure_probs=torch.zeros(batch_node_neighbors),
        children_visits=torch.zeros(batch_node_neighbors),
    )

    # TODO: Add more values for root
    tree.node_mapping[:, Tree.ROOT_INDEX] = root_index

    return tree


def compute_uctf(tree: Tree, indices: torch.Tensor, node_index: torch.Tensor, z: float):
    """Compute UCTF for all children of a node.

    Formula: UCTF(v_j) = Q[v_j]*(1-F[v_j]) + z*sqrt(log(t)/N[v_j])
             ---------   ----------------   -------------------
            for all v_j  Expected Utility       Exploration
    """
    Q = tree.children_values[indices, node_index]
    F = tree.children_failure_probs[indices, node_index]
    N = tree.children_visits[indices, node_index]
    t = tree.node_visits[indices, node_index]

    # In the case of log(1)/0 (first simulation), the exploration term becomes sqrt(-inf).
    # This results in a nan. We add a small term to avoid this and get the desired sqrt(inf).
    return Q * (1 - F) + z * torch.sqrt(torch.log(t + 1e-5).unsqueeze(-1) / N)


def increase_visit_count(
    tree: Tree, indices: torch.Tensor, node_index: torch.Tensor, depth: torch.Tensor
):
    """Update visit counts for node and corresponding parent."""
    tree.node_visits[indices, node_index] += 1

    # Can only update nodes who have parents, so need to mask
    has_parent = depth != 0
    indices = indices[has_parent]
    node_index = node_index[has_parent]

    # Update the corresponding parent children_visits
    action = tree.neighbor_from_parent[indices, node_index]
    parent = tree.parents[indices, node_index]
    tree.children_visits[indices, parent, action] += 1


# TODO: Add contiguous?
# TODO: Add masking for same node
def simulate(
    tree: Tree, max_depth: int, z: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the next leaf node to expand using the tree policy."""
    # TODO: Make this cleaner

    batch_size, _ = tree_lib.infer_tree_shape(tree)
    batch_shape = (batch_size,)

    root_node = torch.full(batch_shape, Tree.ROOT_INDEX)  # [B,]

    # Loop State
    parent_node = root_node
    leaf_node_index = torch.full(batch_shape, Tree.UNVISITED)
    indices = torch.arange(batch_size)  # [B,]
    depth = torch.zeros(batch_shape, dtype=torch.int32)  # [B,]

    while indices.numel() > 0:
        # Only worry about continuing nodes
        current_node = parent_node[indices]  # [I,]

        increase_visit_count(tree, indices, current_node, depth)

        # Compute the uctf score and choose the next best neigbhbor to traverse.
        scores = compute_uctf(tree, indices, current_node, z)  # [I, A]
        next_neighbor_index = torch.argmax(scores, axis=-1)  # [I,]
        next_neighbor = tree.children_index[indices, current_node, next_neighbor_index]

        # Add best neighbors for current_nodes
        leaf_node_index[indices] = next_neighbor_index

        # From this point there are two cases:
        # 1. node is a leaf -> return and expand node
        # 2. node has been expanded -> traverse find best neighbor
        depth[indices] += 1
        is_visited = next_neighbor != Tree.UNVISITED
        is_before_depth_cutoff = depth < max_depth
        is_continuing = torch.logical_and(is_visited, is_before_depth_cutoff)

        # mask out indices and add expanded next_neighbors to state
        indices = indices[is_continuing]  # [I',]
        parent_node[indices] = next_neighbor[is_continuing]

    return parent_node, leaf_node_index, depth


def create_path_from_root(
    tree: Tree,
    graph: Graph,
    parent_node: torch.Tensor,
    leaf_node_index: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    """Backtrack tree to create paths for each node. Paths are in reverse and are padded with -1."""
    batch_size, _ = tree_lib.infer_tree_shape(tree)
    max_depth = torch.max(depth)

    indices = torch.arange(batch_size)
    path = path_lib.create_path_from_start(
        graph, leaf_node_index, max_length=max_depth + 1
    )

    # Traverse back to root
    current_node = parent_node
    while indices.numel() > 0:
        current_node_index = tree.node_mapping[indices, current_node]
        path_lib.add_node(path, graph, current_node_index, indices)

        # mask out nodes without parents
        parents = tree.parents[indices, current_node]
        is_not_root = parents != Tree.NO_PARENT
        indices = indices[is_not_root]
        current_node = parents[is_not_root]

    return path


def sample_cost(weight: torch.Tensor, num_samples: int = 2, kappa: float = 0.5):
    rate = 1 / ((1 - kappa) * weight)
    distribution = torch.distributions.Exponential(rate=rate)
    # (samples, batch_size) -> (batch_size, samples)
    samples = distribution.sample((num_samples,)).T
    # (batch_size, 1) + (batch_size, samples)
    return (kappa * weight).unsqueeze(-1) + samples


def sample_traverse_cost(path: Path, samples: int = 2, kappa: float = 0.5):
    batch_size, max_length = path_lib.infer_path_shape(path)
    total_sampled_cost = torch.zeros((batch_size, samples))

    indices = torch.arange(batch_size)
    path_index = 0

    while path_index < max_length:
        weight = path.dists[indices, path_index]

        # check if weight is -1
        is_continuing = weight != -1
        # mask values
        indices = indices[is_continuing]
        weight = weight[is_continuing]

        if indices.numel() <= 0:
            break

        # Sample and add to total cost
        sampled_cost = sample_cost(weight, samples, kappa)
        total_sampled_cost[indices] += sampled_cost

        path_index += 1

    return total_sampled_cost


def batch_rollout(path: Path, current_node: torch.Tensor, budget: torch.Tensor):
    print(current_node)
    print(current_node.shape)
    # current = current_node
    # path = [current_node, ...]
    # while True
    # get new
    # new = select random
    # or new = select greedy
    # if new != goal
    #   if Pr[C(current,new, vg) < B] <= P_f then
    #       append new to path
    #       B' <- B' - samplecost(current, new)
    #       current <- new
    # else
    #   append new to path
    #   return path
    ...
