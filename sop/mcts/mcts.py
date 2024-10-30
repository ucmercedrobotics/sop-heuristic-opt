from dataclasses import dataclass
import torch
from torch_geometric.data import Data, Batch

from sop.mcts.tree import Tree


def run(
    graph: Data,
    start_node: torch.Tensor,
    end_node: torch.Tensor,
    budget: torch.Tensor,
    num_simulations: int,
):
    """Main SOP function.
    Takes in original Graph, starting node id, ending node id, and budget.
    """
    current_node = start_node

    # the first action
    # while budget > 0 and current_node != end_node:
    # {
    result = MCTS_SOPCC(graph, current_node, budget, num_simulations)
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
    graph: Data, current_node: torch.Tensor, budget: torch.Tensor, num_simulations: int
):
    # Create new MCTS state
    # Loop for K iterations
    #   select new child vertex with UCTF(v)
    #   for S iterations do
    #       t = SampleTraverseTime(v, v_j)
    #       B' = B - t
    #       path <- rollout(v_j, B')
    #   compute Q[v_j] and F[v_j] based on the S rollouts

    # Create new MCTS tree
    tree = instantiate_tree_from_root(current_node, graph, num_simulations)
    # For K iterations
    # Select new child vertex with UCTF
    next_node, next_neighbor_index = simulate(tree, max_depth=num_simulations)
    print(next_node)
    print(next_neighbor_index)
    #   for S iterations do
    #       t = SampleTraverseTime(v, v_j)
    # TODO: Compute Q[v_j] and F[v_j] based on the S rollouts
    return None


def instantiate_tree_from_root(
    root_index: torch.Tensor, graph: Batch, num_simulations: int
):
    # Infer shape
    batch_size = graph.batch_size
    num_nodes = graph.num_nodes // batch_size

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


@dataclass
class _SimulationState:
    node_index: torch.Tensor  # [B,]
    next_neighbor_index: torch.Tensor  # [B,]


# TODO: Add contiguous?
def simulate(tree: Tree, max_depth: int, z: float = 0.1) -> _SimulationState:
    """Select the next leaf node to expand using the tree policy."""
    # TODO: Make this cleaner

    batch_size = tree.infer_batch_size()
    batch_shape = (batch_size,)

    root_node = torch.full(batch_shape, Tree.ROOT_INDEX)  # [B,]
    indices = torch.arange(batch_size)  # [B,]
    depth = torch.zeros(batch_shape)  # [B,]

    state = _SimulationState(
        node_index=root_node,
        next_neighbor_index=torch.full(batch_shape, Tree.UNVISITED),
    )

    while indices.numel() > 0:
        # Only worry about continuing nodes
        current_node = state.node_index[indices]  # [I,]

        increase_visit_count(tree, indices, current_node, depth)

        # Compute the uctf score and choose the next best neigbhbor to traverse.
        scores = compute_uctf(tree, indices, current_node, z)  # [I, A]
        next_neighbor_index = torch.argmax(scores, axis=-1)  # [I,]
        next_neighbor = tree.children_index[indices, current_node, next_neighbor_index]

        # Add best neighbors for current_nodes
        state.next_neighbor_index[indices] = next_neighbor_index

        # From this point there are two cases:
        # 1. node is a leaf -> return and expand node
        # 2. node has been expanded -> traverse find best neighbor
        depth[indices] += 1
        is_visited = next_neighbor != Tree.UNVISITED
        is_before_depth_cutoff = depth < max_depth
        is_continuing = torch.logical_and(is_visited, is_before_depth_cutoff)

        # mask out indices and add expanded next_neighbors to state
        indices = indices[is_continuing]  # [I',]
        state.node_index[indices] = next_neighbor[is_continuing]

    return state.node_index, state.next_neighbor_index
