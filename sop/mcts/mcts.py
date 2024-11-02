import torch
from torch_geometric.data import Data, Batch
import time

import sop.utils.graph_pyg as graph_lib
import sop.utils.path as path_lib
import sop.mcts.tree as tree_lib
import sop.utils.heuristic as heuristic_lib

Graph = graph_lib.Graph
Path = path_lib.Path
Tree = tree_lib.Tree
Heuristic = heuristic_lib.SAAHeuristic


def run(
    graph: Graph,
    start_node: torch.Tensor,
    goal_node: torch.Tensor,
    budget: torch.Tensor,
    num_simulations: int,
):
    """Main SOP function.
    Takes in original Graph, starting node id, ending node id, and budget.
    """
    batch_size, num_nodes = graph_lib.infer_graph_shape(graph)

    # Create path buffer
    path = path_lib.create_path_from_start(graph, start_node)

    # Compute heuristic
    heuristic = heuristic_lib.compute_heuristic(
        graph.rewards, graph.dists, num_samples=100, kappa=0.5
    )

    # TODO: Add indices to this function
    indices = torch.arange(batch_size)
    current_node = start_node

    # the first action
    # while budget > 0 and current_node != end_node:
    # {
    result = MCTS_SOPCC(
        graph, heuristic, goal_node, path, current_node, budget, num_simulations
    )
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
    heuristic: Heuristic,
    goal_node: torch.Tensor,
    current_path: Path,
    current_node: torch.Tensor,
    budget: torch.Tensor,
    num_simulations: int,
):
    # Create new MCTS tree
    tree = instantiate_tree_from_root(current_node, graph, num_simulations)
    # For K iterations
    # Select new child vertex with UCTF
    parent_node, leaf_node_index, depth, path_from_root = simulate(
        tree, graph, current_path, max_depth=num_simulations
    )
    # path = create_path_from_root(tree, graph, parent_node, leaf_node_index, depth)
    ts = sample_traverse_cost(path_from_root, samples=2, kappa=0.5)
    new_budgets = budget.unsqueeze(-1) - ts
    # try on first sample before batching
    new_budgets = new_budgets[:, 0].squeeze()
    paths = batch_rollout(
        graph, heuristic, goal_node, path_from_root, leaf_node_index, new_budgets
    )
    # print(paths)
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
    tree: Tree, graph: Graph, current_path: Path, max_depth: int, z: float = 0.1
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

    # Create path from root_node
    root_node_idx = tree.node_mapping[indices, parent_node]
    path = path_lib.create_path_from_start(graph, root_node_idx)

    # Add mask from current_path to new_path
    path_lib.combine_masks(path, current_path)

    while indices.numel() > 0:
        # Only worry about continuing nodes
        current_node = parent_node[indices]  # [I,]

        increase_visit_count(tree, indices, current_node, depth)

        # Compute the uctf score
        scores = compute_uctf(tree, indices, current_node, z)  # [I, A]
        # mask scores to remove visited nodes
        # scores = scores.masked_fill_(path.mask[indices], 0)  # [I, A]
        scores = torch.masked_fill(scores, path.mask[indices], 0)  # [I, A]
        # Choose next neighbor to traverse
        next_neighbor_index = torch.argmax(scores, axis=-1)  # [I,]
        next_neighbor = tree.children_index[indices, current_node, next_neighbor_index]

        # Add best neighbors for current_nodes
        leaf_node_index[indices] = next_neighbor_index
        path_lib.add_node(path, graph, next_neighbor_index, indices)

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

    return parent_node, leaf_node_index, depth, path


def sample_cost(weight: torch.Tensor, num_samples: int, kappa: float = 0.5):
    rate = 1 / ((1 - kappa) * weight)
    samples = heuristic_lib.sample_exponential_distribution(rate, num_samples)
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


def sample_edge_cost(
    graph: torch.Tensor,
    from_node: torch.Tensor,
    to_node: torch.Tensor,
    indices: torch.Tensor,
    kappa: float = 0.5,
):
    weights = graph.dists[indices, from_node, to_node]
    return sample_cost(weights, num_samples=1, kappa=kappa)


def select_random(distribution: torch.Tensor, indices: torch.Tensor):
    return torch.multinomial(distribution[indices], num_samples=1).squeeze()


def select_greedy(
    path: Path,
    scores: torch.Tensor,
    current_node: torch.Tensor,
    indices: torch.Tensor,
):
    mask = path.mask[indices]
    score = scores[indices, current_node[indices]]
    score = torch.masked_fill(score, mask, 0)
    return torch.argmax(score, dim=-1)


def compute_failure_prob(
    heuristic: Heuristic,
    current_node: torch.Tensor,
    new_node: torch.Tensor,
    goal_node: torch.Tensor,
    budget: torch.Tensor,
    indices: torch.Tensor,
):
    c = current_node[indices]
    n = new_node[indices]
    g = goal_node[indices]
    B = budget[indices]
    sample_c_n = heuristic.samples[indices, c, n]
    sample_n_g = heuristic.samples[indices, n, g]
    total_sample_cost = sample_c_n + sample_n_g
    return (
        torch.sum(total_sample_cost > B.unsqueeze(-1), dim=-1)
        / total_sample_cost.shape[-1]
    )


def batch_rollout(
    graph: Graph,
    heuristic: Heuristic,
    goal_node: torch.Tensor,
    path_from_root: Path,
    leaf_node: torch.Tensor,
    budget: torch.Tensor,
    P_R: float = 0.1,
    P_F: float = 0.1,
):
    batch_size, num_nodes = graph_lib.infer_graph_shape(graph)
    current_node = leaf_node
    simulated_budget = budget
    path = path_lib.create_path_from_start(graph, current_node)
    path_lib.combine_masks(path, path_from_root)

    # create distribution from mask for random selection
    distribution = (~path.mask).float()

    indices = torch.arange(batch_size)

    while indices.numel() > 0:
        p = torch.rand(size=indices.shape)
        should_explore = p < P_R
        # node buffer
        new_nodes = torch.zeros(indices.shape, dtype=torch.long)
        # random nodes
        explore_indices = indices[should_explore]
        new_nodes[explore_indices] = select_random(distribution, explore_indices)
        # greedy nodes
        greedy_indices = indices[~should_explore]
        new_nodes[greedy_indices] = select_greedy(
            path, heuristic.scores, current_node, greedy_indices
        )

        is_not_goal = new_nodes != goal_node

        # case 1: new_node is not goal
        continuing_indices = indices[is_not_goal]
        failure_prob = compute_failure_prob(
            heuristic, current_node, new_nodes, goal_node, budget, continuing_indices
        )
        has_not_failed = failure_prob <= P_F
        # case 1a: failure_prob <= P_F, add nodes to path
        continuing_succeeded_indices = continuing_indices[has_not_failed]
        from_nodes = current_node[continuing_succeeded_indices]
        continuing_nodes = new_nodes[continuing_succeeded_indices]
        path_lib.add_node(path, graph, continuing_nodes, continuing_succeeded_indices)
        sampled_cost = sample_edge_cost(
            graph, from_nodes, continuing_nodes, continuing_succeeded_indices, kappa=0.5
        )
        simulated_budget[continuing_succeeded_indices] -= sampled_cost.squeeze()
        current_node[continuing_succeeded_indices] = continuing_nodes

        # case 1b: failure_prob > P_F, discard nodes
        failed_indices = continuing_indices[~has_not_failed]
        failed_nodes = new_nodes[failed_indices]
        # remove from greedy selection
        path_lib.add_to_mask(path, failed_nodes, failed_indices)

        # for both parts in case 1, if all nodes are visited, return add goal and return
        # remove from nodes from selection
        distribution[continuing_indices, new_nodes[continuing_indices]] = 0
        # if distribution is all 0s, add goal and return path
        all_visited = torch.sum(distribution[continuing_indices], dim=-1) == 0
        finished_indices = continuing_indices[all_visited]
        path_lib.add_node(path, graph, goal_node[finished_indices], finished_indices)

        # case 2: new_node is goal
        goal_indices = indices[~is_not_goal]
        path_lib.add_node(path, graph, goal_node[goal_indices], goal_indices)

        # update state
        indices = continuing_indices[~all_visited]
        current_node = current_node[indices]

    return path
