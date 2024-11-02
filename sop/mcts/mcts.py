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
    # with torch.no_grad():
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
    for k in range(num_simulations):
        start = time.time()
        # For K iterations
        # Select new child vertex with UCTF
        parent_node, leaf_node_index, depth, path_from_root = simulate(
            tree, graph, current_path, max_depth=num_simulations
        )
        Q, F = rollouts(
            graph, heuristic, goal_node, path_from_root, leaf_node_index, budget, 100
        )
        leaf_node = expand(tree, parent_node, leaf_node_index, Q, F)
        backup(tree, graph, leaf_node)
        print(f"{k}: {time.time() - start}")
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
        node_mapping=torch.full(batch_node, Tree.UNVISITED, dtype=torch.long),
        raw_values=torch.zeros(batch_node),
        node_values=torch.zeros(batch_node),
        raw_failure_probs=torch.zeros(batch_node),
        failure_probs=torch.zeros(batch_node),
        node_visits=torch.zeros(batch_node),
        parents=torch.full(batch_node, Tree.NO_PARENT),
        neighbor_from_parent=torch.full(batch_node, Tree.UNVISITED),
        children_index=torch.full(
            batch_node_neighbors, Tree.UNVISITED, dtype=torch.long
        ),
        children_values=torch.zeros(batch_node_neighbors),
        children_failure_probs=torch.zeros(batch_node_neighbors),
        children_visits=torch.zeros(batch_node_neighbors),
        num_nodes=torch.ones((batch_size,), dtype=torch.long),  # start with root node
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


# TODO: clean this up
def expand(
    tree: Tree,
    parent_node: torch.Tensor,
    leaf_node_index: torch.Tensor,
    Q: torch.Tensor,
    F: torch.Tensor,
):
    batch_size, _ = tree_lib.infer_tree_shape(tree)
    num_nodes = tree.num_nodes

    indices = torch.arange(batch_size)

    # Find index of node
    child_node = tree.children_index[indices, parent_node, leaf_node_index]
    is_unvisited = child_node == Tree.UNVISITED

    # add node to tree if unvisited
    ui = indices[is_unvisited]  # unvisited indices
    new_idx = num_nodes[is_unvisited]  # new index in tree
    if ui.numel() > 0:
        p = parent_node[ui]
        l = leaf_node_index[ui]
        q = Q[ui]
        f = F[ui]

        tree.node_mapping[ui, new_idx] = l
        tree.raw_values[ui, new_idx] = q
        tree.raw_failure_probs[ui, new_idx] = f
        tree.node_visits[ui, new_idx] = 1

        tree.neighbor_from_parent[ui, new_idx] = l
        tree.parents[ui, new_idx] = p
        tree.children_index[ui, p, l] = new_idx
        tree.children_values[ui, p, l] = q
        tree.children_failure_probs[ui, p, l] = f

        tree.num_nodes[ui] += 1

    # update Q and F if node is visited
    is_visited = ~is_unvisited
    vi = indices[is_visited]
    idx = child_node[is_visited]
    if vi.numel() > 0:
        p = parent_node[vi]
        l = leaf_node_index[vi]
        q = Q[vi]
        f = F[vi]

        # TODO: Update average instead of replacing
        tree.raw_values[ui, idx] = q
        tree.raw_failure_probs[ui, new_idx] = f
        tree.node_visits[ui, new_idx] += 1  # TODO: Do we need this?
        tree.children_values[ui, p, l] = q
        tree.children_failure_probs[ui, p, l] = f

    leaf_node = torch.zeros((batch_size,), dtype=torch.long)
    leaf_node[ui] = new_idx
    leaf_node[vi] = idx

    return leaf_node


def backup(tree: Tree, graph: Graph, leaf_node: torch.Tensor, p_f: float = 0.1):
    batch_size, _ = tree_lib.infer_tree_shape(tree)
    indices = torch.arange(batch_size)

    vj = leaf_node
    vj_mapping = tree.neighbor_from_parent[indices, vj]
    vi = tree.parents[indices, vj]
    vk = tree.parents[indices, vj]

    # check if v_k is null
    has_parent = vk != Tree.NO_PARENT
    indices = indices[has_parent]

    while indices.numel() > 0:
        vi_mapping = tree.neighbor_from_parent[indices, vi]
        vi_q = tree.children_values[indices, vk, vi_mapping]
        vi_f = tree.children_failure_probs[indices, vk, vi_mapping]
        vi_r = graph.rewards[indices, vi_mapping]

        vj = tree.children_index[indices, vi, vj_mapping]
        vj_q = tree.children_values[indices, vi, vj_mapping]
        vj_f = tree.children_failure_probs[indices, vi, vj_mapping]

        new_q = vj_q + vi_r
        new_f = vj_f

        # condition 1
        # if vk.F[vi] < Pf
        a = vi_f < p_f
        a_i = indices[a]
        #   if vi.F[vj] < Pf
        b = vj_f[a_i] < p_f
        b_i = a_i[b]
        #     if vk.Q[vi] < vi.Q[vj] + r(vi)
        c = vi_q[b_i] < new_q[b_i]
        c_i = b_i[c]

        # condition 2
        # else if vk.F[vi] > vi.F[vj]
        not_a = ~a
        d = vi_f[not_a] > new_f[not_a]
        d_i = indices[not_a][d]

        valid_i = torch.concat([c_i, d_i])

        # condition #1 and #2
        if valid_i.numel() > 0:
            valid_vk = vk[valid_i]
            valid_vi = vi[valid_i]
            valid_vi_mapping = vi_mapping[valid_i]
            new_q = new_q[valid_i]
            new_f = new_f[valid_i]
            tree.node_values[valid_i, valid_vi] = new_q
            tree.children_values[valid_i, valid_vk, valid_vi_mapping] = new_q
            tree.failure_probs[valid_i, valid_vi] = new_f
            tree.children_failure_probs[valid_i, valid_vk, valid_vi_mapping] = new_f

        # update values
        vi = vk
        vk = tree.parents[indices, vk]

        has_parent = vk != Tree.NO_PARENT
        indices = indices[has_parent]


def rollouts(
    graph: Graph,
    heuristic: Heuristic,
    goal_node: torch.Tensor,
    path_from_root: Path,
    leaf_node: torch.Tensor,
    budget: torch.Tensor,
    S: int = 2,
    p_r: float = 0.1,
    p_f: float = 0.1,
    kappa: float = 0.5,
):
    # infer shape
    batch_size, _ = graph_lib.infer_graph_shape(graph)

    # Precompute traversal costs
    ts = sample_traverse_cost(path_from_root, samples=S, kappa=kappa)
    new_budget = budget.unsqueeze(-1) - ts

    # Buffer for Q and F
    Q = torch.zeros((batch_size,))
    F = torch.zeros((batch_size,))

    for s in range(S):
        B_prime = new_budget[:, s]
        q, f = rollout(
            graph,
            heuristic,
            goal_node,
            path_from_root,
            leaf_node,
            B_prime,
            p_r,
            p_f,
            kappa,
        )
        Q += q
        F += f

    # return average value and failure prob
    return Q / S, F / S


# @torch.compile(dynamic=True)
def select_random(distribution):
    return torch.multinomial(distribution, num_samples=1).squeeze()


# @torch.compile(dynamic=True)
def select_greedy(scores: torch.Tensor, mask: torch.Tensor):
    return torch.argmax(torch.masked_fill(scores, mask, 0), dim=-1)


# @torch.compile(dynamic=True)
def compute_failure_prob(sample_cost, B):
    return torch.sum(sample_cost > B.unsqueeze(-1), dim=-1) / sample_cost.shape[-1]


def rollout(
    graph: Graph,
    heuristic: Heuristic,
    goal_node: torch.Tensor,
    path_from_root: Path,
    leaf_node: torch.Tensor,
    B_prime: torch.Tensor,
    p_r: float = 0.1,
    p_f: float = 0.1,
    kappa: float = 0.5,
):
    batch_size, _ = graph_lib.infer_graph_shape(graph)

    # Buffer for q and success
    Q = torch.zeros((batch_size,))
    failures = torch.ones((batch_size,))

    # mask for greedy, distribution for random
    mask = path_from_root.mask.clone()
    distribution = (~mask).float()

    # loop state
    current_node = leaf_node.clone()
    simulated_budget = B_prime.clone()
    indices = torch.arange(batch_size)

    # Add reward of leaf_node to Q
    Q += graph.rewards[indices, current_node]

    while indices.numel() > 0:
        # 0: create new_nodes buffer
        new_nodes = torch.full(size=(batch_size,), fill_value=-1, dtype=torch.long)

        # 1: choose either random or greedy based on p_r
        p = torch.rand(size=indices.shape)
        choose_random = p < p_r
        random_indices = indices[choose_random]
        greedy_indices = indices[~choose_random]

        # 2a: if p < p_r, select random
        if random_indices.numel() > 0:
            d = distribution[random_indices]
            random_nodes = select_random(d)
            new_nodes[random_indices] = random_nodes

        # 2b: else p >= p_r select greedy
        if greedy_indices.numel() > 0:
            m = mask[greedy_indices]
            c = current_node[greedy_indices]
            s = heuristic.scores[greedy_indices, c]
            greedy_nodes = select_greedy(s, m)
            new_nodes[greedy_indices] = greedy_nodes

        # 4: if new != goal, calculate failure_prob of continuing nodes
        is_cont = new_nodes[indices] != goal_node[indices]
        cont_indices = indices[is_cont]

        if cont_indices.numel() > 0:
            # 4a: compute failure probability
            c = current_node[indices]
            n = new_nodes[indices]
            g = goal_node[indices]
            b = simulated_budget[indices]
            sample_c_n = heuristic.samples[cont_indices, c, n]
            sample_n_g = heuristic.samples[cont_indices, n, g]
            total_sample_cost = sample_c_n + sample_n_g
            failure_prob = compute_failure_prob(total_sample_cost, b)

            # 4b: determine whether continuing node failed or succeeded
            below_failure = failure_prob <= p_f
            suc_indices = cont_indices[below_failure]

            # 4c: if Pr[...] <= p_f, add to path
            if suc_indices.numel() > 0:
                c = current_node[suc_indices]
                n = new_nodes[suc_indices]
                # sample cost
                w = graph.dists[suc_indices, c, n]
                simulated_budget[suc_indices] -= sample_cost(
                    w, num_samples=1, kappa=kappa
                ).squeeze()
                # add reward
                Q[suc_indices] += graph.rewards[suc_indices, n]
                # change current_node
                current_node[suc_indices] = n

            # 4d: always update mask
            c = new_nodes[cont_indices]
            mask[cont_indices, c] = 1
            distribution[cont_indices, c] = 0

            # 4e: if all nodes have been visited, go to goal and return
            all_visited = torch.sum(distribution[cont_indices], dim=-1) == 0
            complete_indices = cont_indices[all_visited]
            new_nodes[complete_indices] = goal_node[complete_indices]

        # 7: else new == goal, add to path and return
        is_goal = new_nodes[indices] == goal_node[indices]
        goal_indices = indices[is_goal]

        if goal_indices.numel() > 0:
            c = current_node[goal_indices]
            n = new_nodes[goal_indices]
            # add reward
            Q[goal_indices] += graph.rewards[goal_indices, n]
            # Determine failure or success
            w = graph.dists[goal_indices, c, n]
            b_final = (
                B_prime[goal_indices]
                - sample_cost(w, num_samples=1, kappa=kappa).squeeze()
            )
            is_success = b_final >= 0
            success_indices = goal_indices[is_success]
            failures[success_indices] = 0

        # update indices
        indices = indices[~is_goal]

    return Q, failures
