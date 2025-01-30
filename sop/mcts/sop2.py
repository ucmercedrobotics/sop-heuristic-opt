from typing import Tuple
import time

import torch
from torch import Tensor
from tensordict import tensorclass

from sop.utils.path import Path
from sop.utils.graph_torch import TorchGraph
from sop.utils.sample import sample_costs


# generates random probability of selecting particular edge, as in random rollout
def random_heuristic(batch_size: int, num_nodes: int):
    return torch.rand((batch_size, num_nodes, num_nodes)).softmax(-1)


# generates average utility (reward/average cost) for edges, as in MCTSSOPCC
def mcts_sopcc_heuristic(rewards: Tensor, sampled_costs: Tensor):
    average_cost = sampled_costs.mean(dim=-1)
    return rewards.unsqueeze(-1) / average_cost


def sop_mcts_solver(
    graph: TorchGraph,
    heuristic: Tensor,
    num_simulations: int,
    num_rollouts: int,
    z: float,
) -> Tuple[Path, Tensor]:
    batch_size, num_nodes = graph.size()

    current_node = graph.extra["start_node"].clone()
    current_budget = graph.extra["budget"].clone()
    goal_node = graph.extra["goal_node"].clone()

    # Loop State
    indices = torch.arange(batch_size)
    path = Path.empty(batch_size, num_nodes)
    path.append(indices, current_node)

    while indices.numel() > 0:
        next_node = MCTS_SOPCC(
            graph[indices],
            heuristic[indices],
            current_node[indices],
            current_budget[indices],
            path[indices],
            num_simulations,
            num_rollouts,
            z,
        )

        # Get values
        r = graph.nodes["reward"][indices, next_node]
        w = graph.edges["distance"][indices, current_node[indices], next_node]

        # Update budget
        sampled_cost = sample_costs(w, num_samples=1, kappa=0.5).squeeze()
        current_budget[indices] -= sampled_cost

        # Add to path
        path.append(indices, next_node, reward=r)

        # Update state
        is_not_goal = next_node != goal_node[indices]
        has_budget = current_budget[indices] > 0
        is_continuing = torch.logical_and(is_not_goal, has_budget)
        indices = indices[is_continuing]

    # Check if run was success
    is_success = current_budget > 0

    return path, is_success


def MCTS_SOPCC(
    graph: TorchGraph,
    heuristic: Tensor,
    current_node: Tensor,
    current_budget: Tensor,
    current_path: Path,
    num_simulations: int,
    num_rollouts: int,
    z: float,
) -> Tuple[Tensor, Tensor]:
    tree = Tree.instantiate_from_root(current_node, graph, num_simulations)
    for k in range(num_simulations):
        # s = time.time()
        parent_tree_node, new_graph_node, tree_path, is_expanded, is_finished = select(
            tree, graph, heuristic, current_path, z, score_fn=compute_uctf
        )
        # print(f"Select: {time.time() - s}")
        # s = time.time()
        Q, F = e_greedy_rollout(
            graph, heuristic, tree_path, new_graph_node, current_budget, num_rollouts
        )
        # print(f"Rollout: {time.time() - s}")
        # s = time.time()
        new_tree_node = expand(
            tree, parent_tree_node, new_graph_node, Q, F, is_expanded
        )
        # print(f"Expand: {time.time() - s}")
        # s = time.time()
        backup(tree, graph, new_tree_node)
        # print(f"Backup: {time.time() - s}")
        # s = time.time()
        backupN(tree, new_tree_node)
        # print(f"BackupN: {time.time() - s}")

    return select_action(tree, graph, current_path)


def sop_mcts_aco_solver(
    graph: TorchGraph,
    heuristic: Tensor,
    num_simulations: int,
    num_rollouts: int,
    z: float,
    init_lr: float,
) -> Tuple[Path, Tensor]:
    batch_size, num_nodes = graph.size()

    current_node = graph.extra["start_node"].clone()
    current_budget = graph.extra["budget"].clone()
    goal_node = graph.extra["goal_node"].clone()

    # Clone heuristic
    heuristic = heuristic.clone()

    # Loop State
    indices = torch.arange(batch_size)
    path = Path.empty(batch_size, num_nodes)
    path.append(indices, current_node)

    while indices.numel() > 0:
        next_node = MCTS_ACO(
            graph[indices],
            heuristic[indices],
            current_node[indices],
            current_budget[indices],
            path[indices],
            num_simulations,
            num_rollouts,
            z,
            init_lr,
        )

        # Get values
        r = graph.nodes["reward"][indices, next_node]
        w = graph.edges["distance"][indices, current_node[indices], next_node]

        # Update budget
        sampled_cost = sample_costs(w, num_samples=1, kappa=0.5).squeeze()
        current_budget[indices] -= sampled_cost

        # Add to path
        path.append(indices, next_node, reward=r)

        # Update state
        is_not_goal = next_node != goal_node[indices]
        has_budget = current_budget[indices] > 0
        is_continuing = torch.logical_and(is_not_goal, has_budget)
        indices = indices[is_continuing]

    # Check if run was success
    is_success = current_budget >= 0

    return path, is_success, heuristic


def MCTS_ACO(
    graph: TorchGraph,
    heuristic: Tensor,
    current_node: Tensor,
    current_budget: Tensor,
    current_path: Path,
    num_simulations: int,
    num_rollouts: int,
    z: float,
    init_lr: float,
) -> Tuple[Tensor, Tensor]:
    tree = Tree.instantiate_from_root(current_node, graph, num_simulations)

    # TODO Rollout and BackupPaths on Root node
    Q, F, rollout_paths = adaIR_rollout(
        graph, heuristic, current_path, current_node, current_budget, num_rollouts
    )
    # backupPaths(tree, graph, heuristic, rollout_paths)

    for k in range(num_simulations):
        # -- Select a node to rollout
        parent_tree_node, new_graph_node, tree_path, is_expanded, is_finished = select(
            tree, graph, heuristic, current_path, z, score_fn=compute_uctf
        )

        # -- Rollout
        Q, F, rollout_paths = adaIR_rollout(
            graph,
            heuristic,
            tree_path,
            new_graph_node,
            current_budget,
            num_rollouts,
            lr=init_lr,
        )

        # -- Expand Node
        new_tree_node = expand(
            tree, parent_tree_node, new_graph_node, Q, F, is_expanded
        )

        # -- Backup Paths
        # backupPaths(tree, graph, heuristic, rollout_paths)

        # -- Backup
        backup(tree, graph, new_tree_node)

        # -- BackupN
        backupN(tree, new_tree_node)

    return select_action(tree, graph, current_path)


# -- CONSTANTS
ROOT_INDEX = 0
UNVISITED = -1
NO_PARENT = -1


# -- TREE
@tensorclass
class Tree:
    node_index: Tensor  # [B, N], index of node in tree
    node_mapping: Tensor  # [B, N], index of node in original graph
    visit_count: Tensor  # [B, N], visit counts for node
    parent_index: Tensor  # [B, N], node index for the parents of node
    action_from_parent: Tensor  # [B, N], the action taken from parent to node
    children_index: Tensor  # [B, N, A], node index for all children of node
    children_visit_counts: Tensor  # [B, N, A], visit count of children

    node_value: Tensor  # [B, N], V(S) for each node `S`
    children_Q_values: Tensor  # [B, N, A], Q(S, A) for each parent `S` and child `A`

    # SOP specific
    node_failure_prob: Tensor  # [B, N], F(S) for each node `S`
    children_failure_probs: Tensor  # [B, N], best F(A) for each node `S` and child `A`

    num_nodes: Tensor

    @classmethod
    def instantiate_from_root(
        cls,
        root_graph_node: Tensor,
        graph: TorchGraph,
        num_simulations: int,
    ) -> "Tree":
        batch_size, num_nodes = graph.size()
        num_simulations = num_simulations * num_nodes + 1

        max_nodes = (batch_size, num_simulations)
        max_children = (batch_size, num_simulations, num_nodes)

        tree = cls(
            node_index=torch.arange(num_simulations).unsqueeze(0).expand(max_nodes),
            node_mapping=torch.full(max_nodes, UNVISITED, dtype=torch.long),
            visit_count=torch.zeros(max_nodes),
            parent_index=torch.full(max_nodes, NO_PARENT, dtype=torch.long),
            action_from_parent=torch.full(max_nodes, UNVISITED, dtype=torch.long),
            children_index=torch.full(max_children, UNVISITED, dtype=torch.long),
            children_visit_counts=torch.zeros(max_children),
            node_value=torch.full(max_nodes, -torch.inf, dtype=torch.float32),
            children_Q_values=torch.full(max_children, -torch.inf, dtype=torch.float32),
            node_failure_prob=torch.zeros(max_nodes, dtype=torch.float32),
            children_failure_probs=torch.zeros(max_children, dtype=torch.float32),
            num_nodes=torch.ones((batch_size,), dtype=torch.long),
            batch_size=[batch_size],
        )

        tree.node_mapping[:, ROOT_INDEX] = root_graph_node
        tree.visit_count[:, ROOT_INDEX] = 1

        return tree

    def size(self):
        # [batch_size, num_simulations, num_graph_nodes]
        return self.children_index.shape


# -- SELECT
def compute_uctf(
    tree: Tree, heuristic: Tensor, indices: Tensor, tree_node: Tensor, z: float
) -> Tensor:
    """Compute UCTF for all children of a node.

    Formula: UCTF(v_j) = Q[v_j]*(1-F[v_j]) + z*sqrt(log(t)/N[v_j])
             ---------   ----------------   -------------------
            for all v_j  Expected Utility       Exploration
    """
    Q = tree.children_Q_values[indices, tree_node]
    F = tree.children_failure_probs[indices, tree_node]
    N = tree.children_visit_counts[indices, tree_node]
    t = tree.visit_count[indices, tree_node]

    # In the case of log(1)/0 (first simulation), the exploration term becomes sqrt(-inf).
    # This results in a nan. We add a small term to avoid this and get the desired sqrt(inf).
    return Q * (1 - F) + z * torch.sqrt(torch.log(t + 1e-5).unsqueeze(-1) / N)


def compute_puctf(
    tree: Tree, heuristic: Tensor, indices: Tensor, tree_node: Tensor, z: float
) -> Tensor:
    """Compute pUCTF for all children of a node.

    Formula: pUCTF(v_j) = Q[v_j]*(1-F[v_j]) + p[v_j] * z*sqrt(log(t)/N[v_j])
                          ----------------    -----         -----------
                          Expected Utility    prior         Exploration
    """
    Q = tree.children_Q_values[indices, tree_node]
    F = tree.children_failure_probs[indices, tree_node]
    N = tree.children_visit_counts[indices, tree_node]
    t = tree.visit_count[indices, tree_node]

    mapping = tree.node_mapping[indices, tree_node]
    p = heuristic[indices, mapping]

    # In the case of log(1)/0 (first simulation), the exploration term becomes sqrt(-inf).
    # This results in a nan. We add a small term to avoid this and get the desired sqrt(inf).
    return Q * (1 - F) + p * z * torch.sqrt(torch.log(t + 1e-5).unsqueeze(-1) / N)


def select(
    tree: Tree,
    graph: TorchGraph,
    heuristic: Tensor,
    current_path: Path,
    z: float,
    score_fn=compute_uctf,
) -> Tuple[Tensor, Tensor, Path, Tensor, Tensor]:
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    # Loop State
    parent_tree_node = torch.full((batch_size,), ROOT_INDEX, dtype=torch.long)
    leaf_graph_node = torch.full((batch_size,), UNVISITED, dtype=torch.long)
    tree_path = current_path.clone()
    is_expanded = torch.zeros((batch_size,), dtype=torch.bool)
    is_finished = torch.zeros((batch_size,), dtype=torch.bool)
    goal_node = graph.extra["goal_node"]

    while indices.numel() > 0:
        current_tree_node = parent_tree_node[indices]

        # Compute score for each child
        scores = score_fn(tree, heuristic, indices, current_tree_node, z)
        masked_scores = torch.masked_fill(scores, tree_path.mask[indices], -torch.inf)

        # Select best node
        next_graph_node = torch.argmax(masked_scores, axis=-1)
        next_tree_node = tree.children_index[
            indices, current_tree_node, next_graph_node
        ]

        # Update loop state
        leaf_graph_node[indices] = next_graph_node

        # Add to tree path
        tree_path.append(indices, next_graph_node)

        # 1. Check if node is leaf
        is_expanded[indices] = next_tree_node != UNVISITED
        # 2. Check if node is goal node
        is_finished[indices] = next_graph_node == goal_node[indices]

        # Continue traversal if expanded and not finished
        is_continuing = torch.logical_and(is_expanded[indices], ~is_finished[indices])
        indices = indices[is_continuing]
        if indices.numel() == 0:
            break

        # Update loop state
        parent_tree_node[indices] = next_tree_node[is_continuing]

    return parent_tree_node, leaf_graph_node, tree_path, is_expanded, is_finished


# -- EXPAND
def add_node(
    tree: Tree,
    indices: Tensor,
    parent_tree_node: Tensor,
    new_graph_node: Tensor,
    Q: Tensor,
    F: Tensor,
):
    new_tree_node = tree.num_nodes[indices]

    tree.node_mapping[indices, new_tree_node] = new_graph_node
    tree.action_from_parent[indices, new_tree_node] = new_graph_node
    tree.parent_index[indices, new_tree_node] = parent_tree_node
    tree.children_index[indices, parent_tree_node, new_graph_node] = new_tree_node

    tree.visit_count[indices, new_tree_node] = 1
    tree.children_visit_counts[indices, parent_tree_node, new_graph_node] = 1

    # TODO: Add value and failure prob for individual node

    tree.children_Q_values[indices, parent_tree_node, new_graph_node] = Q
    tree.children_failure_probs[indices, parent_tree_node, new_graph_node] = F

    tree.num_nodes[indices] += 1

    return new_tree_node


def update_node(
    tree: Tree, indices: Tensor, parent_tree_node: Tensor, current_graph_node: Tensor
):
    current_tree_node = tree.children_index[
        indices, parent_tree_node, current_graph_node
    ]

    tree.visit_count[indices, current_tree_node] += 1
    tree.children_visit_counts[indices, parent_tree_node, current_graph_node] += 1

    return current_tree_node


def expand(
    tree: Tree,
    parent_tree_node: Tensor,
    new_graph_node: Tensor,
    Q: Tensor,
    F: Tensor,
    is_expanded: Tensor,
):
    batch_size, _, _ = tree.size()
    indices = torch.arange(batch_size)

    new_tree_nodes = torch.empty((batch_size,), dtype=torch.long)

    add_i = indices[~is_expanded]
    if add_i.numel() > 0:
        pt = parent_tree_node[add_i]
        ng = new_graph_node[add_i]
        q = Q[add_i]
        f = F[add_i]
        new_tree_nodes[add_i] = add_node(tree, add_i, pt, ng, q, f)

    update_i = indices[is_expanded]
    if update_i.numel() > 0:
        pt = parent_tree_node[update_i]
        ng = new_graph_node[update_i]
        new_tree_nodes[update_i] = update_node(tree, update_i, pt, ng)

    return new_tree_nodes


# -- BACKUP
def backup(tree: Tree, graph: TorchGraph, leaf_tree_node: Tensor, p_f: float = 0.1):
    batch_size, _, _ = tree.size()
    indices = torch.arange(batch_size)

    vj = leaf_tree_node.clone()  # [B,]
    vi = tree.parent_index[indices, vj]  # [B,]
    vk = tree.parent_index[indices, vi]  # [B,]

    # check if v_k is null
    has_parent = vk != NO_PARENT
    # mask values
    indices = indices[has_parent]  # [I,]
    vj = vj[has_parent]  # [I,]
    vi = vi[has_parent]  # [I,]
    vk = vk[has_parent]  # [I,]

    while indices.numel() > 0:
        vi_mapping = tree.node_mapping[indices, vi]
        vi_q = tree.children_Q_values[indices, vk, vi_mapping]
        vi_f = tree.children_failure_probs[indices, vk, vi_mapping]
        vi_r = graph.nodes["reward"][indices, vi_mapping]

        vj_mapping = tree.node_mapping[indices, vj]
        vj = tree.children_index[indices, vi, vj_mapping]
        vj_q = tree.children_Q_values[indices, vi, vj_mapping]
        vj_f = tree.children_failure_probs[indices, vi, vj_mapping]

        new_q = vj_q + vi_r
        new_f = vj_f

        # condition 1
        # if vk.F[vi] < Pf
        a = vi_f < p_f
        #   if vi.F[vj] < Pf
        b = vj_f < p_f
        #     if vk.Q[vi] < vi.Q[vj] + r(vi)
        c = vi_q < new_q

        cond1 = a.logical_and(b).logical_and(c)

        # condition 2
        # else if vk.F[vi] > vi.F[vj]
        cond2 = vi_f > new_f

        # if condition 1 or condition 2
        valid = torch.logical_or(cond1, cond2)
        valid_i = indices[valid]

        if valid_i.numel() > 0:
            valid_vk = vk[valid]
            valid_vi = vi[valid]
            valid_vi_mapping = vi_mapping[valid]
            new_q = new_q[valid]
            new_f = new_f[valid]

            tree.node_value[valid_i, valid_vi] = new_q
            tree.children_Q_values[valid_i, valid_vk, valid_vi_mapping] = new_q
            tree.node_failure_prob[valid_i, valid_vi] = new_f
            tree.children_failure_probs[valid_i, valid_vk, valid_vi_mapping] = new_f

        # update values
        vj = tree.parent_index[indices, vj]
        vi = tree.parent_index[indices, vi]
        vk = tree.parent_index[indices, vk]

        has_parent = vk != NO_PARENT
        indices = indices[has_parent]
        vj = vj[has_parent]
        vi = vi[has_parent]
        vk = vk[has_parent]


def backupN(tree: Tree, leaf_tree_node: Tensor):
    batch_size, _, _ = tree.size()
    indices = torch.arange(batch_size)

    # Loop state
    current_tree_node = tree.parent_index[indices, leaf_tree_node]
    parent_tree_node = tree.parent_index[indices, current_tree_node]

    # Backup visit count to root
    while indices.numel() > 0:
        # Update node visit count
        tree.visit_count[indices, current_tree_node] += 1

        # Check if node has parent
        has_parent = parent_tree_node != NO_PARENT
        indices = indices[has_parent]
        if indices.numel() == 0:
            break
        current_tree_node = current_tree_node[has_parent]
        parent_tree_node = parent_tree_node[has_parent]

        # Update parent
        action = tree.action_from_parent[indices, current_tree_node]
        tree.children_visit_counts[indices, parent_tree_node, action] += 1

        # Update Loop state
        current_tree_node = parent_tree_node
        parent_tree_node = tree.parent_index[indices, parent_tree_node]


# -- Action Selection
def select_action(
    tree: Tree, graph: TorchGraph, path: Path, p_f: float = 0.1
) -> Tuple[Tensor, Tensor]:
    batch_size, _, num_nodes = tree.size()
    indices = torch.arange(batch_size)

    scores = tree.children_Q_values[indices, ROOT_INDEX]
    failure_probs = tree.children_failure_probs[indices, ROOT_INDEX]

    failure_mask = failure_probs > p_f
    mask = torch.logical_or(path.mask, failure_mask)
    masked_scores = torch.masked_fill(scores, mask, -torch.inf)

    action = torch.argmax(masked_scores, axis=-1)
    score = masked_scores[indices, action]

    # Go to goal if no valid nodes
    is_invalid = score == -torch.inf
    invalid_i = indices[is_invalid]
    if invalid_i.numel() > 0:
        action[is_invalid] = graph.extra["goal_node"][invalid_i]

    return action


# -- ROLLOUT
def sample_traverse_cost(
    path: Path, graph: TorchGraph, num_samples: int, kappa: float = 0.5
):
    batch_size, max_length = path.size()
    indices = torch.arange(batch_size)

    total_sampled_cost = torch.zeros((batch_size, num_samples))

    path_index = 1
    while path_index < max_length:
        prev_node = path.nodes[indices, path_index - 1]
        current_node = path.nodes[indices, path_index]
        weight = graph.edges["distance"][indices, prev_node, current_node]
        # samples = graph.edges["samples"][indices, prev_node, current_node]

        is_continuing = current_node != -1
        indices = indices[is_continuing]
        if indices.numel() == 0:
            break

        sampled_cost = sample_costs(weight[is_continuing], num_samples, kappa)
        total_sampled_cost[indices] += sampled_cost
        # total_sampled_cost[indices] += samples[is_continuing]

        path_index += 1

    return total_sampled_cost


def adaIR_action_selection(
    p: Tensor, path_mask: Tensor, failure_mask: Tensor, lr: float = 0.1
):
    r = torch.rand_like(p) ** lr
    mask = torch.logical_or(path_mask, failure_mask)
    scores = torch.masked_fill(p * r, mask, 0)
    return torch.argmax(scores, dim=-1), mask


def compute_failure_prob(sample_c_n: Tensor, sample_n_g: Tensor, B: Tensor) -> Tensor:
    total_sample_cost = sample_c_n + sample_n_g
    return (
        torch.sum(total_sample_cost > B.unsqueeze(-1), dim=-1)
        / total_sample_cost.shape[-1]
    )


def adaIR_rollout(
    graph: TorchGraph,
    heuristic: Tensor,
    tree_path: Tensor,
    leaf_graph_node: Tensor,
    budget: Tensor,
    num_rollouts: int,
    p_r: float = 0.1,
    p_f: float = 0.1,
    kappa: float = 0.5,
    lr: float = 1.5,
) -> Tuple[Tensor, Tensor, Path]:
    """Takes inspiration from: https://github.com/EMI-Group/tensoraco/"""
    # Define shapes
    batch_size, num_nodes = graph.size()
    sim_shape = (batch_size, num_rollouts)
    flatten_shape = (batch_size * num_rollouts,)

    # Create indices
    # batch_indices will query batch-level resources, like graph rewards and edge weights
    batch_indices = (
        torch.arange(batch_size).unsqueeze(-1).expand(sim_shape).flatten()
    )  # [B*S]
    # sim_indices will query simulation level resources, like simulated_budget and masks
    sim_indices = torch.arange(batch_size * num_rollouts)  # [B*S]

    # Create Paths
    sim_paths = tree_path.clone().unsqueeze(-1).expand(sim_shape).flatten()

    # Local Failure mask
    failure_mask = torch.zeros((batch_size * num_rollouts, num_nodes))

    # Sample traverse cost
    ts = sample_traverse_cost(tree_path, graph, num_rollouts, kappa)
    sampled_budgets = budget.unsqueeze(-1) - ts  # [B, S]
    simulated_budgets = sampled_budgets.flatten()

    # Preload data
    samples = graph.edges["samples"]
    weights = graph.edges["distance"]
    goal_node = graph.extra["goal_node"]
    rewards = graph.nodes["reward"]

    # Loop State
    current_nodes = (
        leaf_graph_node.clone().unsqueeze(-1).broadcast_to(sim_shape).flatten()
    )  # [B*S]

    while sim_indices.numel() > 0:
        # 0. Define action buffer
        new_nodes = torch.empty(flatten_shape, dtype=torch.long)

        # 1. Action selection
        c = current_nodes[sim_indices]
        p = heuristic[batch_indices, c]
        new_nodes[sim_indices], mask = adaIR_action_selection(
            p, sim_paths.mask[sim_indices], failure_mask[sim_indices], lr
        )

        # 1b. If mask has no valid nodes, go to goal
        is_invalid = mask.sum(-1) == num_nodes
        b_invalid_i, s_invalid_i = batch_indices[is_invalid], sim_indices[is_invalid]
        if is_invalid.numel() > 0:
            new_nodes[s_invalid_i] = goal_node[b_invalid_i]

        # 2. Determine if action is goal
        is_cont = new_nodes[sim_indices] != goal_node[batch_indices]
        b_cont_indices, s_cont_indices = batch_indices[is_cont], sim_indices[is_cont]

        # 3. If not goal, compute failure probability
        if s_cont_indices.numel() > 0:
            # 3a. Compute failure probability
            c = current_nodes[s_cont_indices]
            n = new_nodes[s_cont_indices]
            g = goal_node[b_cont_indices]
            b = simulated_budgets[s_cont_indices]
            sample_c_n = samples[b_cont_indices, c, n]
            sample_n_g = samples[b_cont_indices, n, g]
            failure_prob = compute_failure_prob(sample_c_n, sample_n_g, b)

            # Determine whether node failed or succeeded
            below_failure = failure_prob <= p_f
            b_suc_indices, s_suc_indices = (
                b_cont_indices[below_failure],
                s_cont_indices[below_failure],
            )
            # 3b. if Pr[...] <= p_f, add to path
            if s_suc_indices.numel() > 0:
                c = current_nodes[s_suc_indices]
                n = new_nodes[s_suc_indices]
                w = weights[b_suc_indices, c, n]

                # Sample and update budget
                sampled_cost = sample_costs(w, num_samples=1, kappa=kappa)
                simulated_budgets[s_suc_indices] -= sampled_cost.squeeze(-1)

                # Add to path
                r = rewards[b_suc_indices, n]
                sim_paths.append(s_suc_indices, n, r)

                # Change current node
                current_nodes[s_suc_indices] = n

                # Clear failure mask
                failure_mask[s_suc_indices] = 0

            # 3c. Update local failure mask so we don't choose this again
            s_fail_indices = s_cont_indices[~below_failure]
            if s_fail_indices.numel() > 0:
                n = new_nodes[s_fail_indices]
                failure_mask[s_fail_indices, n] = 1

        # 4: else is goal, add to path and return
        is_goal = ~is_cont
        b_goal_indices, s_goal_indices = batch_indices[is_goal], sim_indices[is_goal]
        if s_goal_indices.numel() > 0:
            c = current_nodes[s_goal_indices]
            n = new_nodes[s_goal_indices]
            w = weights[b_goal_indices, c, n]

            # Sample and update budget
            sampled_cost = sample_costs(w, num_samples=1, kappa=kappa)
            simulated_budgets[s_goal_indices] -= sampled_cost.squeeze(-1)

            # Add to path
            r = rewards[b_goal_indices, n]
            sim_paths.append(s_goal_indices, n, r)

        # 5. Update Loop State
        batch_indices, sim_indices = batch_indices[is_cont], sim_indices[is_cont]

    sim_paths = sim_paths.reshape(sim_shape)
    Q = sim_paths.reward.sum(-1).sum(-1) / num_rollouts
    F = (simulated_budgets.reshape(sim_shape) < 0).sum(-1) / num_rollouts

    return Q, F, sim_paths
