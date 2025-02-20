from typing import Tuple, Callable

import torch
from torch import Tensor
from tensordict import tensorclass

from sop.utils.path import Path
from sop.utils.graph import TorchGraph
from sop.utils.sample import sample_costs
from sop.inference.rollout import RolloutOutput, rollout, categorical_action_selection


# generates average utility (reward/average cost) for edges, as in MCTSSOPCC
def mcts_sopcc_heuristic(rewards: Tensor, sampled_costs: Tensor):
    average_cost = sampled_costs.mean(dim=-1)
    return rewards.unsqueeze(-1) / average_cost


def sop_mcts_solver(
    graph: TorchGraph,
    heuristic: Tensor,
    num_simulations: int,
    num_rollouts: int,
    p_f: float,
    z: float,
    kappa: float,
    action_selection_fn: Callable = categorical_action_selection,
) -> Tuple[Path, Tensor]:
    start_node = graph.extra["start_node"]
    budget = graph.extra["budget"]
    goal_node = graph.extra["goal_node"]
    rewards = graph.nodes["reward"]
    weights = graph.edges["distance"]

    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    # Loop State
    current_node = start_node.clone()
    current_budget = budget.clone()

    # Initialize Path
    path = Path.empty(batch_size, num_nodes)
    path.append(indices, current_node)
    # Mask goal node
    path.mask[indices, goal_node] = 1

    while indices.numel() > 0:
        next_node = MCTS_SOPCC(
            heuristic=heuristic[indices],
            graph=graph[indices],
            current_node=current_node[indices],
            current_budget=current_budget[indices],
            current_path=path[indices],
            num_simulations=num_simulations,
            num_rollouts=num_rollouts,
            z=z,
            p_f=p_f,
            kappa=kappa,
            action_selection_fn=action_selection_fn,
        )

        # Get next values
        r = rewards[indices, next_node]
        w = weights[indices, current_node[indices], next_node]

        # Update budget
        sampled_cost = sample_costs(w, num_samples=1, kappa=0.5)
        current_budget[indices] -= sampled_cost.squeeze(-1)

        # Add to path
        path.append(indices, next_node, r)

        # Check if done
        is_not_goal = next_node != goal_node[indices]
        has_budget = current_budget[indices] > 0
        is_continuing = torch.logical_and(is_not_goal, has_budget)

        # Update loop state
        indices = indices[is_continuing]
        current_node[indices] = next_node[is_continuing]

    # Check if run was success
    is_success = current_budget > 0

    return path, is_success


def MCTS_SOPCC(
    heuristic: Tensor,
    graph: TorchGraph,
    current_node: Tensor,
    current_budget: Tensor,
    current_path: Path,
    num_simulations: int,
    num_rollouts: int,
    z: float,
    p_f: float,
    action_selection_fn: Callable,
) -> Tuple[Tensor, Tensor]:
    tree = Tree.instantiate_from_root(current_node, graph, num_simulations)
    for k in range(num_simulations):
        parent_tree_node, new_graph_node, tree_path, is_expanded, is_finished = select(
            tree, graph, heuristic, current_path, z, score_fn=compute_uctf
        )
        # TODO: Hook up rollout with MCTS
        output = rollout(
            heuristic,
            graph,
            new_graph_node,
            current_budget,
            tree_path,
            num_rollouts,
            p_f,
            action_selection_fn,
        )
        Q, F = ...
        new_tree_node = expand(
            tree, parent_tree_node, new_graph_node, Q, F, is_expanded
        )
        backup(tree, graph, new_tree_node, p_f)
        backupN(tree, new_tree_node)

    return select_action(tree, graph, current_path, p_f)


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
def compute_uctf(tree: Tree, indices: Tensor, tree_node: Tensor, z: float) -> Tensor:
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
        scores = score_fn(tree, indices, current_tree_node, z)
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
def backup(tree: Tree, graph: TorchGraph, leaf_tree_node: Tensor, p_f: float):
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
    tree: Tree, graph: TorchGraph, path: Path, p_f: float
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
        action[is_invalid] = graph.extra["goal_node"][invalid_i].clone()

    return action
