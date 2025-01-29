from typing import Callable, Tuple

import torch
from torch import Tensor
from tensordict import tensorclass

from sop.utils.graph_torch import TorchGraph
from sop.utils.path import Path

"""
A set of resuable functions for batched MCTS.

Based heavily on: https://github.com/google-deepmind/mctx

TODO:
- Add comments
- Transform Q-values
- Support sop
"""

# -- Interface functions
# ScoringFn(indices, tree, current_tree_node) -> scores
ScoringFn = Callable[[Tensor, "Tree", Tensor], Tensor]

# QTransformFn(indices, tree, current_tree_node) -> transformed_Q
QTransformFn = Callable[[Tensor, "Tree", Tensor], Tensor]

# RewardFn(graph, parent_graph_node, next_graph_node) -> costs
RewardFn = Callable[[TorchGraph, Tensor, Tensor], Tensor]

# -- CONSTANTS
ROOT_INDEX = 0
UNVISITED = -1
NO_PARENT = -1


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
            num_nodes=torch.ones((batch_size,), dtype=torch.long),
            batch_size=[batch_size],
        )

        tree.node_mapping[:, ROOT_INDEX] = root_graph_node
        tree.visit_count[:, ROOT_INDEX] = 1

        return tree

    def size(self):
        # [batch_size, num_simulations, num_graph_nodes]
        return self.children_index.shape


# -- Select


def select(
    tree: Tree,
    graph: TorchGraph,
    current_path: Path,
    scoring_fn: ScoringFn,
    reward_fn: RewardFn,
) -> Tuple[Tensor, Path, Tensor, Tensor]:
    """Traverses the tree until it reaches an unexpanded node or reaches the end.

    There are 3 cases:
    1. The node is unexpanded (!is_expanded)
    2. The node is the last node in path and unexpanded (!is_expanded and is_finished)
    3. The node is the last node in path and expanded (is_expanded and is_finished)

    Args:
        tree: batched MCTS Tree state
        graph: batched graph state
        current_path: the current path taken
        scoring_fn: TODO
        reward_fn: TODO
    """
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    # Loop State
    parent_tree_node = torch.full((batch_size,), ROOT_INDEX, dtype=torch.long)
    is_expanded = torch.zeros((batch_size,), dtype=torch.bool)
    is_finished = torch.zeros((batch_size,), dtype=torch.bool)
    tree_path = current_path.clone()

    while indices.numel() > 0:
        # 1. check if node is expanded
        is_expanded[indices] = tree.visit_count[indices, parent_tree_node[indices]] != 1
        # 2. check if node is the last node needed
        is_finished[indices] = tree_path.length[indices] >= num_nodes - 1

        # Continue traversal if expanded and not finished
        is_continuing = torch.logical_and(is_expanded[indices], ~is_finished[indices])

        # Mask finished indices
        indices = indices[is_continuing]
        if indices.numel() == 0:
            break

        # Compute scores
        scores = scoring_fn(indices, tree, parent_tree_node[indices])
        masked_scores = torch.masked_fill(scores, tree_path.mask[indices], -torch.inf)

        # Choose next best node
        next_graph_node = torch.argmax(masked_scores, axis=-1)
        next_tree_node = tree.children_index[
            indices, parent_tree_node[indices], next_graph_node
        ]

        # Add to tree path
        current_graph_node = tree.node_mapping[indices, parent_tree_node[indices]]
        reward = reward_fn(graph[indices], current_graph_node, next_graph_node)
        tree_path.append(indices, next_graph_node, reward)

        # Update loop state
        parent_tree_node[indices] = next_tree_node

    return parent_tree_node, tree_path, is_expanded, is_finished


# -- Expand


def add_node(
    indices: Tensor,
    tree: Tree,
    parent_tree_node: Tensor,
    new_tree_node: Tensor,
    new_graph_node: Tensor,
    q: Tensor,
):
    tree.node_mapping[indices, new_tree_node] = new_graph_node
    tree.action_from_parent[indices, new_tree_node] = new_graph_node
    tree.parent_index[indices, new_tree_node] = parent_tree_node
    tree.children_index[indices, parent_tree_node, new_graph_node] = new_tree_node

    tree.visit_count[indices, new_tree_node] = 1
    tree.children_visit_counts[indices, parent_tree_node, new_graph_node] = 1

    tree.children_Q_values[indices, parent_tree_node, new_graph_node] = q


def update_node(
    indices: torch.Tensor,
    tree: Tree,
    parent_tree_node: torch.Tensor,
    current_graph_node: torch.Tensor,
):
    current_tree_node = tree.children_index[
        indices, parent_tree_node, current_graph_node
    ]

    tree.visit_count[indices, current_tree_node] += 1
    tree.children_visit_counts[indices, parent_tree_node, current_graph_node] += 1


def expand(
    tree: Tree,
    parent_tree_node: Tensor,
    Q: Tensor,
    is_valid: Tensor,
    should_expand: Tensor,
):
    batch_size, _, num_nodes = tree.size()
    # batch: [[0,0,0, ...], [1,1,1, ...], ...]
    batch_indices = torch.arange(batch_size).unsqueeze(-1).expand((-1, num_nodes))
    # graph nodes: [[0,1,2, ...], [0,1,2, ...], ...]
    new_graph_nodes = torch.arange(num_nodes).unsqueeze(0).expand((batch_size, -1))
    # num_nodes: [10,20, ...] -> tree nodes: [[10,11,12, ...], [21,22,23, ...], ...]
    new_tree_nodes = new_graph_nodes + tree.num_nodes.unsqueeze(-1)

    # There are 2 cases for expanding:
    # Case 1: node is valid and should expand -> add node
    add_mask = torch.logical_and(is_valid, should_expand.unsqueeze(-1))
    add_i = batch_indices[add_mask]
    if add_i.numel() > 0:
        pt = parent_tree_node[add_i]
        nt = new_tree_nodes[add_mask]
        ng = new_graph_nodes[add_mask]
        q = Q[add_i, ng]
        add_node(add_i, tree, pt, nt, ng, q)

    # Case 2: node is valid and already expanded -> update node
    update_mask = torch.logical_and(is_valid, ~should_expand.unsqueeze(-1))
    update_i = batch_indices[update_mask]
    if update_i.numel() > 0:
        pt = parent_tree_node[update_i]
        cg = new_graph_nodes[update_mask]
        update_node(update_i, tree, pt, cg)

    tree.num_nodes += num_nodes


# -- Backup


def backup(
    tree: Tree,
    graph: TorchGraph,
    current_tree_node: Tensor,
    discount: float,
    reward_fn: RewardFn,
):
    batch_size, _, _ = tree.size()
    indices = torch.arange(batch_size)

    # Compute value for current V(S) = max Q(S, A)
    children_Q = tree.children_Q_values[indices, current_tree_node]
    max_child = torch.argmax(children_Q, dim=-1)
    new_Q = children_Q[indices, max_child]

    # Loop state
    current_tree_node = current_tree_node
    parent_tree_node = tree.parent_index[indices, current_tree_node]
    new_Q = new_Q

    while indices.numel() > 0:
        # Check if new Q is higher than current value; becomes new value
        current_V = tree.node_value[indices, current_tree_node]
        is_higher = new_Q > current_V
        indices = indices[is_higher]
        if indices.numel() == 0:
            break
        current_tree_node = current_tree_node[is_higher]
        parent_tree_node = parent_tree_node[is_higher]
        new_Q = new_Q[is_higher]

        # Update current_V
        tree.node_value[indices, current_tree_node] = new_Q

        # Check if parent is root node
        has_parent = parent_tree_node != NO_PARENT
        indices = indices[has_parent]
        if indices.numel() == 0:
            break
        current_tree_node = current_tree_node[has_parent]
        parent_tree_node = parent_tree_node[has_parent]
        new_Q = new_Q[has_parent]

        # Update Q value for parent Q(S, S') = R + discount * V(S')
        parent_graph_node = tree.node_mapping[indices, parent_tree_node]
        current_graph_node = tree.node_mapping[indices, current_tree_node]
        R = reward_fn(graph[indices], parent_graph_node, current_graph_node)
        current_Q = R + (discount * new_Q)
        tree.children_Q_values[indices, parent_tree_node, current_graph_node] = (
            current_Q
        )

        # Update Loop State
        current_tree_node = parent_tree_node
        parent_tree_node = tree.parent_index[indices, parent_tree_node]
        new_Q = current_Q


def backupN(tree: Tree, tree_path: Path, current_tree_node: Tensor):
    batch_size, _, _ = tree.size()
    indices = torch.arange(batch_size)

    # Loop state
    # Num visits is the number of valid nodes since we only expand those
    num_visits = torch.sum(tree_path.mask, axis=-1)
    current_tree_node = current_tree_node
    parent_tree_node = tree.parent_index[indices, current_tree_node]

    # Backup visit count to root
    while indices.numel() > 0:
        # Update node
        tree.visit_count[indices, current_tree_node] += num_visits

        # Mask root nodes
        has_parent = parent_tree_node != NO_PARENT
        indices = indices[has_parent]
        if indices.numel() == 0:
            break
        num_visits = num_visits[has_parent]
        current_tree_node = current_tree_node[has_parent]
        parent_tree_node = parent_tree_node[has_parent]

        # Update parent
        action = tree.action_from_parent[indices, current_tree_node]
        tree.children_visit_counts[indices, parent_tree_node, action] += num_visits

        # Update Loop state
        current_tree_node = parent_tree_node
        parent_tree_node = tree.parent_index[indices, parent_tree_node]
