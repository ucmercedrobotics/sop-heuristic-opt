import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sop.utils.path import Path, TreeStats
from sop.utils.graph_torch import TorchGraph
from sop.mcts.core import QTransformFn, RewardFn, ROOT_INDEX
from sop.mcts.core import Tree, select, expand, backup, backupN


def tsp_greedy_gnn_solver(
    model: nn.Module,
    graph: TorchGraph,
    start_graph_node: Tensor,
    device: str = "cpu",
) -> Path:
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    goal_graph_node = start_graph_node.clone()
    current_graph_node = start_graph_node.clone()

    path = Path.empty(batch_size, num_nodes, device)
    path.append(
        indices, current_graph_node, reward=torch.zeros((batch_size,), device=device)
    )

    for step in tqdm(range(num_nodes)):
        if step < num_nodes - 1:
            Q = predict_Q(model, graph, path, current_graph_node, goal_graph_node)
            masked_Q = torch.masked_fill(Q, path.mask, -torch.inf)
            next_graph_node = torch.argmax(masked_Q, axis=-1)
        else:
            next_graph_node = start_graph_node

        reward = tsp_reward_fn(graph, current_graph_node, next_graph_node)
        path.append(indices, next_graph_node, reward)
        current_graph_node = next_graph_node

    return path


def tsp_mcts_gnn_solver(
    model: nn.Module,
    graph: TorchGraph,
    start_graph_node: Tensor,
    num_simulations: int,
    discount: float = 0.99,
    z: float = 0.1,
    device: str = "cpu",
) -> Path:
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    goal_graph_node = start_graph_node.clone()
    current_graph_node = start_graph_node.clone()

    path = Path.empty(batch_size, num_nodes, device)
    path.append(
        indices, current_graph_node, reward=torch.zeros((batch_size,), device=device)
    )
    summary = TreeStats.empty(batch_size, num_nodes, device)

    for step in tqdm(range(num_nodes)):
        if step < num_nodes - 1:
            next_graph_node, tree = MCTS_TSP(
                model,
                graph,
                path,
                current_graph_node,
                goal_graph_node,
                num_simulations,
                discount,
                z,
                device,
            )
            summary.append(
                indices,
                tree.node_value[indices, ROOT_INDEX],
                tree.children_Q_values[indices, ROOT_INDEX],
            )
        else:
            next_graph_node = start_graph_node

        reward = tsp_reward_fn(graph, current_graph_node, next_graph_node)
        path.append(indices, next_graph_node, reward)
        current_graph_node = next_graph_node

    return path, summary


def MCTS_TSP(
    model: nn.Module,
    graph: TorchGraph,
    current_path: Path,
    current_graph_node: torch.Tensor,
    goal_graph_node: torch.Tensor,
    num_simulations: int,
    discount: float,
    z: float,
    device: str,
):
    batch_size, num_nodes = graph.size()

    tree = Tree.instantiate_from_root(
        current_graph_node, graph, num_simulations, device
    )

    def tsp_scoring_fn(indices, tree, current_tree_node):
        return compute_ucb(indices, tree, current_tree_node, z)

    for k in range(num_simulations):
        # start_all = time.time()
        # Select parent tree node
        # start = time.time()
        parent_tree_node, tree_path, is_expanded, is_finished = select(
            tree, graph, current_path, tsp_scoring_fn, tsp_reward_fn
        )
        # print(f"{k} - Select: {time.time() - start}")

        # Predict Q values
        # start = time.time()
        indices = torch.arange(batch_size)
        Q = torch.full((batch_size, num_nodes), -torch.inf, dtype=torch.float32)

        # There are 2 cases for Q values:
        # Case 1: node is unexpanded -> expand node
        is_leaf = torch.logical_and(~is_expanded, ~is_finished)
        leaf_i = indices[is_leaf]
        if leaf_i.numel() > 0:
            pred_Q = predict_Q(
                model,
                graph[leaf_i],
                tree_path[leaf_i],
                tree.node_mapping[leaf_i, parent_tree_node[leaf_i]],
                goal_graph_node[leaf_i],
            )
            Q[leaf_i] = pred_Q

        # Case 2: node is last -> compute true Q value
        is_last = torch.logical_and(~is_expanded, is_finished)
        last_i = indices[is_last]
        if last_i.numel() > 0:
            complete_Q = completed_Q(
                graph[last_i],
                tree_path[last_i],
                tree.node_mapping[last_i, parent_tree_node[last_i]],
                goal_graph_node[last_i],
                discount,
                tsp_reward_fn,
            )
            # data["q_value"][last_i] = complete_Q
            Q[last_i] = complete_Q
        # print(f"{k} - Policy: {time.time() - start}")

        # Expand and update nodes
        # start = time.time()
        should_expand = torch.logical_or(is_leaf, is_last)
        is_valid = ~tree_path.mask
        expand(tree, parent_tree_node, Q, is_valid, should_expand)
        # print(f"{k} - Expand: {time.time() - start}")

        # Backup
        # start = time.time()
        backup(tree, graph, parent_tree_node, discount, tsp_reward_fn)
        backupN(tree, tree_path, parent_tree_node)
        # print(f"{k} - Backup: {time.time() - start}")
        # print(f"{k} - Total: {time.time() - start_all}")

    # Select action
    return select_action(tree), tree


# -- Interface functions


def q_transform_by_parent_and_siblings(
    indices: Tensor, tree: Tree, current_tree_node: Tensor, epsilon: float = 1e-8
):
    Q = tree.children_Q_values[indices, current_tree_node]
    V = tree.node_value[indices, current_tree_node]
    max_value = torch.maximum(V, torch.max(Q, dim=-1).values).unsqueeze(-1)
    min_value = torch.minimum(V, safe_min(Q, dim=-1).values).unsqueeze(-1)
    normalized_Q = (Q - min_value) / torch.maximum(
        max_value - min_value, torch.tensor(epsilon)
    )
    return normalized_Q


def no_qtransform(indices: Tensor, tree: Tree, current_tree_node: Tensor):
    return tree.children_Q_values[indices, current_tree_node]


def compute_ucb(
    indices: Tensor,
    tree: Tree,
    current_tree_node: Tensor,
    z: float,
    q_transform: QTransformFn = q_transform_by_parent_and_siblings,
) -> Tensor:
    Q = q_transform(indices, tree, current_tree_node)
    N = tree.children_visit_counts[indices, current_tree_node]
    t = tree.visit_count[indices, ROOT_INDEX]

    return Q + z * torch.sqrt(torch.log(t + 1e-5).unsqueeze(-1) / N)


def safe_min(x: Tensor, dim: int):
    min_mask = torch.isneginf(x)
    return x.masked_fill(min_mask, torch.inf).min(dim)


def tsp_reward_fn(graph: TorchGraph, current_node: Tensor, next_node: Tensor) -> Tensor:
    return -graph.edges["distance"][
        torch.arange(graph.size()[0]), current_node, next_node
    ]


# -- Predict


def create_node_features(
    graph: TorchGraph, current_node: Tensor, goal_node: Tensor
) -> Tensor:
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    current = torch.zeros((batch_size, num_nodes), dtype=torch.long)
    current[indices, current_node] = 1

    goal = torch.zeros(batch_size, num_nodes, dtype=torch.long)
    goal[indices, goal_node] = 1

    return torch.cat(
        [
            F.one_hot(current, num_classes=2),
            F.one_hot(goal, num_classes=2),
        ],
        dim=-1,
    ).to(torch.float32)  # [B, N, 4]


def create_edge_features(
    graph: TorchGraph, current_node: Tensor, goal_node: Tensor
) -> Tensor:
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    current_matrix = torch.zeros_like(graph.edges["distance"])
    current_matrix[indices, current_node, :] = 1
    current_matrix[indices, :, current_node] = 1

    goal_matrix = torch.zeros_like(graph.edges["distance"])
    goal_matrix[indices, goal_node, :] = 1
    goal_matrix[indices, :, goal_node] = 1

    return torch.cat(
        [
            graph.edges["distance"].unsqueeze(-1),
            current_matrix.unsqueeze(-1),
            goal_matrix.unsqueeze(-1),
        ],
        dim=-1,
    ).to(torch.float32)


def create_node_mask(path: Path, current_node: Tensor, goal_node: Tensor) -> Tensor:
    indices = torch.arange(path.size()[0])
    mask = ~path.mask
    mask[indices, current_node] = True
    mask[indices, goal_node] = True
    return mask


def create_adj_matrix(mask: Tensor) -> Tensor:
    # Expand dimensions to create a square matrix
    # mask[:, :, None] => Shape: [B, N, 1]
    # mask[:, None, :] => Shape: [B, 1, N]
    adj = mask[:, :, None] * mask[:, None, :]  # Shape: [B, N, N]

    # Set diagonal to zero
    _, N = mask.shape
    eye = torch.eye(N, device=mask.device).bool()  # Identity matrix of shape [N, N]
    adj = adj.masked_fill(eye, 0)
    return adj


def predict_Q(
    model: nn.Module,
    graph: TorchGraph,
    path: Path,
    current_node: Tensor,
    goal_node: Tensor,
) -> Tensor:
    # Preprocessing
    node_features = create_node_features(graph, current_node, goal_node)
    edge_features = create_edge_features(graph, current_node, goal_node)
    node_mask = create_node_mask(path, current_node, goal_node)
    adj = create_adj_matrix(node_mask)

    # Model
    with torch.no_grad():
        Q = model(node_features, adj, edge_features, node_mask)
    return Q


def completed_Q(
    graph: TorchGraph,
    path: Path,
    current_node: Tensor,
    goal_node: Tensor,
    discount: float,
    reward_fn: RewardFn = tsp_reward_fn,
) -> Tensor:
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)
    Q = torch.full((batch_size, num_nodes), -torch.inf)

    # Only one node left
    next_node = torch.argmax((~path.mask).float(), dim=-1)
    q = reward_fn(graph, current_node, next_node) + (
        discount * reward_fn(graph, next_node, goal_node)
    )
    Q[indices, next_node] = q

    return Q


# -- Select action
def select_action(tree: Tree):
    batch_size, _, _ = tree.size()
    scores = tree.children_Q_values[torch.arange(batch_size), ROOT_INDEX]
    action = torch.argmax(scores, axis=-1)
    return action
