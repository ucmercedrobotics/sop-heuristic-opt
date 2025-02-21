import torch
import torch.nn.functional as F
from sop.utils.graph import TorchGraph


def preprocess_graph_mean(graph: TorchGraph):
    reward = graph.nodes["reward"]
    start_node = graph.extra["start_node"]
    goal_node = graph.extra["goal_node"]
    samples = graph.edges["samples"]

    # node features: reward, is_start, is_goal
    # edge features: normalized mean samples, is_start_edge, is_goal_edge
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)
    # Adjacency Matrix
    adj = graph.edges["adj"]
    # Node Features
    is_start_node = torch.zeros((batch_size, num_nodes), dtype=torch.long)
    is_goal_node = torch.zeros((batch_size, num_nodes), dtype=torch.long)
    is_start_node[indices, start_node] = 1
    is_goal_node[indices, goal_node] = 1
    node_features = torch.cat(
        [
            reward.unsqueeze(-1),
            F.one_hot(is_start_node, num_classes=2),
            F.one_hot(is_goal_node, num_classes=2),
        ],
        dim=-1,
    )
    # Edge Features
    sample_mean = samples.mean(-1)
    is_start_edge = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.long)
    is_goal_edge = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.long)
    is_start_edge[indices, start_node] = 1
    is_goal_edge[indices, :, goal_node] = 1
    edge_features = torch.cat(
        [
            sample_mean.unsqueeze(-1),
            F.one_hot(is_start_edge, num_classes=2),
            F.one_hot(is_goal_edge, num_classes=2),
        ],
        dim=-1,
    )

    return node_features, edge_features, adj


def preprocess_graph_normalize_budget(graph: TorchGraph):
    reward = graph.nodes["reward"]
    start_node = graph.extra["start_node"]
    goal_node = graph.extra["goal_node"]
    samples = graph.edges["samples"]
    budget = graph.extra["budget"]

    # node features: reward, is_start, is_goal
    # edge features: normalized mean samples, is_start_edge, is_goal_edge
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)
    # Adjacency Matrix
    adj = graph.edges["adj"]
    # Node Features
    is_start_node = torch.zeros((batch_size, num_nodes), dtype=torch.long)
    is_goal_node = torch.zeros((batch_size, num_nodes), dtype=torch.long)
    is_start_node[indices, start_node] = 1
    is_goal_node[indices, goal_node] = 1
    node_features = torch.cat(
        [
            reward.unsqueeze(-1),
            F.one_hot(is_start_node, num_classes=2),
            F.one_hot(is_goal_node, num_classes=2),
        ],
        dim=-1,
    )
    # Edge Features
    weights = samples.mean(-1) / budget.unsqueeze(-1).unsqueeze(-1)
    is_start_edge = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.long)
    is_goal_edge = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.long)
    is_start_edge[indices, start_node] = 1
    is_goal_edge[indices, :, goal_node] = 1
    edge_features = torch.cat(
        [
            weights.unsqueeze(-1),
            F.one_hot(is_start_edge, num_classes=2),
            F.one_hot(is_goal_edge, num_classes=2),
        ],
        dim=-1,
    )

    return node_features, edge_features, adj
