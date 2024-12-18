from typing import Optional
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
from sop.utils.graph_torch import TorchGraph
from sop.utils.replay_buffer import TrainData
from sop.gnn.modules import DenseGATConv, DenseGATv2Conv


class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
        edge_dim: Optional[int] = None,
        edge_out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = DenseGATConv(
            in_channels,
            hidden_channels,
            heads,
            edge_dim=edge_dim,
            edge_out_dim=edge_out_dim,
        )
        self.conv2 = DenseGATConv(
            hidden_channels * heads,
            hidden_channels,
            heads,
            edge_dim=edge_dim,
            edge_out_dim=edge_out_dim,
        )
        self.conv3 = DenseGATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            edge_dim=edge_dim,
            edge_out_dim=edge_out_dim,
        )
        self.global_emb = nn.Linear(in_features=out_channels, out_features=out_channels)
        self.node_emb = nn.Linear(in_features=out_channels, out_features=out_channels)

        self.value_lin = nn.Linear(in_features=out_channels * 2, out_features=1)

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        B, N, _ = node_features.shape

        x = F.relu(self.conv1(node_features, adj, edge_features, mask))
        x = F.relu(self.conv2(x, adj, edge_features, mask))
        x = F.relu(self.conv3(x, adj, edge_features, mask))

        graph_embedding = torch.sum(x, dim=-2).unsqueeze(-2).expand(-1, N, -1)
        x = torch.cat((self.node_emb(x), self.global_emb(graph_embedding)), dim=-1)
        x = F.relu(x)
        x = self.value_lin(x).squeeze()

        return x


class GATv2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = DenseGATv2Conv(
            in_channels,
            hidden_channels,
            heads,
            edge_dim=edge_dim,
        )
        self.conv2 = DenseGATv2Conv(
            hidden_channels * heads,
            hidden_channels,
            heads,
            edge_dim=edge_dim,
        )
        self.conv3 = DenseGATv2Conv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            edge_dim=edge_dim,
        )
        self.global_emb = Linear(
            out_channels,
            out_channels,
            weight_initializer="glorot",
            bias_initializer="zeros",
        )
        self.node_emb = Linear(
            out_channels,
            out_channels,
            weight_initializer="glorot",
            bias_initializer="zeros",
        )

        self.value_lin = Linear(
            out_channels * 2, 1, weight_initializer="glorot", bias_initializer="zeros"
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        B, N, _ = node_features.shape

        x = F.relu(self.conv1(node_features, adj, edge_features, mask))
        x = F.relu(self.conv2(x, adj, edge_features, mask))
        x = F.relu(self.conv3(x, adj, edge_features, mask))

        graph_embedding = torch.sum(x, dim=-2).unsqueeze(-2).expand(-1, N, -1)
        x = torch.cat((self.node_emb(x), self.global_emb(graph_embedding)), dim=-1)
        x = F.relu(x)
        x = self.value_lin(x).squeeze()

        return x


def preprocess_features(
    graph: TorchGraph,
    current_node: torch.Tensor,
    goal_node: torch.Tensor,
    mask: torch.Tensor,
):
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    # -- Node features
    # position = graph.nodes["position"]  # [B, N, 2]
    # reward = graph.nodes["reward"]  # [B, N]

    # visited = (~mask).long()
    # visited[indices, goal_node] = 1
    # visited_one_hot = F.one_hot(visited, num_classes=2)  # [B, N, 2]

    current = torch.zeros((batch_size, num_nodes), dtype=torch.long)
    current[indices, current_node] = 1
    current_one_hot = F.one_hot(current, num_classes=2)  # [B, N, 2]

    goal = torch.zeros(batch_size, num_nodes, dtype=torch.long)
    goal[indices, goal_node] = 1
    goal_one_hot = F.one_hot(goal, num_classes=2)  # [B, N, 2]

    node_features = torch.cat(
        [
            # position,
            # visited_one_hot,
            current_one_hot,
            goal_one_hot,
        ],
        dim=-1,
    ).to(torch.float32)  # [B, N, 4]

    return node_features


def preprocess_edges(
    graph: TorchGraph, current_node: torch.Tensor, goal_node: torch.Tensor
):
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    goal_matrix = torch.zeros_like(graph.edges["distance"])
    goal_matrix[indices, goal_node, :] = 1
    goal_matrix[indices, :, goal_node] = 1

    return torch.cat(
        [graph.edges["distance"].unsqueeze(-1), goal_matrix.unsqueeze(-1)], dim=-1
    ).to(torch.float32)


def mask_adj(mask: torch.Tensor):
    # Step 1: Expand dimensions to create a square matrix
    # mask[:, :, None] => Shape: [B, N, 1]
    # mask[:, None, :] => Shape: [B, 1, N]
    adj = mask[:, :, None] * mask[:, None, :]  # Shape: [B, N, N]

    # Step 2: Set diagonal to zero to avoid self-relationships
    _, N = mask.shape
    eye = torch.eye(N, device=mask.device).bool()  # Identity matrix of shape [N, N]
    adj = adj.masked_fill(eye, 0)
    return adj


def train(model: nn.Module, optimizer, batch: TrainData):
    model.train()
    optimizer.zero_grad()

    indices = torch.arange(batch.batch_size[0])
    node_features = preprocess_features(
        batch.graph, batch.current_node, batch.goal_node, batch.mask
    )
    adj = mask_adj(batch.mask)
    edge_attr = preprocess_edges(batch.graph, batch.current_node, batch.goal_node)
    out = model(
        node_features,
        adj,
        edge_attr,
        batch.mask,
    )
    pred_Q = out[indices, batch.action]
    loss = F.mse_loss(pred_Q, batch.value)
    loss.backward()
    optimizer.step()
    return loss


def summary(model: nn.Module, batch_size: int = 32, num_nodes: int = 20):
    node_features = torch.zeros((batch_size, num_nodes, 4))
    edge_matrix = torch.zeros((batch_size, num_nodes, num_nodes))
    edge_features = torch.zeros((batch_size, num_nodes, num_nodes, 2))
    print(gnn.summary(model, node_features, edge_matrix, edge_features))
