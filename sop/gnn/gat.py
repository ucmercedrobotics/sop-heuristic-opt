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
