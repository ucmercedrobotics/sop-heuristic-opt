import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import torch_geometric
import torch_geometric.transforms as T
import torch_geometric.nn as gnn


class DenseGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()

        self.conv1 = gnn.DenseGATConv(in_channels, hidden_channels, heads)
        self.conv2 = gnn.DenseGATConv(hidden_channels * heads, hidden_channels, heads)
        self.conv3 = gnn.DenseGATConv(hidden_channels * heads, out_channels, heads=1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.GELU(),
            nn.Linear(in_features=out_channels, out_features=1),
        )

    def forward(self, x, edge_matrix):
        x = F.relu(self.conv1(x, edge_matrix))
        x = F.relu(self.conv2(x, edge_matrix))
        x = self.conv3(x, edge_matrix)
        x = self.mlp(x).squeeze()
        return x
