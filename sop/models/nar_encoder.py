from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import Linear

from sop.utils.graph_torch import TorchGraph
from sop.models.modules import DenseGATv2Conv


class GATEmbedding(nn.Module):
    def __init__(
        self, node_dim: int, out_dim: int, edge_dim: int, act_fn: Callable = F.silu
    ):
        super().__init__()

        self.conv = DenseGATv2Conv(
            node_dim,
            out_dim,
            heads=1,
            edge_dim=edge_dim,
        )
        self.global_emb = Linear(
            out_dim,
            out_dim,
            weight_initializer="glorot",
            bias_initializer="zeros",
        )
        self.node_emb = Linear(
            out_dim,
            out_dim,
            weight_initializer="glorot",
            bias_initializer="zeros",
        )
        self.act_fn = act_fn

    def forward(
        self,
        node_features: Tensor,
        adj: Tensor,
        edge_features: Tensor,
        mask: Optional[Tensor] = None,
    ):
        x = self.act_fn(self.conv(node_features, adj, edge_features, mask))
        print(x.shape)


class NAREncoder(nn.Module):
    def __init__(
        self,
        node_embedder: nn.Module,
    ):
        super().__init__()

        # Node Embeddings -> edge embeddings -> MLP -> heuristic
        self.node_embedder = node_embedder

    def forward(
        self,
        node_features: Tensor,
        adj: Tensor,
        edge_features: Tensor,
        mask: Optional[Tensor] = None,
    ):
        x = self.node_embedder(node_features, adj, edge_features, mask)

        return x
