from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros


class DenseGATv2Conv(torch.nn.Module):
    r"""Adapted from torch_geometric.nn.Gatv2Conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_l = Linear(
            in_channels, heads * out_channels, bias=bias, weight_initializer="glorot"
        )
        self.lin_r = Linear(
            in_channels, heads * out_channels, bias=bias, weight_initializer="glorot"
        )
        self.att = Parameter(torch.empty(1, 1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(
                edge_dim,
                heads * self.out_channels,
                bias=False,
                weight_initializer="glorot",
            )
        else:
            self.lin_edge = None

        # The number of output channels:
        total_out_channels = out_channels * (heads if concat else 1)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        edge_attr: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        add_loop: bool = True,
    ):
        r"""Forward pass.

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            edge_attr (torch.Tensor, optional): Edge feature tensor
                :math:`\mathbf{E} \in \mathbb{R}^{B \times N \times N \times F}`,
                with edge feature dimension :math:`F`.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x  # [B, N, F]
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # [B, N, N]

        if edge_attr is not None:
            edge_attr = (
                edge_attr.unsqueeze(0) if edge_attr.dim() == 3 else edge_attr
            )  # [B, N, N, F]

        H, C = self.heads, self.out_channels
        B, N, _ = x.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1.0

        x_l = self.lin_l(x).view(B, N, H, C)  # [B, N, H, C]
        x_r = self.lin_r(x).view(B, N, H, C)  # [B, N, H, C]
        alpha = x_l.unsqueeze(1) + x_r.unsqueeze(2)  # [B, N, N, H, C]

        # Edge-level attention coefficients
        if edge_attr is not None and self.lin_edge is not None:
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(B, N, N, H, C)
            alpha = alpha + edge_attr

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = (alpha * self.att).sum(dim=-1)
        alpha = alpha.masked_fill(adj.unsqueeze(-1) == 0, -torch.inf)
        alpha = alpha.softmax(dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.matmul(alpha.movedim(3, 1), x_r.movedim(2, 1))
        out = out.movedim(1, 2)  # [B, N, H, C]

        if self.concat:
            out = out.reshape(B, N, H * C)
        else:
            out = out.mean(dim=2)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )
