import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from sop.utils.graph_torch import TorchGraph
from sop.utils.replay_buffer import TrainData


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

    def forward(self, x, edge_matrix, mask=None):
        x = F.relu(self.conv1(x, edge_matrix, mask))
        x = F.relu(self.conv2(x, edge_matrix, mask))
        x = F.relu(self.conv3(x, edge_matrix, mask))

        # -- Global Pool Graph node
        # graph_embedding = torch.sum(x, dim=-2).unsqueeze(-2)
        x = self.mlp(x).squeeze()
        return x


def preprocess_features(
    graph: TorchGraph, current_node: torch.Tensor, goal_node: torch.Tensor
):
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    # -- Node features
    # position = graph.nodes["position"]  # [B, N, 2]
    # reward = graph.nodes["reward"]  # [B, N]

    current = torch.zeros((batch_size, num_nodes), dtype=torch.long)
    current[indices, current_node] = 1
    current_one_hot = F.one_hot(current, num_classes=2)  # [B, N, 2]

    goal = torch.zeros(batch_size, num_nodes, dtype=torch.long)
    goal[indices, goal_node] = 1
    goal_one_hot = F.one_hot(goal, num_classes=2)  # [B, N, 2]

    node_features = torch.cat(
        [
            # position,
            # reward.unsqueeze(-1),
            current_one_hot,
            goal_one_hot,
        ],
        dim=-1,
    ).to(torch.float32)  # [B, N, 4]

    return node_features


def train(model: DenseGAT, optimizer, batch: TrainData):
    model.train()
    optimizer.zero_grad()

    indices = torch.arange(batch.batch_size[0])
    node_features = preprocess_features(
        batch.graph, batch.current_node, batch.goal_node
    )
    out = model(node_features, batch.graph.edge_matrix, batch.mask)
    pred_Q = out[indices, batch.action]
    loss = F.mse_loss(pred_Q, batch.value)

    loss.backward()
    optimizer.step()
    return loss


def summary(model: DenseGAT, batch_size: int = 32, num_nodes: int = 20):
    node_features = torch.zeros((batch_size, num_nodes, 4))
    edge_matrix = torch.zeros((batch_size, num_nodes, num_nodes))
    print(gnn.summary(model, node_features, edge_matrix))
