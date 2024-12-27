from typing import Optional
import torch
from tensordict import tensorclass
from sop.utils.path import Path, TreeStats
from sop.utils.graph_torch import TorchGraph
from torchrl.data import ReplayBuffer, LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement


@tensorclass
class TrainData:
    graph: torch.Tensor  # [B, N, N]
    current_node: torch.Tensor  # [B,]
    goal_node: torch.Tensor  # [B,]
    action: torch.Tensor  # [B,]
    q_value: torch.Tensor  # [B,]
    mask: torch.Tensor  # [B, N]


def create_replay_buffer(max_size: int, batch_size: int):
    return ReplayBuffer(
        storage=LazyMemmapStorage(max_size),
        batch_size=batch_size,
        sampler=SamplerWithoutReplacement(),
    )


def add_to_buffer(buffer: ReplayBuffer, graph: TorchGraph, path: Path, discount: float):
    batch_size, path_length = path.size()
    indices = torch.arange(batch_size)

    goal_node = path.nodes[indices, 0]
    gnn_mask = ~path.mask.clone()
    gnn_mask[indices, goal_node] = True

    # Compute discounted value for each node
    q_value_buffer = torch.zeros_like(path.rewards)
    q_values = torch.zeros((batch_size,))
    for i in reversed(range(path_length - 1)):
        q_values = path.rewards[indices, i + 1] + discount * q_values
        q_value_buffer[indices, i] = q_values

    for i in reversed(range(path_length - 1)):
        current_node = path.nodes[indices, i]
        action = path.nodes[indices, i + 1]
        q_value = q_value_buffer[indices, i]
        gnn_mask[indices, action] = True
        gnn_mask[indices, current_node] = True

        data = TrainData(
            graph=graph.clone(),
            current_node=current_node.clone(),
            goal_node=goal_node.clone(),
            action=action.clone(),
            q_value=q_value.clone(),
            mask=gnn_mask.clone(),
            batch_size=[batch_size],
        )
        buffer.extend(data)
