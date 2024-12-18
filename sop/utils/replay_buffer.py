import torch
from tensordict import tensorclass
from sop.utils.path2 import Path
from sop.utils.graph_torch import TorchGraph
from torchrl.data import ReplayBuffer, LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement


@tensorclass
class TrainData:
    graph: torch.Tensor  # [B, N, N]
    current_node: torch.Tensor  # [B,]
    goal_node: torch.Tensor  # [B,]
    action: torch.Tensor  # [B,]
    value: torch.Tensor  # [B,]
    mask: torch.Tensor  # [B, N]


def create_replay_buffer(max_size: int, batch_size: int):
    return ReplayBuffer(
        storage=LazyMemmapStorage(max_size),
        batch_size=batch_size,
        sampler=SamplerWithoutReplacement(),
    )


def add_to_buffer(buffer: ReplayBuffer, graph: TorchGraph, path: Path):
    batch_size, path_length = path.size()
    indices = torch.arange(batch_size)

    goal_node = path.nodes[indices, 0]
    gnn_mask = ~path.mask.clone()
    gnn_mask[indices, goal_node] = True

    for i in reversed(range(1, path_length - 1)):
        current_node = path.nodes[indices, i - 1]
        action = path.nodes[indices, i]
        value = torch.sum(path.costs[indices, i + 1 :], axis=-1)
        print(path.costs)
        print(value)
        gnn_mask[indices, action] = True
        gnn_mask[indices, current_node] = True

        data = TrainData(
            graph=graph.clone(),
            current_node=current_node.clone(),
            goal_node=goal_node.clone(),
            action=action.clone(),
            value=value.clone(),
            mask=gnn_mask.clone(),
            batch_size=[batch_size],
        )
        buffer.extend(data)
