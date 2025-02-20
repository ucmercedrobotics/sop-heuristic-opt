import torch
from tensordict import tensorclass, TensorDict
from sop.utils.sample import sample_costs


@tensorclass
class TorchGraph:
    nodes: TensorDict
    edges: TensorDict
    extra: TensorDict

    def size(self, key="adj"):
        """Returns size of graph as (batch_size, num_nodes)"""
        return self.edges[key].shape[0], self.edges[key].shape[-1]


# -- Generation Utils
def generate_uniform_positions(size: tuple[int]) -> torch.Tensor:
    """Creates 2D positions for each node in the graph."""
    xs = torch.rand(size)  # (...size,)
    ys = torch.rand(size)  # (...size,)
    return torch.stack([xs, ys], dim=-2)  # (2, size...)


def generate_uniform_reward(size: tuple) -> torch.Tensor:
    """Computes random reward for each node in the graph."""
    return torch.rand(size)


def compute_distances(x: torch.Tensor, p: int = 2):
    return torch.cdist(x, x, p=p)


def complete_adjacency_matrix(batch_size: int, num_nodes: int):
    adj = torch.ones((num_nodes, num_nodes))
    adj.fill_diagonal_(0)
    adj = adj.unsqueeze(0).expand((batch_size, num_nodes, num_nodes))
    return adj


# -- SOP
def generate_sop_graphs(
    batch_size: int,
    num_nodes: int,
    start_node: int,
    goal_node: int,
    budget: float,
    num_samples: int,
    kappa: float,
) -> TorchGraph:
    # -- Nodes
    positions = generate_uniform_positions(size=(batch_size, num_nodes))
    rewards = generate_uniform_reward(size=(batch_size, num_nodes))
    # Convert from (B, 2, N) -> (B, N, 2)
    positions = positions.permute(0, 2, 1)

    # -- Edges
    # p=2 is L2 norm, or euclidean distance
    adj = complete_adjacency_matrix(batch_size, num_nodes)
    distances = compute_distances(positions, p=2)
    samples = sample_costs(distances, num_samples, kappa)

    # -- Extra
    start_nodes = torch.full((batch_size,), start_node)
    goal_nodes = torch.full((batch_size,), goal_node)
    budgets = torch.full((batch_size,), budget, dtype=torch.float32)

    # -- TensorDicts
    nodes = TensorDict(
        {"position": positions, "reward": rewards},
        batch_size=[batch_size],
    )
    edges = TensorDict(
        {"adj": adj, "distance": distances, "samples": samples},
        batch_size=[batch_size],
    )
    extra = TensorDict(
        {"start_node": start_nodes, "goal_node": goal_nodes, "budget": budgets},
        batch_size=[batch_size],
    )

    return TorchGraph(nodes=nodes, edges=edges, extra=extra, batch_size=[batch_size])
