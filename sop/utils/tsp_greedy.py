import torch
from tsp_solver.greedy import solve_tsp

from sop.utils.graph_torch import TorchGraph
from sop.utils.path2 import Path


def run_greedy_tsp_solver(graphs: TorchGraph, start_nodes: torch.Tensor):
    batch_size, num_nodes = graphs.size()
    indices = torch.arange(batch_size)

    # -- Run solver on all graphs
    greedy_paths = torch.tensor(
        [
            solve_tsp(
                graphs.edges["distance"][i], endpoints=(start_nodes[i], start_nodes[i])
            )
            for i in range(batch_size)
        ]
    )

    # -- Add to path buffer
    path_buffer = Path.empty(batch_size, num_nodes)
    path_buffer.append(
        indices,
        start_nodes,
        cost=torch.zeros_like(start_nodes, dtype=torch.float32),
    )
    for i in range(1, greedy_paths.shape[-1]):
        prev_node = greedy_paths[indices, i - 1]
        current_node = greedy_paths[indices, i]
        cost = graphs.edges["distance"][indices, prev_node, current_node]
        path_buffer.append(indices, current_node, cost)

    return path_buffer
