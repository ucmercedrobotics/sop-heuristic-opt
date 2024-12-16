import torch
from tsp_solver.greedy import solve_tsp

from sop.utils.graph_torch import TorchGraph


def run_greedy_tsp_solver(graphs: TorchGraph, start_nodes: torch.Tensor):
    batch_size, num_nodes = graphs.size()
    indices = torch.arange(batch_size)

    # -- Run solver on all graphs
    paths = torch.tensor(
        [
            solve_tsp(graphs.edge_matrix[i], endpoints=(start_nodes[i], start_nodes[i]))
            for i in range(batch_size)
        ]
    )

    # -- Calcualte costs
    costs = torch.zeros((batch_size,))
    for i in range(paths.shape[-1] - 1):
        costs += graphs.edge_matrix[indices, paths[indices, i], paths[indices, i + 1]]

    return paths, costs
