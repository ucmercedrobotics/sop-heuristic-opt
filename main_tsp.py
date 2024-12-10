import time

import torch
from sop.utils.graph_torch import generate_random_graph_batch
from sop.mcts.mcts_tsp import run_tsp_solver


def main():
    # -- Config
    batch_size = 100
    num_nodes = 20
    device = "cpu"
    start_node = 2
    num_simulations = 10

    # -- Create a Batch of Graphs
    start = time.time()
    graphs = generate_random_graph_batch(batch_size, num_nodes, device)
    start_nodes = torch.full(size=(batch_size,), fill_value=start_node)
    print(f"Created graphs: {time.time() - start}")
    # -- Generate Path w/ Solver
    start = time.time()
    path = run_tsp_solver(graphs, start_nodes, num_simulations)
    print(f"Solved TSP: {time.time() - start}")
    print(path.nodes)
    print(torch.sum(path.costs, dim=-1))
    # -- Add to Replay Buffer
    # -- Train w/ Batch Samples
    ...


if __name__ == "__main__":
    import time

    start = time.time()
    main()
    print(f"Time elapsed: {time.time() - start}")
