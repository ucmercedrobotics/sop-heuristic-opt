import time

import torch
from tsp_solver.greedy import solve_tsp

from sop.utils.graph_torch import generate_random_graph_batch
from sop.mcts.mcts_tsp import run_tsp_solver
from sop.utils.visualization import plot_solution
from sop.gnn.gat import DenseGAT


def main():
    # -- Config
    batch_size = 10
    num_nodes = 20
    device = "cpu"
    start_node = 2
    num_simulations = 50

    # -- Create GNN
    dense_gat = DenseGAT(in_channels=9, hidden_channels=128, out_channels=128, heads=2)

    # -- Create a Batch of Graphs
    start = time.time()
    graphs = generate_random_graph_batch(batch_size, num_nodes, device)
    start_nodes = torch.full(size=(batch_size,), fill_value=start_node)
    print(f"Created graphs: {time.time() - start}")

    # -- Generate Path w/ Solver
    start = time.time()
    mcts_paths = run_tsp_solver(graphs, dense_gat, start_nodes, num_simulations)
    print(f"Solved TSP with mcts: {time.time() - start}")

    # -- Generate Path w/ solve_tsp
    start = time.time()
    greedy_paths = torch.tensor(
        [
            solve_tsp(graphs.edge_matrix[i], endpoints=(start_node, start_node))
            for i in range(batch_size)
        ]
    )
    print(f"Solved TSP with greedy: {time.time() - start}")

    # -- Calculate costs
    mcts_costs = torch.sum(mcts_paths.costs, dim=-1)

    indices = torch.arange(batch_size)
    greedy_costs = torch.zeros((batch_size,))
    for i in range(greedy_paths.shape[-1] - 1):
        greedy_costs += graphs.edge_matrix[
            indices, greedy_paths[indices, i], greedy_paths[indices, i + 1]
        ]
    print(f"MCTS costs: {mcts_costs}")
    print(f"Greedy costs: {greedy_costs}")
    print(f"mcts/greedy: {(mcts_costs / greedy_costs)}")

    # -- Visualize
    plot_solution(graphs[0], mcts_paths.nodes[0])
    plot_solution(graphs[0], greedy_paths[0])

    # -- Tree based value targets? benefit would be we get to have more training targets
    # -- Use gumbel for this, muzero reanalyze?
    # The value we need is the remaining cost
    # Q_i = r_i+1 + r_i+2 ... r_i+n

    # -- Add to Replay Buffer
    # -- Train w/ Batch Samples
    ...


if __name__ == "__main__":
    import time

    start = time.time()
    main()
    print(f"Time elapsed: {time.time() - start}")
