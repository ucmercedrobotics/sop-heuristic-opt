import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sop.utils.graph_torch import generate_random_graph_batch
from sop.mcts.mcts_tsp import run_tsp_solver
from sop.utils.tsp_greedy import run_greedy_tsp_solver
from sop.utils.visualization import plot_solution
from sop.gnn.gat import DenseGAT, train, summary
import sop.utils.replay_buffer as rb


def main():
    # -- Config
    batch_size = 8
    num_nodes = 20
    device = "cpu"
    start_node = 0
    num_simulations = 50

    # -- Create GNN
    model = DenseGAT(in_channels=4, hidden_channels=128, out_channels=128, heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)

    # -- Create a Batch of Graphs
    start = time.time()
    graphs = generate_random_graph_batch(batch_size, num_nodes, device)
    start_nodes = torch.full(size=(batch_size,), fill_value=start_node)
    print(f"Created graphs: {time.time() - start}")

    # -- Generate Path w/ Solver
    # TODO: Make sure select works properly; Definitely doesn't, 0 appears twice for some reason...
    # TODO: I need to make sure the backup works properly
    # TODO: THERE IS A MEMORY LEAK WITH THE GNN UGH; UPDATE: I needed a torch.no_grad()
    # TODO: Mask out nodes, you don't need to compute everything... (it might actually be possible with torch_geometric...)
    start = time.time()
    with torch.no_grad():
        mcts_paths = run_tsp_solver(graphs, model, start_nodes, num_simulations)
    print(f"Solved TSP with mcts: {time.time() - start}")

    # -- Generate Path w/ solve_tsp
    start = time.time()
    greedy_paths, greedy_costs = run_greedy_tsp_solver(graphs, start_nodes)
    print(f"Solved TSP with greedy: {time.time() - start}")

    # -- Compare Costs
    mcts_costs = torch.sum(mcts_paths.costs, dim=-1)
    print(f"mcts/greedy: {torch.mean((mcts_costs / greedy_costs))}")

    # -- Visualize
    plot_solution(graphs[0], mcts_paths.nodes[0])
    plot_solution(graphs[0], greedy_paths[0])

    # -- Add to Replay Buffer
    # -- Tree based value targets? benefit would be we get to have more training targets
    # -- Use gumbel for this, muzero reanalyze?
    # The value we need is the remaining cost
    # Q_i = r_i+1 + r_i+2 ... r_i+n
    # TODO: Find a way to not have to copy the graph for each training sample..
    start = time.time()
    replay_buffer = rb.create_replay_buffer(
        max_size=batch_size * (num_nodes - 1), batch_size=batch_size
    )
    rb.add_to_buffer(replay_buffer, graphs, mcts_paths)
    print(f"Added to buffer: {time.time() - start}")

    # -- Train w/ Batch Samples
    start = time.time()
    for i, batch in enumerate(replay_buffer):
        loss = train(model, optimizer, batch)
        print(f"{i} - loss: {loss}")
    print(f"Trained on samples: {time.time() - start}")


def train_model():
    # -- Config
    batch_size = 64
    num_nodes = 20
    device = "cpu"
    start_node = 0
    num_simulations = 50
    num_epochs = 100
    eval_interval = 2

    # -- Tensorboard
    writer = SummaryWriter()

    # -- Create GNN
    model = DenseGAT(in_channels=4, hidden_channels=128, out_channels=128, heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    summary(model)

    # -- Create Replay Buffer
    replay_buffer = rb.create_replay_buffer(
        max_size=batch_size * (num_nodes - 1), batch_size=batch_size
    )

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch}...")

        # -- Generate sample graphs
        print(f"{epoch} - Generating Graphs...")
        graphs = generate_random_graph_batch(batch_size, num_nodes, device)
        start_nodes = torch.full(size=(batch_size,), fill_value=start_node)

        # -- Generate Path w/ Solver
        print(f"{epoch} - Solving TSP w/ MCTS...")
        with torch.no_grad():
            mcts_paths = run_tsp_solver(graphs, model, start_nodes, num_simulations)

        # -- Add to Replay Buffer
        print(f"{epoch} - Adding to Replay Buffer...")
        rb.add_to_buffer(replay_buffer, graphs, mcts_paths)

        # -- Train Model w/ Batch Samples
        print(f"{epoch} - Training Model w/ Batch Samples...")
        for i, batch in tqdm(enumerate(replay_buffer), total=num_nodes - 1):
            loss = train(model, optimizer, batch)

            # -- Update Tensorboard
            writer.add_scalar("Loss/train", loss, (epoch * (num_nodes - 1)) + i)
            writer.flush()

        # -- Compare to Greedy
        if (epoch + 1) % eval_interval == 0:
            print(f"{epoch} - Comparing against greedy...")
            greedy_paths, greedy_costs = run_greedy_tsp_solver(graphs, start_nodes)
            mcts_costs = torch.sum(mcts_paths.costs, dim=-1)
            writer.add_scalar(
                "Eval/ratio", torch.mean((mcts_costs / greedy_costs)), epoch
            )
            writer.flush()

        print(f"{epoch} - Epoch Complete!")

    # -- Test out the model
    print("Testing the model...")
    graphs = generate_random_graph_batch(batch_size, num_nodes, device)
    start_nodes = torch.full(size=(batch_size,), fill_value=start_node)
    with torch.no_grad():
        mcts_paths = run_tsp_solver(graphs, model, start_nodes, num_simulations)
    print(torch.sum(mcts_paths.costs, dim=-1))
    plot_solution(graphs[0], mcts_paths.nodes[0])


if __name__ == "__main__":
    # import time

    start = time.time()
    # main()
    train_model()
    print(f"Time elapsed: {time.time() - start}")
