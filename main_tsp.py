import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sop.utils.graph_torch import generate_random_graph_batch
from sop.mcts.mcts_tsp import run_tsp_solver, run_greedy_gnn_solver
from sop.utils.tsp_greedy import run_greedy_tsp_solver
from sop.utils.visualization import plot_solution
from sop.gnn.gat import GAT, GATv2, train, summary
import sop.utils.replay_buffer as rb

"""
TODO List:
run_tsp_solver()
- TODO: Make sure select works properly; Definitely doesn't, 0 appears twice for some reason...
- TODO: I need to make sure the backup works properly
- TODO: Mask out nodes to save computation
- TODO: Add torch.no_grad() to the run_tsp_solver function itself

replay_buffer
- TODO: Tree based value targets? benefit would be we get to have more training targets
- TODO: Use gumbel for this, muzero reanalyze?
- TODO: Chance value to discounted sum
"""


def train_model():
    # -- Config
    batch_size = 8
    num_nodes = 20
    device = "cpu"
    start_node = 0
    num_simulations = 50
    num_epochs = 1000
    eval_interval = 25

    # -- Tensorboard
    writer = SummaryWriter()

    # -- Create GNN
    model = GAT(
        in_channels=4,
        hidden_channels=64,
        out_channels=64,
        heads=4,
        edge_dim=2,
        edge_out_dim=32,
    )
    # model = GATv2(
    #     in_channels=4,
    #     hidden_channels=64,
    #     out_channels=64,
    #     heads=2,
    #     edge_dim=2,
    # )
    model = torch.jit.script(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
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
        if epoch % eval_interval == 0:
            print(f"{epoch} - Comparing against greedy...")
            greedy_paths = run_greedy_tsp_solver(graphs, start_nodes)
            greedy_costs = torch.sum(greedy_paths.costs, dim=-1)
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
    greedy_paths = run_greedy_tsp_solver(graphs, start_nodes)
    gnn_costs = torch.sum(mcts_paths.costs, dim=-1)
    greedy_costs = torch.sum(greedy_paths.costs, dim=-1)
    print("ratio:", torch.mean((gnn_costs / greedy_costs)))
    plot_solution(graphs[0], mcts_paths.nodes[0])
    plot_solution(graphs[0], greedy_paths.nodes[0])


def train_greedy():
    # -- Config
    batch_size = 1024
    num_nodes = 100
    device = "cpu"
    start_node = 0
    num_simulations = 50
    num_epochs = 1000
    eval_interval = 25

    # -- Tensorboard
    writer = SummaryWriter()

    # -- Create GNN
    # model = GAT(
    #     in_channels=4,
    #     hidden_channels=64,
    #     out_channels=64,
    #     heads=2,
    #     edge_dim=2,
    #     edge_out_dim=16,
    # )
    model = GATv2(
        in_channels=4,
        hidden_channels=64,
        out_channels=64,
        heads=2,
        edge_dim=2,
    )
    model = torch.jit.script(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
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
        greedy_paths = run_greedy_tsp_solver(graphs, start_nodes)

        # -- Add to Replay Buffer
        print(f"{epoch} - Adding to Replay Buffer...")
        rb.add_to_buffer(replay_buffer, graphs, greedy_paths)

        # -- Train Model w/ Batch Samples
        print(f"{epoch} - Training Model w/ Batch Samples...")
        for i, batch in tqdm(enumerate(replay_buffer), total=num_nodes - 1):
            loss = train(model, optimizer, batch)

            # -- Update Tensorboard
            writer.add_scalar("Loss/train", loss, (epoch * (num_nodes - 1)) + i)
            writer.flush()

        # -- Evaluate Performance
        if epoch % eval_interval == 0:
            print(f"{epoch} - Comparing against greedy...")
            # -- Run with MCTS
            with torch.no_grad():
                gnn_paths = run_greedy_gnn_solver(graphs, model, start_nodes)
            gnn_costs = torch.sum(gnn_paths.costs, dim=-1)
            greedy_costs = torch.sum(greedy_paths.costs, dim=-1)
            writer.add_scalar(
                "Eval/ratio", torch.mean((gnn_costs / greedy_costs)), epoch
            )
            writer.flush()

        print(f"{epoch} - Epoch Complete!")

    # -- Test out the model
    print("Testing the model...")
    graphs = generate_random_graph_batch(batch_size, num_nodes, device)
    start_nodes = torch.full(size=(batch_size,), fill_value=start_node)
    with torch.no_grad():
        gnn_paths = run_greedy_gnn_solver(graphs, model, start_nodes)
    greedy_paths = run_greedy_tsp_solver(graphs, start_nodes)
    # print(torch.sum(gnn_paths.costs, dim=-1))
    gnn_costs = torch.sum(gnn_paths.costs, dim=-1)
    greedy_costs = torch.sum(greedy_paths.costs, dim=-1)
    print("ratio:", torch.mean((gnn_costs / greedy_costs)))
    plot_solution(graphs[0], gnn_paths.nodes[0])
    plot_solution(graphs[0], greedy_paths.nodes[0])


if __name__ == "__main__":
    start = time.time()
    train_model()
    # train_greedy()
    print(f"Time elapsed: {time.time() - start}")
