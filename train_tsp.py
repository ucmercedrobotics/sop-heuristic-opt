from tqdm import tqdm
import time
from datetime import datetime
import os

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW

from sop.utils.graph_torch import generate_random_graph_batch
from sop.utils.visualization import plot_solutions
import sop.utils.replay_buffer as rb
from sop.gnn.gat import GAT
from sop.utils.tsp_greedy import run_greedy_tsp_solver
from sop.mcts.tsp import tsp_mcts_gnn_solver, tsp_greedy_gnn_solver
from sop.mcts.tsp import create_node_features, create_edge_features, create_adj_matrix

# -- GNN Utils


def create_model():
    model = GAT(
        in_channels=4,
        hidden_channels=32,
        out_channels=32,
        heads=2,
        edge_dim=3,
        edge_out_dim=16,
    )

    # model = GATv2(
    #     in_channels=4,
    #     hidden_channels=32,
    #     out_channels=32,
    #     heads=2,
    #     edge_dim=2,
    # )

    optimizer = AdamW(model.parameters(), lr=1e-4)
    return model, optimizer


def save_model(
    model, optimizer, epoch, train_step, inference_step, checkpoint_dir="checkpoints"
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(checkpoint_dir, f"{current_time}_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "train_step": train_step,
            "inference_step": inference_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_model(model, optimizer, path):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        model,
        optimizer,
        checkpoint["epoch"] + 1,
        checkpoint["train_step"] + 1,
        checkpoint["inference_step"] + 1,
    )


# -- Train Step


def train(model: nn.Module, optimizer: Optimizer, batch: rb.TrainData):
    indices = torch.arange(batch.batch_size[0])

    model.train()
    optimizer.zero_grad()

    node_features = create_node_features(
        batch.graph, batch.current_node, batch.goal_node
    )
    edge_features = create_edge_features(
        batch.graph, batch.current_node, batch.goal_node
    )
    adj = create_adj_matrix(batch.mask)
    out = model(node_features, adj, edge_features, batch.mask)

    pred_Q = out[indices, batch.action]
    loss = F.mse_loss(pred_Q, batch.q_value)
    loss.backward()
    optimizer.step()

    return loss


# -- Main


def main():
    # -- Config
    device = "cpu"
    # Graph
    batch_size = 128
    num_nodes = 20
    start_node = 2
    # MCTS
    num_simulations = 50
    discount = 0.99
    z = 1.0

    # -- Tensorboard
    writer = SummaryWriter()

    # -- Initialization

    model, optimizer = create_model()

    replay_buffer = rb.create_replay_buffer(
        max_size=batch_size * (num_nodes - 1), batch_size=batch_size
    )

    # Train
    num_epochs = 10
    num_steps = 15

    train_step = 0
    inference_step = 0
    epoch_step = 0
    resume_checkpoint = None

    if resume_checkpoint is not None:
        model, optimizer, epoch_step, train_step, inference_step = load_model(
            model, optimizer, resume_checkpoint
        )

    graphs = generate_random_graph_batch(batch_size, num_nodes, device)
    start_nodes = torch.full(size=(batch_size,), fill_value=start_node)

    for epoch in range(num_epochs):
        for step in range(num_steps):
            print(f"{epoch}:{step} - Generating Graphs...")

            print(f"{epoch}:{step} - Solving TSP w/ MCTS...")
            mcts_paths, tree_stats = tsp_mcts_gnn_solver(
                model, graphs, start_nodes, num_simulations, discount, z, device
            )

            mcts_score = mcts_paths.rewards.sum(dim=-1).mean(dim=-1)
            writer.add_scalar("Train/score", mcts_score, inference_step)

            rb.add_to_buffer(replay_buffer, graphs, mcts_paths, discount)

            print(f"{epoch}:{step} - Training Model w/ Batch Samples...")
            for batch in tqdm(replay_buffer, total=num_nodes - 1):
                loss = train(model, optimizer, batch)

                writer.add_scalar("Train/loss", loss, train_step)
                train_step += 1

            inference_step += 1
            writer.flush()
            replay_buffer.empty()

        # Evaluation
        print(f"{epoch} - Starting Evaluation...")
        graphs = generate_random_graph_batch(batch_size, num_nodes, device)
        start_nodes = torch.full(size=(batch_size,), fill_value=start_node)

        print(f"{epoch} - Solving TSP w/ MCTS...")
        mcts_paths, tree_stats = tsp_mcts_gnn_solver(
            model, graphs, start_nodes, num_simulations, discount, z, device
        )
        print(f"{epoch} - Solving TSP w/ GNN...")
        gnn_paths = tsp_greedy_gnn_solver(model, graphs, start_nodes, device)
        print(f"{epoch} - Solving TSP w/ Greedy...")
        greedy_paths = run_greedy_tsp_solver(graphs, start_nodes)

        mcts_score = mcts_paths.rewards.sum(dim=-1).mean(dim=-1)
        gnn_score = gnn_paths.rewards.sum(dim=-1).mean(dim=-1)
        greedy_score = greedy_paths.rewards.sum(dim=-1).mean(dim=-1)

        writer.add_scalar("Eval/mcts", mcts_score, epoch_step)
        writer.add_scalar("Eval/gnn", gnn_score, epoch_step)
        writer.add_scalar("Eval/greedy", greedy_score, epoch_step)
        writer.add_scalar("Eval/mcts-greedy", mcts_score / greedy_score, epoch_step)
        writer.flush()

        # -- Save model
        save_model(model, optimizer, epoch_step, train_step, inference_step)

        epoch_step += 1

    # -- Testing

    batch_size = 2

    graphs = generate_random_graph_batch(batch_size, num_nodes, device)
    start_nodes = torch.full(size=(batch_size,), fill_value=start_node)

    gnn_paths = tsp_greedy_gnn_solver(model, graphs, start_nodes, device)
    mcts_paths, tree_stats = tsp_mcts_gnn_solver(
        model, graphs, start_nodes, num_simulations, discount, z, device
    )
    greedy_paths = run_greedy_tsp_solver(graphs, start_nodes)

    print("gnn", gnn_paths.rewards.sum(dim=-1).mean(dim=-1))
    print("mcts", mcts_paths.rewards.sum(dim=-1).mean(dim=-1))
    print("greedy", greedy_paths.rewards.sum(dim=-1).mean(dim=-1))

    plot_solutions(
        graphs[0],
        paths=[
            gnn_paths.nodes[0],
            mcts_paths.nodes[0],
            greedy_paths.nodes[0],
        ],
        titles=[
            f"GNN Cost: {torch.sum(gnn_paths.rewards, dim=-1)[0]:.5f}",
            f"MCTS Cost: {torch.sum(mcts_paths.rewards, dim=-1)[0]:.5f}",
            f"Greedy Cost: {torch.sum(greedy_paths.rewards, dim=-1)[0]:.5f}",
        ],
        rows=1,
        cols=3,
    )


if __name__ == "__main__":
    main()
