import torch
import time

import torch_geometric.nn as gnn

from sop.utils.graph_torch import generate_random_graph_batch
from sop.utils.path2 import Path
from sop.mcts.mcts_tsp import preprocess_features, preprocess_mask
from sop.gnn.gat import DenseGAT


def main():
    # -- Config
    batch_size = 256
    num_nodes = 100
    device = "cpu"
    start_node = 2

    # -- Generate a batch of graphs
    graphs = generate_random_graph_batch(batch_size, num_nodes, device)

    # -- Create initial paths
    indices = torch.arange(batch_size)
    current_graph_node = torch.full((batch_size,), start_node)
    tree_path = Path.empty(batch_size, num_nodes, device)
    tree_path.append(
        indices,
        current_graph_node,
        cost=torch.zeros(batch_size, device=device),
    )
    dense_gat = DenseGAT(in_channels=4, hidden_channels=128, out_channels=128, heads=2)

    mask = preprocess_mask(tree_path, current_graph_node, current_graph_node)
    node_features = preprocess_features(graphs, current_graph_node, current_graph_node)
    edge_matrix = graphs.edge_matrix
    print(gnn.summary(dense_gat, node_features, edge_matrix))

    start = time.time()
    x = dense_gat(node_features, edge_matrix, mask)
    print(f"Time elapsed: {time.time() - start}")
    print(x.shape)


if __name__ == "__main__":
    main()
