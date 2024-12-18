import torch
import time

import torch_geometric.nn as gnn

from sop.utils.graph_torch import generate_random_graph_batch
from sop.utils.path2 import Path
from sop.mcts.mcts_tsp import preprocess_mask
from sop.gnn.gat import GAT, GATv2, mask_adj, preprocess_features, preprocess_edges


def main():
    # -- Config
    batch_size = 256
    num_nodes = 100
    device = "cpu"
    start_node = 0

    # -- Generate a batch of graphs
    graphs = generate_random_graph_batch(batch_size, num_nodes, device)

    # -- Create initial paths
    indices = torch.arange(batch_size)
    current_graph_node = torch.full((batch_size,), start_node)
    tree_path = Path.empty(batch_size, num_nodes, device)
    for i in range(10):
        tree_path.append(
            indices,
            current_graph_node + i,
            cost=torch.zeros(batch_size, device=device),
        )

    # -- Create model
    model = GATv2(
        in_channels=4,
        hidden_channels=64,
        out_channels=64,
        heads=2,
        edge_dim=2,
    )
    model = torch.jit.script(model)

    # -- Create dummy features
    mask = preprocess_mask(tree_path, current_graph_node, current_graph_node)
    node_features = preprocess_features(
        graphs, current_graph_node, current_graph_node, mask
    )
    adj = mask_adj(mask)
    edge_attr = preprocess_edges(graphs, current_graph_node, current_graph_node)
    print(edge_attr.shape)
    print(gnn.summary(model, node_features, adj, edge_attr))

    start = time.time()
    x = model(node_features, adj, edge_attr, mask)
    print(f"Time elapsed: {time.time() - start}")
    print(x.shape)


if __name__ == "__main__":
    main()
