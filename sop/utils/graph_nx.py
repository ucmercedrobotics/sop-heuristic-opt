import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# -- Main
def generate_random_graph(num_nodes: int) -> nx.Graph:
    """Generates a random complete graph.

    Node attributes:
    "position": tuple(int, int) -> 2D position between 0 and 1
    "reward": int -> random reward between 0 and 1

    Edge attributes:
    "weight": int -> distance between u and v
    """
    # -- Complete Graph
    G = nx.complete_graph(n=num_nodes)

    # -- Generate node attributes
    positions = generate_uniform_positions(num_nodes)
    dist_matrix = compute_distance_matrix(positions)
    rewards = generate_uniform_reward(num_nodes)

    # -- Add node attributes to graph
    p_dict = {i: (x, y) for i, (x, y) in enumerate(positions)}
    r_dict = {i: r for i, r in enumerate(rewards)}
    nx.set_node_attributes(G, values=p_dict, name="position")
    nx.set_node_attributes(G, values=r_dict, name="reward")

    # -- Add edge weights
    w_dict = {(u, v): dist_matrix[u, v] for u, v in G.edges}
    nx.set_edge_attributes(G, values=w_dict, name="weight")

    return G


# -- Position
def generate_uniform_positions(N: int) -> np.ndarray:
    """Creates 2D positions for each node in the graph."""
    xs = np.random.uniform(low=0, high=1, size=N)  # (N,)
    ys = np.random.uniform(low=0, high=1, size=N)  # (N,)
    positions = np.stack([xs, ys])  # (2, N)
    return positions.T  # (N, 2)


def compute_distance_matrix(pos: np.ndarray) -> np.ndarray:
    """Computes distances for every node to every other node.
    Essentially batch computes L2 norm.
    """
    # (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    # (N, N, 2) -> (N, N)
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return dist_matrix


# -- Reward
def generate_uniform_reward(N: int) -> np.ndarray:
    """Computes random reward for each node in the graph."""
    return np.random.uniform(low=0, high=1, size=N)


# -- Visualization
def visualize_graph(G: nx.Graph, pos_key="position"):
    pos = nx.get_node_attributes(G, pos_key)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, node_size=80)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    import time

    N = 100  # num_nodes

    _ = generate_random_graph(N)  # warmup

    start = time.time()
    G = generate_random_graph(N)
    print(time.time() - start)

    # visualize_graph(G)
    print(G)
