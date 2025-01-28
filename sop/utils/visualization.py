import matplotlib.pyplot as plt
import graphviz
import torch

from sop.utils.graph_torch import TorchGraph
from sop.utils.path import Path
from sop.mcts.core import Tree


# -- Visualization


def plot_solutions(
    graph: TorchGraph,
    paths: list[Path],
    titles: list[str],
    rows: int = 1,
    cols: int = 2,
):
    assert len(paths) == len(titles)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    for i, (path, title) in enumerate(zip(paths, titles)):
        ax = axes[i]
        starting_node = path.nodes[0]
        goal_node = path.nodes[path.length - 1]
        pos = graph.nodes["position"]

        # -- Draw nodes
        colors = ["b"] * pos.shape[0]
        colors[starting_node] = "r"
        colors[goal_node] = "g"

        alpha = (1.0 * graph.nodes["reward"]).tolist()
        alpha[starting_node] = 0.5
        alpha[goal_node] = 0.5

        ax.scatter(x=pos[:, 0], y=pos[:, 1], c=colors, alpha=alpha)

        # -- Draw edges
        path_index = 1
        while path_index < path.length:
            prev_node = path.nodes[path_index - 1]
            current_node = path.nodes[path_index]
            ax.annotate(
                "",
                xy=pos[current_node],
                xytext=pos[prev_node],
                arrowprops=dict(arrowstyle="->"),
            )
            path_index += 1

        ax.set_title(title)
        ax.grid(True)

    # Hide unused subplots
    for i in range(len(paths), len(axes)):
        fig.delaxes(axes[i])  # Removes empty subplot spaces

    plt.tight_layout()
    plt.show()


def plot_tree(tree: Tree):
    dot = graphviz.Digraph("Tree", comment="MCTS Tree")

    def add_node(dot: graphviz.Digraph, tree: Tree, node: int):
        node_mapping = tree.node_mapping[node]
        visit_count = tree.visit_count[node]
        V = tree.node_value[node]
        s = f"{node_mapping}\nv: {visit_count}\nV: {V:.4f}"
        dot.node(f"{node}", s)

        children = tree.children_index[node]
        for child in children:
            if child == -1:
                continue
            add_node(dot, tree, child)
            child_mapping = tree.node_mapping[child]
            Q = tree.children_Q_values[node, child_mapping]
            dot.edge(f"{node}", f"{child}", label=f"{Q:.4f}")

    add_node(dot, tree, 0)

    return dot
