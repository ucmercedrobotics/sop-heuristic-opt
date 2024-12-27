import matplotlib.pyplot as plt
import graphviz
import torch

from sop.utils.graph_torch import TorchGraph
from sop.mcts.core import Tree


# -- Visualization


def plot_solutions(
    graph: TorchGraph,
    paths: list[torch.Tensor],
    titles: list[str],
    rows: int = 1,
    cols: int = 2,
):
    assert len(paths) == len(titles)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    for i, (path, title) in enumerate(zip(paths, titles)):
        ax = axes[i]
        starting_node = path[0]
        pos = graph.nodes["position"]

        # -- Draw nodes
        colors = ["b"] * pos.shape[0]
        colors[starting_node] = "r"

        ax.scatter(x=pos[:, 0], y=pos[:, 1], c=colors, alpha=0.5)

        # -- Draw edges
        points = pos[path]
        for i in range(points.shape[0] - 1):
            ax.annotate(
                "", xy=points[i + 1], xytext=points[i], arrowprops=dict(arrowstyle="->")
            )

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
