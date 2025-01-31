from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
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
    out_path: Optional[str] = None,
    rows: int = 1,
    cols: int = 2,
):
    assert len(paths) == len(titles)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    if len(paths) == 1:
        axes = [axes]

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
        # Label each point
        for i in range(len(pos)):
            ax.annotate(
                i, pos[i], textcoords="offset points", xytext=(5, 5), ha="right"
            )

        # -- Label

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
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()


def heatmap(data, ax, cbar_kw=None, cbarlabel="", xstep=5, ystep=1, **kwargs):
    """
    Create a heatmap from a 2x2 matrix.
    Taken from: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    # Show all ticks and label them with the respective list entries
    x_labels = [i for i in range(0, data.shape[1], xstep)]
    y_labels = [i for i in range(0, data.shape[0], ystep)]

    ax.set_xticks(range(0, data.shape[1], 5), labels=x_labels)
    ax.set_yticks(range(data.shape[0]), labels=y_labels)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)

    return im


def plot_heuristics(
    heuristics: list[torch.Tensor],
    titles: list[str],
    out_path: Optional[str] = None,
    rows: int = 1,
    cols: int = 2,
):
    assert len(heuristics) == len(titles)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    if len(heuristics) == 1:
        axes = [axes]

    for i, (H, title) in enumerate(zip(heuristics, titles)):
        ax = axes[i]
        im = heatmap(H, ax=ax, cmap="YlGn", cbarlabel="score")
        ax.set_title(title)

    # Hide unused subplots
    for i in range(len(heuristics), len(axes)):
        fig.delaxes(axes[i])  # Removes empty subplot spaces

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    else:
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
