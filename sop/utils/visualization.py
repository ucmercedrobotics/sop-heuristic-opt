# Types
from typing import Optional, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

import math
import matplotlib.pyplot as plt
import numpy as np
import torch

from sop.utils.graph import TorchGraph
from sop.utils.path import Path


# -- Utilities
def subplots(rows: int, cols: int) -> Tuple[Figure, Axes]:
    fig, axs = plt.subplots(rows, cols)
    if type(axs) is not list:
        axs = [axs]
    for ax in axs:
        ax.set_axis_off()
    return fig, axs


def show(out_path: Optional[str]):
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()


# -- Multi Plots
def plot_solutions(
    graph: TorchGraph,
    paths: list[Path],
    titles: list[str],
    out_path: Optional[str] = None,
    cols: int = 2,
):
    assert len(paths) == len(titles)
    rows = math.ceil(len(titles) / cols)
    fig, axs = subplots(rows, cols)

    for i, (path, title) in enumerate(zip(paths, titles)):
        ax = axs[i]
        ax.set_axis_on()
        plot_graph(graph, ax)
        plot_path(graph, path, ax)
        ax.set_title(title)

    show(out_path)


def plot_heuristics(
    heuristics: list[Tensor],
    titles: list[str],
    out_path: Optional[str] = None,
    cols: int = 2,
):
    assert len(heuristics) == len(titles)
    rows = math.ceil(len(titles) / cols)
    fig, axs = subplots(rows, cols)

    for i, (heuristic, title) in enumerate(zip(heuristics, titles)):
        ax = axs[i]
        ax.set_axis_on()
        plot_heatmap(heuristic, ax, cmap="YlGn", cbarlabel="score")
        ax.set_title(title)

    show(out_path)


def plot_statistics(
    statistics: list[dict[str, list[float]]],
    titles: list[str],
    out_path: Optional[str] = None,
    cols: int = 2,
    data_labels: list[str] = ["Min", "Avg", "Max", "p_f"],
):
    assert len(statistics) == len(titles)
    rows = math.ceil(len(titles) / cols)
    fig, axs = subplots(rows, cols)

    # data = {
    #     "Walk": (4.21, 9.10, 10.50, 0.11),
    #     "Vanilla": (5.13, 10.13, 12.25, 0.1),
    #     "ACO": (5.32, 10.82, 13.19, 0.0),
    # }

    for i, (stat, title) in enumerate(zip(statistics, titles)):
        ax = axs[i]
        ax.set_axis_on()
        plot_bars(data_labels, stat, ax)
        ax.set_ylabel("Reward")
        ax.set_title(title)

    show(out_path)


# -- Plot Items


def plot_graph(
    graph: TorchGraph,
    ax: Axes,
):
    start_node = graph.extra["start_node"]
    goal_node = graph.extra["goal_node"]
    pos = graph.nodes["position"]
    reward = graph.nodes["reward"]

    # Define colors
    colors = ["b"] * pos.shape[0]
    colors[start_node] = "r"
    colors[goal_node] = "g"
    # Define alpha based on reward
    alpha = torch.clamp((1.0 * reward), min=0, max=1).tolist()
    alpha[start_node] = 0.5
    alpha[goal_node] = 0.5
    # Plot
    ax.scatter(x=pos[:, 0], y=pos[:, 1], c=colors, alpha=alpha)
    # Label each point w/ index
    for i in range(len(pos)):
        ax.annotate(i, pos[i], textcoords="offset points", xytext=(5, 5), ha="right")

    ax.grid(True)


def plot_path(graph: TorchGraph, path: Path, ax: Axes):
    pos = graph.nodes["position"]

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


def plot_heatmap(
    heatmap: Tensor, ax: Axes, cbar_kw=None, cbarlabel="", xstep=5, ystep=1, **kwargs
):
    """Display a heatmap from a 2d matrix."""
    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(heatmap, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    # Show all ticks and label them with the respective list entries
    x_labels = [i for i in range(0, heatmap.shape[1], xstep)]
    y_labels = [i for i in range(0, heatmap.shape[0], ystep)]

    ax.set_xticks(range(0, heatmap.shape[1], 5), labels=x_labels)
    ax.set_yticks(range(heatmap.shape[0]), labels=y_labels)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)

    return im


def plot_bars(
    data_labels: list[str], data: dict[str, list[float]], ax: Axes, width: float = 0.25
):
    x = np.arange(len(data_labels))  # the label locations
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    for attribute, measurement in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_xticks(x + width, data_labels)
    ax.legend(loc="upper left", ncols=3)
