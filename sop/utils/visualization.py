import matplotlib.pyplot as plt
import torch

from sop.utils.graph_torch import TorchGraph
from sop.utils.path2 import Path


# -- Visualization
def plot_solution(graph: TorchGraph, path: torch.Tensor):
    fig, ax = plt.subplots()

    starting_node = path[0]
    pos = graph.nodes["position"]

    # -- Draw nodes
    colors = ["b"] * pos.shape[0]
    colors[starting_node] = "r"

    ax.scatter(x=pos[:, 0], y=pos[:, 1], c=colors, alpha=0.5)

    # -- Draw edges
    points = pos[path]
    for i in range(points.shape[0] - 1):
        ax.plot(
            points[i : i + 2, 0],
            points[i : i + 2, 1],
            linewidth="1",
            linestyle="solid",
        )

    ax.grid(True)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    import matplotlib.cbook as cbook

    # Load a numpy record array from yahoo csv data with fields date, open, high,
    # low, close, volume, adj_close from the mpl-data/sample_data directory. The
    # record array stores the date as an np.datetime64 with a day unit ('D') in
    # the date column.
    price_data = cbook.get_sample_data("goog.npz")["price_data"]
    price_data = price_data[-250:]  # get the most recent 250 trading days

    delta1 = np.diff(price_data["adj_close"]) / price_data["adj_close"][:-1]

    # Marker size in units of points^2
    volume = (15 * price_data["volume"][:-2] / price_data["volume"][0]) ** 2
    close = 0.003 * price_data["close"][:-2] / 0.003 * price_data["open"][:-2]

    fig, ax = plt.subplots()
    ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)

    ax.set_xlabel(r"$\Delta_i$", fontsize=15)
    ax.set_ylabel(r"$\Delta_{i+1}$", fontsize=15)
    ax.set_title("Volume and percent change")

    ax.grid(True)
    fig.tight_layout()

    plt.show()
