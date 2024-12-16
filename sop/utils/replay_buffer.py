from sop.utils.path2 import Path
from sop.utils.graph_torch import TorchGraph


def compute_value_targets(graph: TorchGraph, path: Path):
    batch_size, path_length = path.size()
    print(path)
    print(path.nodes)
    print(path.costs)
