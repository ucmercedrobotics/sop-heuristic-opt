from typing import ClassVar
from dataclasses import dataclass
import torch

# Based heavily on: https://github.com/google-deepmind/mctx

Array = torch.Tensor


@dataclass
class Tree:
    """State of the search tree."""

    node_mapping: Array # [B, N], index of node in original graph
    raw_values: Array  # [B, N], raw computed value for each node
    node_values: Array  # [B, N], cumulative search value for each node
    failure_probs: Array  # [B, N], raw failure probability for each node
    node_visits: Array  # [B, N], visit counts for each node
    parents: Array  # [B, N], node index for the parents of each node
    neighbor_from_parent: Array # [B, N], the neighbor index to take from the parent to each node
    children_index: Array  # [B, N, num_neighbors], the node index of the neighbor if visited
    children_values: Array  # [B, N, num_neighbors], the value of traveling to neighbor from node
    children_failure_probs: Array  # [B, N, num_nodes], the failure prob of traveling to neighbor from node
    children_visits: Array  # [B, N, num_nodes], the visit counts for each neighbor

    # Static class variables
    ROOT_INDEX: ClassVar[int] = 0
    NO_PARENT: ClassVar[int] = -1
    UNVISITED: ClassVar[int] = -1
    
    def infer_batch_size(self):
        return self.node_mapping.shape[0]
