import time
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from sop.utils.graph_pyg import generate_random_graph_batch, get_bytes
from sop.utils.perf import profile
from sop.mcts.mcts import run


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # TODO: Add configuration to hydra config.yaml
    BATCH_SIZE = 2
    NUM_SIMULATIONS = 50
    NUM_NODES = 3
    BUDGET = 2.0
    START_NODE = 1
    END_NODE = 1

    graph = generate_random_graph_batch(num_nodes=NUM_NODES, batch_size=BATCH_SIZE)

    start_node = torch.full(size=(BATCH_SIZE,), fill_value=START_NODE)
    end_node = torch.full(size=(BATCH_SIZE,), fill_value=END_NODE)
    budget = torch.full(size=(BATCH_SIZE,), fill_value=BUDGET)

    result = run(graph, start_node, end_node, budget, NUM_SIMULATIONS)
    # result = profile(100, run, graph, start_node, end_node, budget, NUM_SIMULATIONS)

    print(result)


if __name__ == "__main__":
    main()
