from typing import Optional
from dataclasses import dataclass
import time
import os
from datetime import datetime

import hydra
from hydra.core.config_store import ConfigStore
import torch
import rootutils
import optuna

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sop.utils.graph_torch import TorchGraph, generate_sop_graphs
from sop.utils.visualization import plot_solutions, plot_heuristics
from sop.utils.path import evaluate_path, path_to_heatmap
from sop.utils.seed import random_seed, set_seed

from sop.milp.pulp_milp_sop import sop_milp_solver
from sop.mcts.aco import (
    ACOParams,
    sop_aco_solver,
    mcts_sopcc_heuristic,
    random_heuristic,
    small_heuristic,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -- Config
@dataclass
class Config:
    # Data
    dataset_dir: str = "data"
    visual_dir: str = "viz"
    seed: Optional[int] = None
    # Batch
    batch_size: int = 8
    device: str = DEVICE
    # Graph
    num_nodes: int = 20
    budget: int = 2
    start_node: int = 1
    goal_node: int = 19
    # Sampling
    num_samples: int = 100
    kappa: float = 0.5
    # ACO
    num_rollouts: int = 100
    num_iterations: int = 10
    p_f: float = 0.1
    # MILP
    milp_time_limit: Optional[int] = None


cs = ConfigStore.instance()
cs.store(name="aco_sop", node=Config)


def run_milp(cfg: Config, graph: TorchGraph, viz_prefix: Optional[None]):
    milp_path = sop_milp_solver(
        graph, time_limit=cfg.milp_time_limit, num_samples=cfg.num_samples
    )
    failure_prob, avg_cost = evaluate_path(
        milp_path, graph.unsqueeze(0), cfg.num_samples, cfg.kappa
    )
    milp_info = (
        "MILP; "
        + f"R: {milp_path.reward.sum(-1)[0]:.5f}, "
        + f"B: {cfg.budget}, "
        + f"C: {float(avg_cost):.5f}, "
        + f"F: {float(failure_prob):.3f} "
        + f"N: {int(milp_path.length[0])}"
    )
    print(milp_info)
    out_path = viz_prefix + "_milp" if viz_prefix is not None else None
    plot_solutions(
        graph,
        paths=[milp_path[0]],
        titles=[milp_info],
        out_path=out_path,
        rows=1,
        cols=1,
    )

    out_path = viz_prefix + "_milp_heatmap" if viz_prefix is not None else None
    plot_heuristics(
        heuristics=[path_to_heatmap(milp_path)[0]],
        titles=["Milp_H"],
        out_path=out_path,
        rows=1,
        cols=1,
    )


@hydra.main(version_base=None, config_name="aco_sop")
def main(cfg: Config) -> None:
    torch.set_default_device(cfg.device)

    # -- Set seed
    if cfg.seed is None:
        cfg.seed = random_seed()
    # set_seed(cfg.seed)

    # -- Generate Data
    # TODO: if file exists, import
    # TODO: if we begin to have multiple experiments of the same time, we can add more flags to the path
    expected_graph_tensor_path = (
        cfg.dataset_dir
        + "/"
        + str(cfg.seed)
        + "_graphs_"
        + str(cfg.batch_size)
        + "_"
        + str(cfg.num_nodes)
    )
    if os.path.isfile(expected_graph_tensor_path):
        print(f"Loading graphs from {expected_graph_tensor_path}...")
        graphs = TorchGraph.load(expected_graph_tensor_path)
    else:
        print(f"Generating {cfg.batch_size} graphs...")
        graphs = generate_sop_graphs(
            cfg.batch_size,
            cfg.num_nodes,
            cfg.start_node,
            cfg.goal_node,
            cfg.budget,
            cfg.num_samples,
            cfg.kappa,
        )

    viz_path = cfg.dataset_dir + "/" + cfg.visual_dir
    print(f"Creating vizualization folder {viz_path}...")
    os.makedirs(viz_path, exist_ok=True)
    viz_prefix = f"{viz_path}/{cfg.seed}_{cfg.batch_size}_{cfg.num_nodes}"

    # -- Get first graph for testing
    first_graph = graphs[0].cpu()

    # -- MILP
    # run_milp(cfg, first_graph, viz_prefix)

    # -- Generate Heuristic
    print("Generating Heuristic...")
    random_H = random_heuristic(cfg.batch_size, cfg.num_nodes)
    small_H = small_heuristic(cfg.batch_size, cfg.num_nodes)
    sopcc_H = mcts_sopcc_heuristic(graphs.nodes["reward"], graphs.edges["samples"])
    heuristic = random_H

    # -- ACO Tuning
    def objective(trial):
        params = ACOParams(
            alpha=trial.suggest_float("alpha", 0.0, 10.0),
            beta=trial.suggest_float("beta", 0.0, 10.0),
            rho=trial.suggest_float("rho", 0.0, 1.0),
            topk=trial.suggest_int("topk", 1, cfg.num_rollouts),
            a=trial.suggest_int("a", 1, 100),
            fail_penalty=trial.suggest_float("fail_penalty", 0.0, 1.0),
            # lr=trial.suggest_float("lr", 0.0, 2.0),
            # alpha=1.3,
            # beta=0.2,
            # rho=0.8,
            # topk=int(cfg.num_rollouts / 3),
            # fail_penalty=1.0,
            lr=1.0,
        )
        Q = sop_aco_solver(
            params,
            graphs,
            heuristic,
            num_rollouts=cfg.num_rollouts,
            p_f=cfg.p_f,
            kappa=cfg.kappa,
        )

        return Q.mean(-1)

    # study = optuna.create_study(
    #     direction="maximize",
    #     storage="sqlite:///db.sqlite3",
    #     study_name="ACO",
    #     load_if_exists=True,
    # )
    # study.optimize(objective, n_trials=100)

    # Test avg performance on a single graph
    n = 10
    graph = graphs[0]
    broadcasted_graph = graph.unsqueeze(0).expand(n)
    heuristic = heuristic[0].unsqueeze(0).expand(n, cfg.num_nodes, cfg.num_nodes)

    # -- Solve SOP
    print("ACO SOP...")
    start = time.time()
    params = ACOParams()
    path, is_success = sop_aco_solver(
        params,
        broadcasted_graph,
        heuristic,
        num_rollouts=cfg.num_rollouts,
        num_iterations=cfg.num_iterations,
        p_f=cfg.p_f,
        kappa=cfg.kappa,
    )
    print(f"Results averaged over {n} runs:")
    print(
        "Params:\n"
        + f"- num_nodes: {cfg.num_nodes}\n"
        + f"- p_f: {cfg.p_f}\n"
        + f"- num_iterations: {cfg.num_iterations}\n"
        + f"- num_rollouts: {cfg.num_rollouts}\n"
        + f"- num_samples: {cfg.num_samples}"
    )
    print(f"Time elapsed: {time.time() - start}")
    print(f"Avg. Reward: {path.reward.sum(-1).sum(-1) / n}")
    print(f"Failure Prob: {1 - (is_success.sum(-1) / n)}")

    # -- Evaluation
    # print("Visualize...")
    failure_prob, avg_cost = evaluate_path(
        path[0].unsqueeze(0), first_graph.unsqueeze(0), cfg.num_samples, cfg.kappa
    )
    aco_info = (
        "ACO; "
        + f"R: {path[0].reward.sum(-1):.5f}, "
        + f"B: {cfg.budget}, "
        + f"C: {float(avg_cost):.5f}, "
        + f"F: {float(failure_prob):.3f} "
        + f"N: {int(path[0].length)}"
    )
    print(aco_info)
    plot_solutions(
        first_graph,
        paths=[path[0]],
        titles=[aco_info],
        out_path=viz_prefix + "_aco",
        rows=1,
        cols=1,
    )


if __name__ == "__main__":
    main()
