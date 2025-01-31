from typing import Tuple, Callable, Optional
from dataclasses import dataclass
import time

import torch
from torch import Tensor

from tensordict import TensorDict, tensorclass

from sop.utils.path import Path
from sop.utils.graph_torch import TorchGraph
from sop.utils.sample import sample_costs

# Debugging
from sop.utils.visualization import plot_solutions, plot_heuristics
from sop.utils.path import evaluate_path, path_to_heatmap

import optuna


# -- Heuristics
def random_heuristic(batch_size: int, num_nodes: int):
    return torch.rand((batch_size, num_nodes, num_nodes)).softmax(-1)


def mcts_sopcc_heuristic(rewards: Tensor, sampled_costs: Tensor):
    average_cost = sampled_costs.mean(dim=-1)
    return rewards.unsqueeze(-1) / average_cost


# ACO
@dataclass
class ACOParams:
    # Hyperparams
    # alpha: Pheremone weight
    # beta: Heuristic weight
    # rho: Evaporation Rate
    # lr: ada_ir random weight
    # fail_penalty: scoring_fn failure weight
    # a: Constant for tau_min

    alpha: float = 1.0
    beta: float = 1.0
    rho: float = 0.1
    lr: float = 1.0
    fail_penalty: float = 0.9
    a: float = 10


@torch.no_grad
def sop_aco_solver(
    params: ACOParams,
    graph: TorchGraph,
    heuristic: Tensor,
    num_rollouts: int,
    p_f: float,
    kappa: float,
) -> Tuple[Path, Tensor]:
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    start_node = graph.extra["start_node"]
    budget = graph.extra["budget"]
    goal_node = graph.extra["goal_node"]
    rewards = graph.nodes["reward"]
    weights = graph.edges["distance"]

    # Init ACS state
    state = MMACSState.init(heuristic)

    # Loop state
    current_node = start_node.clone()
    current_budget = budget.clone()

    # Initialize Path
    path = Path.empty(batch_size, num_nodes)
    path.append(indices, current_node)
    path.mask[indices, goal_node] = 1

    while indices.numel() > 0:
        next_node = aco_search(
            params=params,
            graph=graph[indices],
            state=state[indices],
            current_node=current_node[indices],
            current_budget=current_budget[indices],
            current_path=path[indices],
            num_rollouts=num_rollouts,
            p_f=p_f,
            kappa=kappa,
        )
        return next_node

        # Get values
        r = rewards[indices, next_node]
        w = weights[indices, current_node[indices], next_node]

        # Update budget
        sampled_cost = sample_costs(w, num_samples=1, kappa=kappa)
        current_budget[indices] -= sampled_cost

        # Add to path
        path.append(indices, next_node, r)

        # Update state
        is_not_goal = next_node != goal_node[indices]
        has_budget = current_budget[indices] > 0
        is_continuing = torch.logical_and(is_not_goal, has_budget)
        indices = indices[is_continuing]
        current_node[indices] = next_node[is_continuing]

    # Check if run was success
    is_success = current_budget >= 0

    return path, is_success


# -- ACO Search
@tensorclass
class MMACSState:
    heuristic: Tensor
    pheremone: Tensor

    # MMACS
    tau_min: Tensor
    tau_max: Tensor
    best_score: Tensor

    @classmethod
    def init(
        cls,
        heuristic: Tensor,
    ):
        batch_size = heuristic.shape[0]

        def init_param(value: float):
            return torch.full((batch_size,), value, dtype=torch.float32)

        return cls(
            heuristic=heuristic,
            pheremone=torch.ones_like(heuristic),
            tau_min=init_param(0),
            tau_max=init_param(1),
            best_score=init_param(-torch.inf),
            batch_size=[batch_size],
        )

    def update_max_min(self, iteration_best_score: Tensor, params: ACOParams):
        indices = torch.arange(iteration_best_score.shape[0])

        is_better = iteration_best_score > self.best_score
        better_i = indices[is_better]
        if better_i.numel() == 0:
            return

        self.best_score[better_i] = iteration_best_score[better_i]
        self.tau_max[better_i] = params.rho * self.best_score[better_i]
        self.tau_min[better_i] = self.tau_max[better_i] / params.a


def aco_search(
    params: ACOParams,
    graph: TorchGraph,
    state: MMACSState,
    current_node: Tensor,
    current_budget: Tensor,
    current_path: Path,
    num_rollouts: int,
    p_f: float,
    kappa: float,
) -> Tuple[Tensor]:
    # old_pheremone = state.pheremone

    for i in range(10):
        # TODO: There is a looping bug somewhere..
        output = aco_rollout(
            params,
            graph,
            state,
            current_path,
            current_node,
            current_budget,
            num_rollouts,
            p_f,
            kappa,
            action_selection_fn=ada_ir_action_selection,
        )
        bounded_topk_update(params, output, state, k=None)
        evaluate_ranking(graph, output, num_rollouts, kappa, visualize=False)

    # new_pheremone = state.pheremone

    # evaluate_ranking(graph, output, num_rollouts, kappa)

    # plot_heuristics(
    #     heuristics=[old_pheremone[0], new_pheremone[0]],
    #     titles=["old", "new"],
    #     rows=1,
    #     cols=2,
    # )

    # assert False

    Q = output.path.reward.sum(-1).sum(-1) / num_rollouts
    return Q


# -- Rollout action selection
def aco_action_selection(
    pheremone: Tensor, heuristic: Tensor, mask: Tensor, params: ACOParams
):
    score = (pheremone**params.alpha) * (heuristic**params.beta)
    return torch.multinomial(torch.masked_fill(score, mask, 0), num_samples=1).squeeze()


def ir_action_selection(
    pheremone: Tensor, heuristic: Tensor, mask: Tensor, params: ACOParams
):
    score = (pheremone**params.alpha) * (heuristic**params.beta)
    r = torch.rand_like(score)
    return torch.argmax(torch.masked_fill(score * r, mask, 0), dim=-1).squeeze()


def ada_ir_action_selection(
    pheremone: Tensor, heuristic: Tensor, mask: Tensor, params: ACOParams
):
    score = (pheremone**params.alpha) * (heuristic**params.beta)
    r = torch.rand_like(score) ** params.lr
    return torch.argmax(torch.masked_fill(score * r, mask, 0), dim=-1).squeeze()


# -- Rollout
@tensorclass
class RolloutOutput:
    path: Path
    cost: Tensor
    residual: Tensor


def aco_rollout(
    params: ACOParams,
    graph: TorchGraph,
    state: MMACSState,
    current_path: Tensor,
    starting_node: Tensor,
    budget: Tensor,
    num_rollouts: int,
    p_f: float,
    kappa: float,
    action_selection_fn: Callable,
) -> RolloutOutput:
    # Define shapes
    batch_size, num_nodes = graph.size()
    sim_shape = (batch_size, num_rollouts)
    flatten_shape = (batch_size * num_rollouts,)

    # Preload data
    samples = graph.edges["samples"]
    weights = graph.edges["distance"]
    goal_node = graph.extra["goal_node"]
    rewards = graph.nodes["reward"]

    # Create indices
    indices = torch.arange(batch_size)
    # batch_indices will query batch-level resources, like graph rewards and edge weights
    batch_indices = (
        torch.arange(batch_size).unsqueeze(-1).expand(sim_shape).flatten()
    )  # [B*S]
    # sim_indices will query simulation level resources, like simulated_budget and masks
    sim_indices = torch.arange(batch_size * num_rollouts)  # [B*S]

    # Local Failure mask
    failure_mask = torch.zeros((batch_size * num_rollouts, num_nodes))

    # Sample traverse cost
    ts = sample_traverse_cost(current_path, graph, num_rollouts, kappa)
    sampled_budgets = budget.unsqueeze(-1) - ts  # [B, S]
    simulated_budgets = sampled_budgets.flatten()
    starting_budgets = simulated_budgets.clone()

    # Loop State
    current_nodes = (
        starting_node.unsqueeze(-1).broadcast_to(sim_shape).flatten().clone()
    )  # [B*S]

    # Create Paths and add mask for previous nodes
    sim_paths = Path.empty(batch_size, num_nodes)
    sim_paths.mask = current_path.mask.clone()
    leaf_r = rewards[indices, starting_node]
    sim_paths.append(indices, starting_node, leaf_r)

    # Expand and flatten paths to size [B*S]
    sim_paths = sim_paths.unsqueeze(-1).expand(sim_shape).flatten().clone()

    while sim_indices.numel() > 0:
        # 0. Define action buffer
        new_nodes = torch.empty(flatten_shape, dtype=torch.long)

        # 1. Action selection
        # 1a. Compute valid node mask
        mask = torch.logical_or(sim_paths.mask[sim_indices], failure_mask[sim_indices])
        is_invalid = mask.sum(-1) == num_nodes
        is_valid = ~is_invalid

        # 1b. If mask has no valid nodes, go to goal
        b_invalid_i, s_invalid_i = batch_indices[is_invalid], sim_indices[is_invalid]
        if s_invalid_i.numel() > 0:
            new_nodes[s_invalid_i] = goal_node[b_invalid_i].clone()

        # 1c. Action Selection
        b_valid_i, s_valid_i = batch_indices[is_valid], sim_indices[is_valid]
        mask = mask[is_valid]
        c = current_nodes[s_valid_i]
        p = state.pheremone[b_valid_i, c]
        h = state.heuristic[b_valid_i, c]
        new_nodes[s_valid_i] = action_selection_fn(p, h, mask, params)

        # 2. Determine if action is goal
        b_cont_i, s_cont_i = b_valid_i, s_valid_i

        # 3. If not goal, compute failure probability
        if s_cont_i.numel() > 0:
            # 3a. Compute failure probability
            c = current_nodes[s_cont_i]
            n = new_nodes[s_cont_i]
            g = goal_node[b_cont_i]
            b = simulated_budgets[s_cont_i]
            sample_c_n = samples[b_cont_i, c, n]
            sample_n_g = samples[b_cont_i, n, g]
            failure_prob = compute_failure_prob(sample_c_n, sample_n_g, b)

            # Determine whether node failed or succeeded
            below_failure = failure_prob <= p_f
            b_suc_i, s_suc_i = b_cont_i[below_failure], s_cont_i[below_failure]
            # 3b. if Pr[...] <= p_f, add to path
            if s_suc_i.numel() > 0:
                c = current_nodes[s_suc_i]
                n = new_nodes[s_suc_i]
                w = weights[b_suc_i, c, n]

                # Sample and update budget
                sampled_cost = sample_costs(w, num_samples=1, kappa=kappa)
                simulated_budgets[s_suc_i] -= sampled_cost.squeeze(-1)

                # Add to path
                r = rewards[b_suc_i, n]
                sim_paths.append(s_suc_i, n, r)

                # Change current node
                current_nodes[s_suc_i] = n

                # Clear failure mask
                failure_mask[s_suc_i] = 0

            # 3c. Update local failure mask so we don't choose this again
            s_fail_i = s_cont_i[~below_failure]
            if s_fail_i.numel() > 0:
                n = new_nodes[s_fail_i]
                failure_mask[s_fail_i, n] = 1

        # 4: else is goal, add to path and return
        b_goal_i, s_goal_i = b_invalid_i, s_invalid_i
        if s_goal_i.numel() > 0:
            c = current_nodes[s_goal_i]
            n = new_nodes[s_goal_i]
            w = weights[b_goal_i, c, n]

            # Sample and update budget
            sampled_cost = sample_costs(w, num_samples=1, kappa=kappa)
            simulated_budgets[s_goal_i] -= sampled_cost.squeeze(-1)

            # Add to path
            r = rewards[b_goal_i, n]
            sim_paths.append(s_goal_i, n, r)

        # 5. Update Loop State
        batch_indices, sim_indices = b_valid_i, s_valid_i

    # Format output
    sim_paths = sim_paths.reshape(sim_shape)
    sim_costs = (starting_budgets - simulated_budgets).reshape(sim_shape)
    sim_residuals = simulated_budgets.reshape(sim_shape)

    return RolloutOutput(
        path=sim_paths,
        cost=sim_costs,
        residual=sim_residuals,
        batch_size=[batch_size, num_rollouts],
    )


# -- Rollout Utils
def sample_traverse_cost(path: Path, graph: TorchGraph, num_samples: int, kappa: float):
    batch_size, max_length = path.size()
    indices = torch.arange(batch_size)

    weights = graph.edges["distance"]
    total_sampled_cost = torch.zeros((batch_size, num_samples))

    path_index = 1
    while path_index < max_length:
        prev_node = path.nodes[indices, path_index - 1]
        current_node = path.nodes[indices, path_index]
        weight = weights[indices, prev_node, current_node]

        is_continuing = current_node != -1
        indices = indices[is_continuing]
        if indices.numel() == 0:
            break

        sampled_cost = sample_costs(weight[is_continuing], num_samples, kappa)
        total_sampled_cost[indices] += sampled_cost

        path_index += 1

    return total_sampled_cost


def compute_failure_prob(sample_c_n: Tensor, sample_n_g: Tensor, B: Tensor) -> Tensor:
    total_sample_cost = sample_c_n + sample_n_g
    return (
        torch.sum(total_sample_cost > B.unsqueeze(-1), dim=-1)
        / total_sample_cost.shape[-1]
    )


# -- Path scoring methods
def penalty_scoring_fn(output: RolloutOutput, params: ACOParams):
    R = output.path.reward.sum(-1)
    F = (output.residual < 0).float()
    return R * (1 - params.fail_penalty * F)


def penalty_cost_scoring_fn(output: RolloutOutput, params: ACOParams):
    R = output.path.reward.sum(-1)
    F = (output.residual < 0).float()
    L = output.cost
    return (R / L) * (1 - params.fail_penalty * F)


# -- Update Pheremone
# formulas for MMACS w/ rank based pheremone update
# Bounds:
# tau_max = rho * score_best
# tau_min = tau_max / a
# tunable parameters
# 1. rho: global evaporation rate
# 2. a: scaling factor
def bounded_topk_update(
    params: ACOParams,
    output: RolloutOutput,
    state: MMACSState,
    k: Optional[int] = None,
    scoring_fn: Callable = penalty_scoring_fn,
):
    """Rank-based Top-k pheremone update"""
    batch_size, num_rollouts = output.shape
    k = num_rollouts if k is None else k

    # Batch indices
    indices = torch.arange(batch_size)
    b_indices = indices.unsqueeze(-1).expand((-1, num_rollouts))

    # Compute ranking weights
    ranks = torch.arange(1, k + 1)
    weights = (k - ranks + 1) / torch.sum(ranks, dim=-1)
    # Expand weights to batch size
    weights = weights.unsqueeze(0).expand((batch_size, -1))

    # Score and sort outputs
    score = scoring_fn(output, params)
    sorted_i = torch.argsort(score, descending=True, dim=-1)
    sorted_output = output[b_indices, sorted_i]
    sorted_weights = weights[b_indices, sorted_i]
    sorted_score = score[b_indices, sorted_i]

    # Compute weighted scores
    weighted_score = sorted_weights * sorted_score

    # Compute update pheremone matrix
    update_matrix = compute_update(sorted_output, state, weighted_score)

    # Compute pheremone
    new_pheremone = (1 - params.rho) * state.pheremone + params.rho * update_matrix

    # Update tau_min and tau_max
    best_score = weighted_score[indices, 0]
    state.update_max_min(best_score, params)

    # Clamp pheremone between tau_min and tau_max
    # Element-wise unsqueeze
    tau_min = state.tau_min.unsqueeze(-1).unsqueeze(-1)
    tau_max = state.tau_max.unsqueeze(-1).unsqueeze(-1)

    new_pheremone = torch.clamp(new_pheremone, min=tau_min, max=tau_max)

    # Update pheremone
    state.pheremone = new_pheremone


def compute_update(output: RolloutOutput, state: MMACSState, score: Tensor):
    """Compute pheremone update matrix"""
    batch_size, num_rollouts = output.shape
    _, _, max_length = output.path.nodes.shape
    update_matrix = torch.zeros_like(state.pheremone)

    # Batch and sim indices
    b_indices = (
        torch.arange(batch_size).unsqueeze(-1).expand((-1, num_rollouts)).flatten()
    )
    s_indices = torch.arange(batch_size * num_rollouts)

    output = output.flatten()
    score = score.flatten()

    path_index = 1
    while path_index < max_length:
        current_node = output.path.nodes[s_indices, path_index]

        is_continuing = current_node != -1
        b_indices, s_indices = b_indices[is_continuing], s_indices[is_continuing]
        if s_indices.numel() == 0:
            break

        current_node = current_node[is_continuing]
        prev_node = output.path.nodes[s_indices, path_index - 1]
        w = score[s_indices]

        torch.index_put_(
            update_matrix, [b_indices, prev_node, current_node], w, accumulate=True
        )

        path_index += 1

    return update_matrix


# -- Debugging
def evaluate_ranking(
    graph: TorchGraph,
    output: RolloutOutput,
    num_samples: int,
    kappa: float,
    penalty: float = 0.1,
    visualize: bool = True,
):
    batch_size, num_nodes = graph.size()
    _, num_rollouts, max_length = output.path.nodes.shape

    # Indices
    b_indices = torch.arange(batch_size).unsqueeze(-1).expand((-1, num_rollouts))

    # Compute score
    R = output.path.reward.sum(-1)
    F = (output.residual < 0).float()
    L = output.cost
    # score = (R / L) * (1 - penalty * F)
    score = R * (1 - penalty * F)

    sorted_i = torch.argsort(score, descending=True, dim=-1)
    sorted_output = output[b_indices, sorted_i]

    # Print Average Values
    avg_R = output.path.reward.sum(-1).sum(-1) / num_rollouts
    failure_prob = (output.residual < 0).sum(-1) / num_rollouts
    print(f"Q: {float(avg_R[0]):.5f}, F: {float(failure_prob[0]):.5f}")

    # evalutae the best, median, and worst path
    b_i = 0
    med_idx = int(num_rollouts / 2)
    viz_idx = [0, med_idx, -1]
    viz_outputs = [sorted_output[b_i][i] for i in viz_idx]
    visualize_output(graph[b_i], viz_outputs, num_samples, kappa, visualize)


def visualize_output(
    graph: TorchGraph,
    outputs: list[RolloutOutput],
    num_samples: int,
    kappa: float,
    visualize: bool = True,
):
    infos = []
    paths = []
    for o in outputs:
        failure_prob, avg_cost = evaluate_path(
            o.path.unsqueeze(0), graph.unsqueeze(0), num_samples, kappa
        )
        info = (
            "ACO; "
            + f"R: {o.path.reward.sum(-1):.5f}, "
            + f"Sc: {float(o.cost):.5f}, "
            + f"Sr: {float(o.residual):.5f}, "
            + f"B: {float(o.cost - o.residual):.5f}, "
            + f"C: {float(avg_cost):.5f}, "
            + f"F: {float(failure_prob):.3f} "
            + f"N: {int(o.path.length)}"
        )
        print(info)
        paths.append(o.path)
        infos.append(info)

    if visualize:
        plot_solutions(graph, paths=paths, titles=infos, rows=len(outputs), cols=1)


def visualize_pheremones(state: MMACSState):
    plot_heuristics(
        heuristics=[state.heuristic[0], state.pheremone[0]],
        titles=["heuristic", "pheremone"],
        rows=1,
        cols=2,
    )
