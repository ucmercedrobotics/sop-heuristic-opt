from typing import Tuple, Callable, Optional
from dataclasses import dataclass
import time
from tqdm import tqdm

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


def small_heuristic(batch_size: int, num_nodes: int):
    return torch.ones((batch_size, num_nodes, num_nodes)).softmax(-1)


def mcts_sopcc_heuristic(rewards: Tensor, sampled_costs: Tensor):
    average_cost = sampled_costs.mean(dim=-1)
    return rewards.unsqueeze(-1) / average_cost


def mcts_sopcc_norm_heuristic(rewards: Tensor, sampled_costs: Tensor):
    # Average and normalize costs
    s = sampled_costs.mean(dim=-1)
    s_norm = s / s.sum(dim=-1, keepdim=True)
    # Average and normalize rewards
    r_norm = rewards / rewards.sum(-1, keepdim=True)
    return (r_norm.unsqueeze(-1) + 1e-5) / s_norm


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
    fail_penalty: float = 1.0
    a: float = 5
    topk: int = 10


@torch.no_grad
def sop_aco_solver(
    params: ACOParams,
    graph: TorchGraph,
    heuristic: Tensor,
    num_rollouts: int,
    num_iterations: int,
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

    # TQDM
    # pbar = tqdm()

    while indices.numel() > 0:
        next_node = aco_search(
            params=params,
            state=state[indices],
            graph=graph[indices],
            current_node=current_node[indices],
            current_budget=current_budget[indices],
            current_path=path[indices],
            num_rollouts=num_rollouts,
            num_iterations=num_iterations,
            p_f=p_f,
            kappa=kappa,
        )

        # Get values
        r = rewards[indices, next_node]
        w = weights[indices, current_node[indices], next_node]

        # Update budget
        sampled_cost = sample_costs(w, num_samples=1, kappa=kappa)
        current_budget[indices] -= sampled_cost.squeeze(-1)

        # Add to path
        path.append(indices, next_node, r)

        # Update state
        is_not_goal = next_node != goal_node[indices]
        has_budget = current_budget[indices] > 0
        is_continuing = torch.logical_and(is_not_goal, has_budget)
        indices = indices[is_continuing]
        current_node[indices] = next_node[is_continuing]

        # pbar.update()

    # pbar.close()

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
    best_path: Tensor

    @classmethod
    def init(
        cls,
        heuristic: Tensor,
    ):
        batch_size, num_nodes, _ = heuristic.shape

        def init_param(value: float):
            return torch.full((batch_size,), value, dtype=torch.float32)

        return cls(
            heuristic=heuristic,
            pheremone=torch.ones_like(heuristic),
            tau_min=init_param(0),
            tau_max=init_param(1),
            best_score=init_param(-torch.inf),
            best_path=torch.zeros((batch_size, num_nodes + 1), dtype=torch.long),
            batch_size=[batch_size],
        )


def aco_search(
    params: ACOParams,
    state: MMACSState,
    graph: TorchGraph,
    current_node: Tensor,
    current_budget: Tensor,
    current_path: Path,
    num_rollouts: int,
    num_iterations: int,
    p_f: float,
    kappa: float,
) -> Tuple[Tensor]:
    def scoring_fn(params, graph, output):
        # return fail_prob_scoring_fn(params, graph, output, p_f)
        return weighted_fail_prob_scoring_fn(params, graph, output, p_f)

    for k in range(num_iterations):
        # TODO: There is a looping bug somewhere.. appears sometimes...
        score = (state.pheremone**params.alpha) * (state.heuristic**params.beta)

        output = aco_rollout(
            params,
            score,
            graph,
            current_path,
            current_node,
            current_budget,
            num_rollouts,
            p_f,
            action_selection_fn=ada_ir_action_selection2,
        )

        score = bounded_topk_update(
            params, state, graph, output, k=None, scoring_fn=scoring_fn
        )

    return select_action(state)


def select_action(state: MMACSState):
    indices = torch.arange(state.shape[0])
    next_node = state.best_path[indices, 1]
    state.best_score = torch.full_like(state.best_score, -torch.inf)
    state.pheremone = torch.ones_like(state.heuristic)
    return next_node


# -- Rollout action selection
def aco_action_selection(
    params: ACOParams, pheremone: Tensor, heuristic: Tensor, mask: Tensor
):
    score = (pheremone**params.alpha) * (heuristic**params.beta)
    return torch.multinomial(torch.masked_fill(score, mask, 0), num_samples=1).squeeze()


def ir_action_selection(
    params: ACOParams, pheremone: Tensor, heuristic: Tensor, mask: Tensor
):
    score = (pheremone**params.alpha) * (heuristic**params.beta)
    r = torch.rand_like(score)
    return torch.argmax(torch.masked_fill(score * r, mask, 0), dim=-1)


def ada_ir_action_selection(
    params: ACOParams, pheremone: Tensor, heuristic: Tensor, mask: Tensor
):
    score = (pheremone**params.alpha) * (heuristic**params.beta)
    r = torch.rand_like(score) ** params.lr
    sr = score * r
    action = torch.argmax(torch.masked_fill(sr, mask, 0), dim=-1)
    return action


def ada_ir_action_selection2(params: ACOParams, score: Tensor, mask: Tensor):
    r = torch.rand_like(score) ** params.lr
    sr = score * r
    action = torch.argmax(torch.masked_fill(sr, mask, 0), dim=-1)
    return action


# -- Rollout
@tensorclass
class RolloutOutput:
    path: Path
    cost: Tensor
    residual: Tensor


@tensorclass
class RolloutState:
    batch_i: Tensor
    current_node: Tensor
    budget: Tensor
    path: Path
    failure_mask: Tensor

    @classmethod
    def empty(
        cls,
        graph: TorchGraph,
        current_path: Tensor,
        current_node: Tensor,
        current_budget: Tensor,
    ):
        batch_size, num_nodes = graph.size()
        indices = torch.arange(batch_size)

        # Create path and add current node
        sim_path = Path.empty(batch_size, num_nodes)
        sim_path.mask = current_path.mask
        r = graph.nodes["reward"][indices, current_node]
        sim_path.append(indices, current_node, r)

        return RolloutState(
            batch_i=indices,
            current_node=current_node,
            budget=current_budget,
            path=sim_path,
            failure_mask=torch.zeros((batch_size, num_nodes), dtype=torch.bool),
            batch_size=[batch_size],
        )

    def update(self, indices: Tensor, new_node: Tensor, reward: Tensor, cost: Tensor):
        self.budget[indices] -= cost
        self.path.append(indices, new_node, reward)
        self.current_node[indices] = new_node
        self.failure_mask[indices] = 0

    def update_failure_mask(self, indices: Tensor, new_node: Tensor):
        self.failure_mask[indices, new_node] = 1


def aco_rollout2(
    params: ACOParams,
    score: Tensor,
    graph: TorchGraph,
    current_path: Tensor,
    current_node: Tensor,
    current_budget: Tensor,
    num_rollouts: int,
    p_f: float,
    action_selection_fn: Callable,
):
    # Define shapes
    batch_size, num_nodes = graph.size()
    sim_shape = (batch_size, num_rollouts)
    flatten_shape = (batch_size * num_rollouts,)

    # Data alias
    goal_node = graph.extra["goal_node"]
    rewards = graph.nodes["reward"]
    samples = graph.edges["samples"]

    # Create rollout state
    rollout_state = RolloutState.empty(
        graph, current_path, current_node, current_budget
    )
    # Expand rollout from [batch_size,] -> [batch_size, num_rollouts]
    rollout_state = rollout_state.unsqueeze(-1).expand(sim_shape)
    # Flatten state [batch_size, num_rollouts] -> [batch_size * num_rollouts]
    rollout_state = rollout_state.flatten().clone()

    # Create sim indices
    sim_i = torch.arange(batch_size * num_rollouts)

    # Sample index to avoid resampling cost
    num_samples = samples.shape[-1]
    sample_i = torch.randint(low=0, high=num_samples, size=flatten_shape)

    while sim_i.numel() > 0:
        # -- 1. Compute mask
        rs = rollout_state[sim_i]
        mask = torch.logical_or(rs.path.mask, rs.failure_mask)
        is_invalid = mask.sum(-1) == num_nodes
        is_valid = ~is_invalid

        # -- 2. Valid Nodes
        # a. Choose new node
        # b. check if failure_prob < p_f
        # c. Add to path or update failure mask
        valid_i = sim_i[is_valid]
        if valid_i.numel() > 0:
            rs = rollout_state[valid_i]
            valid_m = mask[is_valid]

            # 2a. Sample new node
            s = score[rs.batch_i, rs.current_node]
            new_node = action_selection_fn(params, s, valid_m)

            # 2b. Check if failure_prob < p_f
            g = goal_node[rs.batch_i]
            sample_c_n = samples[rs.batch_i, rs.current_node, new_node]
            sample_n_g = samples[rs.batch_i, new_node, g]
            failure_prob = compute_failure_prob(sample_c_n, sample_n_g, rs.budget)

            below_failure = failure_prob <= p_f
            below_i = valid_i[below_failure]
            # 2c. if Pr[...] <= p_f, add to path
            if below_i.numel() > 0:
                rs = rollout_state[below_i]
                n = new_node[below_failure]

                # Sample w/ buffer
                si = sample_i[below_i]
                sampled_cost = samples[rs.batch_i, rs.current_node, n, si]
                sample_i[below_i] = (si + 1) % num_samples
                # Get reward
                r = rewards[rs.batch_i, n]

                # Update rollout state
                rollout_state.update(below_i, n, r, sampled_cost)

            above_failure = ~below_failure
            fail_i = valid_i[above_failure]
            # 2d. Update local failure mask so we don't choose this again
            if fail_i.numel() > 0:
                n = new_node[above_failure]
                rollout_state.update_failure_mask(fail_i, n)

        # -- 3. Invalid nodes
        # a. Go to goal node
        invalid_i = sim_i[is_invalid]
        if invalid_i.numel() > 0:
            rs = rollout_state[invalid_i]
            g = goal_node[rs.batch_i]

            # Sample w/ buffer
            si = sample_i[invalid_i]
            sampled_cost = samples[rs.batch_i, rs.current_node, g, si]
            sample_i[invalid_i] = (si + 1) % num_samples
            # Get reward
            r = rewards[rs.batch_i, g]

            rollout_state.update(invalid_i, g, r, sampled_cost)

        # 4. Update Loop State
        sim_i = valid_i

    # Format output
    rollout_state = rollout_state.reshape(sim_shape)
    rollout_cost = current_budget.unsqueeze(-1) - rollout_state.budget

    return RolloutOutput(
        path=rollout_state.path,
        cost=rollout_cost,
        residual=rollout_state.budget,
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
def penalty_scoring_fn(params: ACOParams, graph: TorchGraph, output: RolloutOutput):
    R = output.path.reward.sum(-1)
    F = (output.residual < 0).float()
    return R * (1 - params.fail_penalty * F)


def penalty_cost_scoring_fn(
    params: ACOParams, graph: TorchGraph, output: RolloutOutput
):
    R = output.path.reward.sum(-1)
    F = (output.residual < 0).float()
    L = output.cost
    return (R / L) * (1 - params.fail_penalty * F)


def fail_prob_scoring_fn(
    params: ACOParams, graph: TorchGraph, output: RolloutOutput, p_f: float
):
    R = output.path.reward.sum(-1)
    avg_cost, failure_prob = evaluate_rollouts(graph, output)
    F = failure_prob > p_f
    return R * (1 - params.fail_penalty * F) + 1e-9


def fail_prob_avg_cost_scoring_fn(
    params: ACOParams, graph: TorchGraph, output: RolloutOutput, p_f: float
):
    R = output.path.reward.sum(-1)
    avg_cost, failure_prob = evaluate_rollouts(graph, output)
    F = failure_prob > p_f
    return (R / avg_cost) * (1 - params.fail_penalty * F)


def weighted_fail_prob_scoring_fn(
    params: ACOParams, graph: TorchGraph, output: RolloutOutput, p_f: float
):
    R = output.path.reward.sum(-1)
    avg_cost, failure_prob = evaluate_rollouts(graph, output)
    F = torch.clamp(failure_prob - p_f, min=0)
    return R * (1 - params.fail_penalty * F)


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
    state: MMACSState,
    graph: TorchGraph,
    output: RolloutOutput,
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
    score = scoring_fn(params, graph, output)
    sorted_i = torch.argsort(score, descending=True, dim=-1)
    sorted_output = output[b_indices, sorted_i]
    sorted_weights = weights[b_indices, sorted_i]
    sorted_score = score[b_indices, sorted_i]

    # Compute weighted scores
    weighted_score = sorted_weights * sorted_score

    # Compute update pheremone matrix
    update_matrix = compute_update(state, sorted_output, weighted_score)

    # Compute pheremone
    new_pheremone = (1 - params.rho) * state.pheremone + params.rho * update_matrix

    # Update tau_min and tau_max
    best_score = weighted_score[indices, 0]
    best_output = sorted_output[indices, 0]
    update_max_min(params, state, best_output, best_score)

    # Clamp pheremone between tau_min and tau_max
    # Element-wise unsqueeze
    tau_min = state.tau_min.unsqueeze(-1).unsqueeze(-1)
    tau_max = state.tau_max.unsqueeze(-1).unsqueeze(-1)

    new_pheremone = torch.clamp(new_pheremone, min=tau_min, max=tau_max)

    # Update pheremone
    state.pheremone = new_pheremone

    # For testing purposes
    return score


def compute_update(state: MMACSState, output: RolloutOutput, score: Tensor):
    """Compute pheremone update matrix"""
    batch_size, num_rollouts, max_length = output.path.nodes.shape
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


def update_max_min(
    params: ACOParams,
    state: MMACSState,
    best_output: RolloutOutput,
    best_score: Tensor,
):
    indices = torch.arange(state.shape[0])

    is_better = best_score > state.best_score
    better_i = indices[is_better]
    if better_i.numel() == 0:
        return

    state.best_score[better_i] = best_score[better_i]
    state.best_path[better_i] = best_output.path.nodes[better_i]
    state.tau_max[better_i] = params.rho * state.best_score[better_i]
    state.tau_min[better_i] = state.tau_max[better_i] / params.a


# -- Evaluation
def evaluate_rollouts(graph: TorchGraph, output: RolloutOutput):
    batch_size, num_rollouts, max_length = output.path.nodes.shape

    # Fetch precomputed samples
    samples = graph.edges["samples"]
    num_samples = samples.shape[-1]

    # Batch and sim indices
    b_indices = (
        torch.arange(batch_size).unsqueeze(-1).expand((-1, num_rollouts)).flatten()
    )
    s_indices = torch.arange(batch_size * num_rollouts)

    samples = graph.edges["samples"]

    output = output.flatten()
    total_sampled_cost = torch.zeros((batch_size * num_rollouts, num_samples))

    path_index = 1
    while path_index < max_length:
        current_node = output.path.nodes[s_indices, path_index]

        is_continuing = current_node != -1
        b_indices, s_indices = b_indices[is_continuing], s_indices[is_continuing]
        if s_indices.numel() == 0:
            break

        current_node = current_node[is_continuing]
        prev_node = output.path.nodes[s_indices, path_index - 1]
        s = samples[b_indices, prev_node, current_node]
        total_sampled_cost[s_indices] += s

        path_index += 1

    # Avg cost length
    avg_cost = (
        total_sampled_cost.reshape(batch_size, num_rollouts, -1).sum(-1) / num_rollouts
    )

    # Failure Prob
    budget = output.cost + output.residual
    F = (total_sampled_cost > budget.unsqueeze(-1)).reshape(
        (batch_size, num_rollouts, -1)
    ).sum(-1) / num_samples

    return avg_cost, F


# -- Debugging
def evaluate_ranking(
    graph: TorchGraph,
    output: RolloutOutput,
    score: Tensor,
    num_samples: int,
    kappa: float,
    penalty: float = 0.1,
    visualize: bool = True,
):
    batch_size, num_nodes = graph.size()
    _, num_rollouts, max_length = output.path.nodes.shape

    # Indices
    b_indices = torch.arange(batch_size).unsqueeze(-1).expand((-1, num_rollouts))

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
            + f"B: {float(o.cost + o.residual):.5f}, "
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
