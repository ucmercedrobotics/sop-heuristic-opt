from typing import Tuple, Callable, Optional
import random
from dataclasses import dataclass
import time
from tqdm import tqdm
import math

import torch
from torch import Tensor
from torch.distributions import Categorical

from tensordict import tensorclass

from sop.utils.path import Path
from sop.utils.graph_torch import TorchGraph
from sop.utils.sample import sample_costs

# Debugging
from sop.utils.visualization import plot_solutions, plot_heuristics
from sop.utils.path import evaluate_path, path_to_heatmap


@torch.no_grad
def sop_aco_solver(
    graph: TorchGraph,
    heuristic: Tensor,
    num_rollouts: int,
    num_iterations: int,
    p_f: float,
    kappa: float,
    search_fn: Callable,
) -> Tuple[Path, Tensor]:
    start_node = graph.extra["start_node"]
    budget = graph.extra["budget"]
    goal_node = graph.extra["goal_node"]
    rewards = graph.nodes["reward"]
    weights = graph.edges["distance"]

    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    # Loop State
    current_node = start_node.clone()
    current_budget = budget.clone()

    # Initialize Path
    path = Path.empty(batch_size, num_nodes)
    path.append(indices, current_node)
    # Mask goal node
    path.mask[indices, goal_node] = 1

    # Progress
    pbar = tqdm()

    # Main Loop
    while indices.numel() > 0:
        # Get next node
        # next_node = search(
        next_node = search_fn(
            heuristic=heuristic[indices],
            graph=graph[indices],
            current_node=current_node[indices],
            current_budget=current_budget[indices],
            current_path=path[indices],
            num_rollouts=num_rollouts,
            num_iterations=num_iterations,
            p_f=p_f,
        )

        # Get next values
        r = rewards[indices, next_node]
        w = weights[indices, current_node[indices], next_node]

        # Update budget
        sampled_cost = sample_costs(w, num_samples=1, kappa=kappa)
        current_budget[indices] -= sampled_cost.squeeze(-1)

        # Add to path
        path.append(indices, next_node, r)

        # Check if done
        is_not_goal = next_node != goal_node[indices]
        has_budget = current_budget[indices] > 0
        is_continuing = torch.logical_and(is_not_goal, has_budget)

        # Update loop state
        indices = indices[is_continuing]
        current_node[indices] = next_node[is_continuing]

        # Update pbar
        pbar.update()

    pbar.close()

    # Check if run was a success
    is_success = current_budget >= 0

    return path, is_success


def search(
    heuristic: Tensor,
    graph: TorchGraph,
    current_node: Tensor,
    current_budget: Tensor,
    current_path: Path,
    num_rollouts: int,
    num_iterations: int,
    p_f: float,
) -> Tensor:
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    output = rollout(
        heuristic, graph, current_node, current_budget, current_path, num_rollouts, p_f
    )
    scores = reward_failure_scoring_fn(output, p_f)

    sorted_output, _, _ = rank_output(output, scores)

    best_path = sorted_output.path[indices, 0]
    next_node = best_path.nodes[indices, 1]

    return next_node


def aco_search(
    heuristic: Tensor,
    graph: TorchGraph,
    current_node: Tensor,
    current_budget: Tensor,
    current_path: Path,
    num_rollouts: int,
    num_iterations: int,
    p_f: float,
) -> Tensor:
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    state = ACOState.init(heuristic)

    alpha = 1
    beta = 1

    for k in range(num_iterations):
        score = (state.pheremone**alpha) * (heuristic**beta)
        output = rollout(
            score,
            graph,
            current_node,
            current_budget,
            current_path,
            num_rollouts,
            p_f,
        )
        scores = reward_failure_scoring_fn(output, p_f)
        pheremone_update(state, graph, output, scores, topk=10)

    next_node = state.best_path[indices, 1]
    return next_node


# -- Action Selection
def categorical_action_selection(
    score: Tensor, mask: Tensor, store_log_probs: bool = False
):
    # Taken from DeepACO: https://github.com/henry-yeh/DeepACO/blob/main/tsp/aco.py
    masked_score = torch.masked_fill(score, mask, -torch.inf)
    dist = Categorical(logits=masked_score)
    actions = dist.sample()
    if store_log_probs:
        log_prob = dist.log_prob(actions)
        return actions, log_prob
    return actions


def eps_greedy_action_selection(
    score: Tensor, mask: Tensor, store_log_probs: bool = False, eps: float = 0.1
):
    # Sample actions
    masked_score = torch.masked_fill(score, mask, -torch.inf)
    dist = Categorical(logits=masked_score)
    actions = dist.sample()

    # Sample random if p < eps
    batch_size = score.shape[0]
    indices = torch.arange(batch_size)
    p = torch.rand((batch_size,))
    is_random = p < eps
    random_i = indices[is_random]
    if random_i.numel() > 0:
        random_dist = ~(mask[random_i])
        actions[random_i] = torch.multinomial(random_dist.float(), 1).squeeze(-1)

    if store_log_probs:
        log_prob = dist.log_prob(actions)
        return actions, log_prob
    return actions


# -- ROLLOUT
# B: Batch Size, N: Num Nodes, R: Num Rollouts, S: Num Samples


@tensorclass
class RolloutOutput:
    path: Path  # [B, R]
    residual: Tensor  # [B, R, S]
    log_probs: Optional[Tensor]  # [B, R, N + 1]


@tensorclass
class RolloutState:
    batch_i: Tensor  # [B,]
    path: Path  # [B, R]
    current_node: Tensor  # [B, R]
    budgets: Tensor  # [B, R, S]
    failure_mask: Tensor  # [B, R, N]
    log_probs: Optional[Tensor]  # [B, R, N+1]

    @classmethod
    def init(
        cls,
        current_node: Tensor,
        current_budget: Tensor,
        current_path: Path,
        num_samples: int,
        store_log_probs: bool = False,
    ):
        batch_size, max_length = current_path.size()
        num_nodes = max_length - 1
        indices = torch.arange(batch_size)

        # Create path and add current node and mask
        sim_path = Path.empty(batch_size, num_nodes)
        sim_path.mask = current_path.mask
        sim_path.append(indices, current_node)

        # Expand budget to # [B, S]
        budgets = current_budget.unsqueeze(-1).expand((batch_size, num_samples))

        # Log prob buffer
        log_probs = torch.zeros((batch_size, num_nodes)) if store_log_probs else None

        return RolloutState(
            batch_i=indices,
            path=sim_path,
            current_node=current_node,
            budgets=budgets,
            failure_mask=torch.zeros((batch_size, num_nodes), dtype=torch.bool),
            log_probs=log_probs,
            batch_size=[batch_size],
        )

    def update(
        self,
        indices: Tensor,
        new_node: Tensor,
        reward: Tensor,
        sampled_costs: Tensor,
        log_probs: Optional[Tensor] = None,
    ):
        self.budgets[indices] -= sampled_costs
        self.path.append(indices, new_node, reward)
        self.current_node[indices] = new_node
        self.failure_mask[indices] = 0

        if log_probs is not None:
            index = self.path.length[indices] - 1
            self.log_probs[indices, index] = log_probs

    def update_failure_mask(self, indices: Tensor, new_node: Tensor):
        self.failure_mask[indices, new_node] = 1


def rollout(
    score: Tensor,
    graph: TorchGraph,
    current_node: Tensor,
    current_budget: Tensor,
    current_path: Tensor,
    num_rollouts: int,
    p_f: float,
    action_selection_fn: Callable = categorical_action_selection,
    store_log_probs: bool = False,
):
    goal_node = graph.extra["goal_node"]
    rewards = graph.nodes["reward"]
    samples = graph.edges["samples"]

    # Define shapes
    batch_size, num_nodes = graph.size()
    num_samples = samples.shape[-1]
    sim_shape = (batch_size, num_rollouts)

    # Create Rollout state
    rollout_state = RolloutState.init(
        current_node, current_budget, current_path, num_samples, store_log_probs
    )
    # Expand rollout from [batch_size,] -> [batch_size, num_rollouts]
    rollout_state = rollout_state.unsqueeze(-1).expand(sim_shape)
    # Flatten state [batch_size, num_rollouts] -> [batch_size * num_rollouts]
    rollout_state = rollout_state.flatten().clone()

    # Create sim indices
    sim_i = torch.arange(batch_size * num_rollouts)

    # Sample index to avoid resampling cost
    num_samples = samples.shape[-1]
    sample_roll = random.randint(0, num_samples)

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
            if store_log_probs:
                new_node, log_probs = action_selection_fn(s, valid_m, store_log_probs)
            else:
                new_node = action_selection_fn(s, valid_m)

            # 2b. Check if failure_prob < p_f
            g = goal_node[rs.batch_i]
            sample_c_n = samples[rs.batch_i, rs.current_node, new_node]
            sample_n_g = samples[rs.batch_i, new_node, g]
            failure_prob = compute_failure_prob(sample_c_n, sample_n_g, rs.budgets)

            below_failure = failure_prob <= p_f
            below_i = valid_i[below_failure]
            # 2c. if Pr[...] <= p_f, add to path
            if below_i.numel() > 0:
                rs = rollout_state[below_i]
                n = new_node[below_failure]

                # Get samples
                sampled_costs = samples[rs.batch_i, rs.current_node, n]
                sampled_costs = torch.roll(sampled_costs, shifts=sample_roll, dims=-1)
                sample_roll = (sample_roll + 1) % num_samples
                # Get reward
                r = rewards[rs.batch_i, n]

                # Update rollout state
                if store_log_probs:
                    lp = log_probs[below_failure]
                    rollout_state.update(below_i, n, r, sampled_costs, lp)
                else:
                    rollout_state.update(below_i, n, r, sampled_costs)

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

            # Get samples
            sampled_costs = samples[rs.batch_i, rs.current_node, g]
            sampled_costs = torch.roll(sampled_costs, shifts=sample_roll, dims=-1)
            sample_roll = (sample_roll + 1) % num_samples
            # Get reward
            r = rewards[rs.batch_i, g]

            rollout_state.update(invalid_i, g, r, sampled_costs)

        # 4. Update Loop State
        sim_i = valid_i

    # Reshape output
    rollout_state = rollout_state.reshape(sim_shape)

    return RolloutOutput(
        path=rollout_state.path,
        residual=rollout_state.budgets,
        log_probs=rollout_state.log_probs,
        batch_size=[batch_size, num_rollouts],
    )


def compute_failure_prob(
    sample_c_n: Tensor, sample_n_g: Tensor, budgets: Tensor
) -> Tensor:
    total_sample_cost = sample_c_n + sample_n_g
    return torch.sum(total_sample_cost > budgets, dim=-1) / total_sample_cost.shape[-1]


# -- SCORING
def reward_failure_scoring_fn(output: RolloutOutput, p_f: float):
    reward = output.path.reward.sum(-1)
    failure_prob = (output.residual < 0).sum(-1) / output.residual.shape[-1]
    F = torch.clamp(failure_prob - p_f, min=0)
    scores = reward * (1 - F)
    return scores


def rank_output(output: RolloutOutput, scores: Tensor, topk: Optional[int] = None):
    batch_size, num_rollouts = output.shape
    topk = topk if topk is not None else num_rollouts
    topk_shape = (batch_size, topk)

    b_indices = torch.arange(batch_size).unsqueeze(-1).expand(topk_shape)

    ranked_score, ranked_i = torch.topk(scores, k=topk, dim=-1, sorted=True)
    ranked_output = output[b_indices, ranked_i]

    return ranked_output, ranked_score, ranked_i


# -- Local Heuristic Optimization
@tensorclass
class ACOState:
    pheremone: Tensor
    best_path: Tensor
    best_score: Tensor
    tau_min: Tensor
    tau_max: Tensor

    @classmethod
    def init(
        cls,
        heuristic: Tensor,
    ):
        batch_size, num_nodes, _ = heuristic.shape

        def init_param(value: float):
            return torch.full((batch_size,), value, dtype=torch.float32)

        return cls(
            pheremone=torch.ones_like(heuristic),
            best_path=torch.zeros((batch_size, num_nodes + 1), dtype=torch.long),
            best_score=init_param(-torch.inf),
            tau_min=init_param(0),
            tau_max=init_param(1),
            batch_size=[batch_size],
        )


def pheremone_update(
    state: ACOState,
    graph: TorchGraph,
    output: RolloutOutput,
    scores: Tensor,
    topk: Optional[int] = None,
    rho: float = 0.1,
    a: int = 10,
):
    batch_size, num_nodes = graph.size()
    indices = torch.arange(batch_size)

    topk_output, topk_score, topk_i = rank_output(output, scores, topk)

    # Compute ranking weights
    ranks = torch.arange(1, topk + 1)
    weights = (topk - ranks + 1) / torch.sum(ranks)
    weights = weights.unsqueeze(0).expand((batch_size, -1))

    # Compute weighted scores
    weighted_score = weights * topk_score

    # Update best
    best_output = topk_output[indices, 0]
    best_score = topk_score[indices, 0]
    update_best_path(state, best_output, best_score, rho=rho, a=a)

    # Compute update pheremone matrix
    update_matrix = compute_update_matrix(state, topk_output, weighted_score)

    # Update pheremone
    new_pheremone = (1 - rho) * (state.pheremone + update_matrix)

    # Compute exploration update
    new_pheremone = penalize_paths(new_pheremone, topk_output)

    # Normalize pheremone
    new_pheremone = new_pheremone / torch.sum(new_pheremone, dim=-1, keepdim=True)

    # Clamp between tau_min and tau_max
    # Element-wise unsqueeze
    tau_min = state.tau_min.unsqueeze(-1).unsqueeze(-1)
    tau_max = state.tau_max.unsqueeze(-1).unsqueeze(-1)
    new_pheremone = torch.clamp(new_pheremone, min=tau_min, max=tau_max)

    # Update state
    state.pheremone = new_pheremone


def compute_update_matrix(state: ACOState, output: RolloutOutput, score: Tensor):
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


def penalize_paths(pheremone: Tensor, output: RolloutOutput, eps: float = 0.9):
    batch_size, num_rollouts, max_length = output.path.nodes.shape

    # Batch and sim indices
    b_indices = (
        torch.arange(batch_size).unsqueeze(-1).expand((-1, num_rollouts)).flatten()
    )
    s_indices = torch.arange(batch_size * num_rollouts)

    output = output.flatten()

    path_index = 1
    while path_index < max_length:
        current_node = output.path.nodes[s_indices, path_index]
        is_continuing = current_node != -1
        b_indices, s_indices = b_indices[is_continuing], s_indices[is_continuing]
        if s_indices.numel() == 0:
            break

        current_node = current_node[is_continuing]
        prev_node = output.path.nodes[s_indices, path_index - 1]

        p = pheremone[b_indices, prev_node, current_node]
        torch.index_put_(pheremone, [b_indices, prev_node, current_node], eps * p)

        path_index += 1

    return pheremone


def update_best_path(
    state: ACOState,
    best_output: RolloutOutput,
    best_score: Tensor,
    rho: float,
    a: int,
):
    indices = torch.arange(state.shape[0])

    is_better = best_score > state.best_score
    better_i = indices[is_better]
    if better_i.numel() == 0:
        return

    state.best_score[better_i] = best_score[better_i]
    state.best_path[better_i] = best_output.path.nodes[better_i]
    state.tau_max[better_i] = rho * state.best_score[better_i]
    state.tau_min[better_i] = state.tau_max[better_i] / a
