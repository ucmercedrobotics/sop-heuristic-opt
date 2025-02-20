from typing import Callable, Optional
import random

import torch
from torch import Tensor
from torch.distributions import Categorical
from tensordict import tensorclass

from sop.utils.path import Path
from sop.utils.graph import TorchGraph
from sop.utils.sample import sample_costs


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
    score: Tensor, mask: Tensor, eps: float = 0.1, store_log_probs: bool = False
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


# TODO: Integrate this into Rollout
def sample_traverse_cost(
    path: Path, graph: TorchGraph, num_samples: int, kappa: float = 0.5
) -> Tensor:
    weights = graph.edges["distance"]

    batch_size, max_length = path.size()
    indices = torch.arange(batch_size)

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
