from typing import Optional
import pulp
import torch
from torch import Tensor
import re

from sop.utils.graph_torch import TorchGraph
from sop.utils.path import Path

"""
The MILP formulation for SOP is taken from the following paper:
https://link.springer.com/content/pdf/10.1007/978-3-642-41575-3_30.pdf

It is defined as follows:

.. math::

# Objective function: maximize the reward of the path. (8)
\max_{\pi} \sum_{ij} \pi_{ij}*R_i

s.t.
    # Edge \pi_{i,j} is defined as a binary variable (9)
    \pi_{i,j} \in {0,1}, \forall v_i, v_j \in V

    # Only one incoming edge for each node, if any (10)
    \sum_j \pi_{ji} \leq 1, \forall v_i \in V
    # Only one outgoing edge for each node, if any (11)
    \sum_j \pi_{ij} \leq 1, \forall v_i \in V
    # Start has one outgoing edge, End has one incoming edge (12)
    \sum_j \pi_{1j} = 1; \sum_j \pi_{jn} = 1

    # Enforces flow conservation (13)
    \sum_j \pi_{ij} - \sum_j \pi_{ji} = 
    \begin{cases}
    1 & \text{if } i = 1;
    -1 & \text{if } i = n;
    0 & \text{otherwise}
    \end{cases}
    
    # Ranking to enforce no cycles in the path; M is a large constant (14)
    r_i \leq r_j - 1 + (1 - \pi_{ij}) * M, \forall v_i, v_j \in V

    # Enforce start and end node order (15)
    r_1 = 1, r_n = n, r_i \in [1, n], \forall v_i \in V

    # Chance constraint using SAA; Generate Q samples and average (17-19)
    z^q \geq \frac{\sum_{ij} \pi_{ij}*t^q_{ij} - H}{H}, \forall q \in Q
    z^q \in {0, 1}, \forall q \in Q
    \frac{\sum_q z^q}{Q} \leq \alpha'
"""


def create_pulp_instance(
    rewards: Tensor,
    costs: Tensor,
    samples: Tensor,
    budget: float,
    start: int,
    goal: int,
    num_samples: int,
    M: int = 1000,
    alpha_prime: float = 0.1,
) -> pulp.LpProblem:
    # Make sure graph is not batched
    assert rewards.ndim == 1 and costs.ndim == 2, "Graph is not in corrrect format."
    num_nodes = int(rewards.shape[0])

    # Setup maximization problem
    prob = pulp.LpProblem("SOP", pulp.LpMaximize)

    # Graph indices
    nodes = [i for i in range(num_nodes)]
    edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    adj_list = [[j for j in range(num_nodes) if i != j] for i in range(num_nodes)]

    # Define variables:
    # 1. Edge \pi_{ij} (9)
    pi = pulp.LpVariable.dicts("pi", edges, cat="Binary")

    # 2. Ranking r_i (15)
    r_indices = [i for i in nodes]
    r = pulp.LpVariable.dicts(
        "r", r_indices, lowBound=0, upBound=num_nodes - 1, cat="Integer"
    )

    # Setup objective function (8)
    prob += pulp.lpSum(pi[i, j] * rewards[j] for (i, j) in edges)

    # Add constraint (10)
    for i in nodes:
        prob += pulp.lpSum(pi[j, i] for j in adj_list[i]) <= 1

    # Add constraint (11)
    for i in nodes:
        prob += pulp.lpSum(pi[i, j] for j in adj_list[i]) <= 1

    # Add constraint (12)
    prob += pulp.lpSum(pi[start, j] for j in adj_list[start]) == 1
    prob += pulp.lpSum(pi[j, goal] for j in adj_list[goal]) == 1

    # Add constraint (13)
    # Case 1: i = start
    outgoing = pulp.lpSum(pi[start, j] for j in adj_list[start])
    incoming = pulp.lpSum(pi[j, start] for j in adj_list[start])
    prob += outgoing - incoming == 1

    # Case 2: i = end
    outgoing = pulp.lpSum(pi[goal, j] for j in adj_list[goal])
    incoming = pulp.lpSum(pi[j, goal] for j in adj_list[goal])
    prob += outgoing - incoming == -1

    # Case 3: i not start or end
    for i in range(num_nodes):
        if i == start or i == goal:
            continue
        outgoing = pulp.lpSum(pi[i, j] for j in adj_list[i])
        incoming = pulp.lpSum(pi[j, i] for j in adj_list[i])
        prob += outgoing - incoming == 0

    # Add constraint (14)
    for i in nodes:
        for j in nodes:
            if i != j:
                prob += r[i] <= r[j] - 1 + (1 - pi[i, j]) * M

    # Add constraint (15)
    prob += r[start] == 0
    prob += r[goal] == num_nodes - 1

    # SAA Constraints (17-19)
    # Define variables:
    # 1. Auxiliary variable z^q (18)
    z = pulp.LpVariable.dicts("z", range(num_samples), cat="Binary")

    # Add constraint (17)
    for q in range(num_samples):
        prob += budget + budget * z[q] >= (
            pulp.lpSum(pi[i, j] * samples[i, j, q] for (i, j) in edges)
        )

    # Add constraint (19)
    prob += pulp.lpSum(z[q] for q in range(num_samples)) / num_samples <= alpha_prime

    return prob


def extract_solution(prob):
    # Regex to extract the two edges
    # Ex. 'pi_(10,_19)' -> 10, 19
    pattern = r"pi_\((\d+),_(\d+)\)"

    edge_dict = {}
    for v in prob.variables():
        # if v.varValue > 0: # There are inaccuracies where the varValue is a very small number but still > 0
        if v.varValue > 0.1:
            if v.name.startswith("pi"):
                a, b = [int(x) for x in re.search(pattern, v.name).groups()]
                edge_dict[a] = b
    return edge_dict


def edge_list_to_path(
    edge_dict: dict, start_node: int, goal_node: int, rewards: Tensor
) -> Path:
    num_nodes = int(rewards.shape[0])
    indices = torch.arange(1)

    # Initialize path
    path = Path.empty(1, num_nodes)
    path.append(indices, start_node)

    # iterate edge_dict until goal node
    current_node = start_node
    while current_node != goal_node:
        next_node = edge_dict[current_node]
        reward = rewards[next_node]
        path.append(indices, next_node, reward=reward)

        current_node = next_node

    return path


def sop_milp_solver(
    graph: TorchGraph,
    time_limit: Optional[int] = 180,
    num_samples: int = 100,
    M: int = 1000,
    alpha_prime: float = 0.075,
) -> Path:
    R = graph.nodes["reward"]
    C = graph.edges["distance"]
    S = graph.edges["samples"]
    b = float(graph.extra["budget"])
    s = int(graph.extra["start_node"])
    g = int(graph.extra["goal_node"])

    print("Generating PuLP instance...")
    prob = create_pulp_instance(R, C, S, b, s, g, num_samples, M, alpha_prime)

    solver = pulp.getSolver("HiGHS")
    if time_limit is not None:
        solver.timeLimit = time_limit

    print("Solving instance...")
    result = prob.solve(solver)
    print(result)
    edge_dict = extract_solution(prob)
    return edge_list_to_path(edge_dict, s, g, R)
