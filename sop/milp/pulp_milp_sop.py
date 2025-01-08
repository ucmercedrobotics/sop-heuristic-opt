import pulp
import torch

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


# Utils
def generate_uniform_reward(N: tuple, device: str = "cpu") -> torch.Tensor:
    """Computes random reward for each node in the graph."""
    return torch.rand((N), device=device)


def generate_uniform_positions(N: int, device: str = "cpu") -> torch.Tensor:
    """Creates 2D positions for each node in the graph."""
    xs = torch.rand((N), device=device)  # (...size,)
    ys = torch.rand((N), device=device)  # (...size,)
    return torch.stack([xs, ys], dim=-2)  # (2, size...)


def compute_distances(x: torch.Tensor, p: int = 2):
    return torch.cdist(x, x, p=p)


def generate_example_graph(N: int, device: str = "cpu") -> torch.Tensor:
    positions = generate_uniform_positions(N, device)
    rewards = generate_uniform_reward(N, device)
    positions = positions.T
    distances = compute_distances(positions, p=2)

    return rewards, distances


# -- Sample costs
def sample_exponential_distribution(rate: torch.Tensor, num_samples: int):
    """Faster Exponential distribution w/ https://en.wikipedia.org/wiki/Inverse_transform_sampling.
    x = -(1/rate)*ln(y)
    """
    sample_shape = (*rate.shape, num_samples)
    y = torch.rand(sample_shape)
    samples = -(1 / rate).unsqueeze(-1) * torch.log(y)
    return samples


def sample_costs(weights: torch.Tensor, num_samples: int, kappa: float = 0.5):
    rate = 1 / ((1 - kappa) * weights)
    samples = sample_exponential_distribution(rate, num_samples)
    sampled_costs = (kappa * weights).unsqueeze(-1) + samples
    return sampled_costs


def create_pulp_instance(
    num_nodes: int,
    budget: int,
    num_samples: int,
    rewards: torch.Tensor,
    costs: torch.Tensor,
    M: int = 1000,
    kappa: float = 0.5,
    alpha_prime: float = 0.1,
) -> pulp.LpProblem:
    # Assume start node is 0 and end node is N - 1
    start = 0
    end = num_nodes - 1

    # Generate samples
    sampled_costs = sample_costs(costs, num_samples, kappa)  # Samples

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
    r = pulp.LpVariable.dicts("r", r_indices, lowBound=0, upBound=N - 1, cat="Integer")

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
    prob += pulp.lpSum(pi[j, end] for j in adj_list[end]) == 1

    # Add constraint (13)
    # Case 1: i = start
    outgoing = pulp.lpSum(pi[start, j] for j in adj_list[start])
    incoming = pulp.lpSum(pi[j, start] for j in adj_list[start])
    prob += outgoing - incoming == 1

    # Case 2: i = end
    outgoing = pulp.lpSum(pi[end, j] for j in adj_list[end])
    incoming = pulp.lpSum(pi[j, end] for j in adj_list[end])
    prob += outgoing - incoming == -1

    # Case 3: i not start or end
    for i in range(num_nodes):
        if i == start or i == end:
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
    prob += r[end] == num_nodes - 1

    # SAA Constraints (17-19)
    # Define variables:
    # 1. Auxiliary variable z^q (18)
    z = pulp.LpVariable.dicts("z", range(num_samples), cat="Binary")

    # Add constraint (17)
    for q in range(num_samples):
        prob += budget + budget * z[q] >= (
            pulp.lpSum(pi[i, j] * sampled_costs[i, j, q] for (i, j) in edges)
        )

    # Add constraint (19)
    prob += pulp.lpSum(z[q] for q in range(num_samples)) / num_samples <= alpha_prime

    return prob


def extract_solution(prob):
    edge_list = []
    for v in prob.variables():
        if v.varValue > 0:
            if v.name.startswith("pi"):
                edge_list.append(v.name)
    return edge_list


if __name__ == "__main__":
    # Input
    N = 20  # Num nodes
    Q = 100  # Num samples
    alpha_prime = 0.1  # Chance constraint
    H = 2  # Budget
    M = 1000  # Constant to ensure ranking
    kappa = 0.5

    R, T = generate_example_graph(N)  # Reward, Cost Matrix
    prob = create_pulp_instance(N, H, Q, R, T, M, kappa, alpha_prime)
    print("Created Problem")

    solver = pulp.getSolver("HiGHS")
    solver.timeLimit = 180

    result = prob.solve(solver)
    print(result)
    extract_solution(prob)
