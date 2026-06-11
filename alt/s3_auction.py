import numpy as np
import ot
from scipy.optimize import linear_sum_assignment

def auction_lap(cost, eps=None):
    N = cost.shape[0]
    if eps is None:
        vals = np.unique(cost[np.isfinite(cost)])
        delta_c = np.diff(vals).min() if len(vals) > 1 else 1.0
        eps = delta_c / (N + 1)   # must satisfy eps < delta_c / N for optimality

    assignment  = np.full(N, -1, dtype=int)
    assigned_to = np.full(N, -1, dtype=int)
    beta        = np.zeros(N, dtype=float)
    unassigned  = set(range(N))

    while unassigned:
        bids = {}
        for x in unassigned:
            reduced = cost[x] - beta          # c(x,y) - beta(y) for all y
            idx     = np.argsort(reduced)
            y_star  = idx[0]
            alpha_prime = reduced[idx[1]]     # Eq. (7): second-best reduced cost
            bid = cost[x, y_star] - alpha_prime - eps   # Eq. (8)
            if y_star not in bids or bid < bids[y_star][0]:
                bids[y_star] = (bid, x)

        for y, (bid, x_star) in bids.items():
            beta[y] = bid                     # Eq. (9): update price
            x_old = assigned_to[y]
            if x_old != -1:
                assignment[x_old] = -1
                unassigned.add(x_old)
            assignment[x_star] = y
            assigned_to[y]     = x_star
            unassigned.discard(x_star)

    total = float(sum(cost[x, assignment[x]] for x in range(N)))
    return assignment, total, beta


def solve_ot(cost, mu_x, mu_y):
    a = np.array(mu_x, dtype=float); a /= a.sum()
    b = np.array(mu_y, dtype=float); b /= b.sum()
    coupling = ot.emd(a, b, cost)
    return coupling, float(np.sum(cost * coupling))

if __name__ == "__main__":
    cost = np.array([[4., 1., 3.], [2., 0., 5.], [3., 2., 2.]])

    assignment, total, beta = auction_lap(cost)
    print("Auction assignment:", assignment)   # [1 0 2]
    print("Auction cost:", total)              # 5.0
    print("Beta (prices):", np.round(beta, 4))

    row, col = linear_sum_assignment(cost)
    print("Scipy reference:", col, "cost:", cost[row, col].sum())

    rng = np.random.default_rng(42)
    N = 8
    px = rng.random((N, 2))
    py = rng.random((N, 2))
    cost_ot = np.sum((px[:, None] - py[None, :]) ** 2, axis=2)
    mu_x = np.array([0.4, 0.3, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03])
    mu_y = np.ones(N) / N
    coupling, ot_cost = solve_ot(cost_ot, mu_x, mu_y)
    print("\nOT coupling (non-uniform masses):\n", np.round(coupling, 3))
    print("OT cost:", round(ot_cost, 4))