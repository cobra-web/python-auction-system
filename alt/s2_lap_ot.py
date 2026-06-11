import numpy as np
import ot
from scipy.optimize import linear_sum_assignment

def solve_lap(cost):
    row, col = linear_sum_assignment(cost)
    return col, float(cost[row, col].sum())

def solve_ot(cost, mu_x, mu_y):
    a = np.array(mu_x, dtype=float); a /= a.sum()
    b = np.array(mu_y, dtype=float); b /= b.sum()
    coupling = ot.emd(a, b, cost)
    return coupling, float(np.sum(cost * coupling))


if __name__ == "__main__":

    cost_lap = np.array([[4., 1., 3.], [2., 0., 5.], [3., 2., 2.]])
    assignment, total = solve_lap(cost_lap)
    print("LAP assignment:", assignment)  
    print("LAP cost:", total)             

    positions = np.array([0., 1., 2., 3.])
    mu_x = np.array([0.5, 0.3, 0.1, 0.1])
    mu_y = np.array([0.1, 0.1, 0.3, 0.5])
    cost_ot = (positions[:, None] - positions[None, :]) ** 2
    coupling, total_ot = solve_ot(cost_ot, mu_x, mu_y)
    print("\nOT coupling:\n", np.round(coupling, 3))
    print("OT cost:", round(total_ot, 4))

    mu_u = np.ones(3) / 3
    coup_u, cost_u = solve_ot(cost_lap, mu_u, mu_u)
    print("\nOT with uniform masses (= LAP):\n", np.round(coup_u * 3))
    print("Cost:", round(cost_u * 3, 4))