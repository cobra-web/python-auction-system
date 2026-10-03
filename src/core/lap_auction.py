import warnings

import numpy as np


class AuctionLAP:
    """Synchronous (Jacobi) auction for the linear assignment problem, SS13 sign convention.

    beta(y) is a VALUE: the net cost of y for x is c(x, y) - beta(y), each source bids for the
    target with the smallest net cost, and the LOWEST bid wins (SS13 Eq. (7) to (9)). A bid lowers beta.

    Exactness: eps-CS gives cost <= OPT + N * eps. For costs on a grid c in g * Z, eps < g / N
    yields an optimal assignment (Bertsekas 1979, Prop. 1). SS13's "Delta c / |X|" is exactly this
    statement and only holds on a grid. For real-valued costs, only the additive bound N * eps holds.
    """

    def __init__(self, cost_matrix, epsilon=None):
        self.C = np.array(cost_matrix, dtype=float)
        if self.C.ndim != 2 or self.C.shape[0] != self.C.shape[1]:
            raise ValueError("AuctionLAP needs a square cost matrix.")
        self.N = self.C.shape[0]

        self.beta = np.zeros(self.N)
        self.x_to_y = np.full(self.N, -1)
        self.y_to_x = np.full(self.N, -1)

        if epsilon is None:
            unique_costs = np.unique(self.C)
            gaps = np.diff(unique_costs)
            gaps = gaps[gaps > 1e-12]
            if gaps.size:
                self.epsilon = float(gaps.min()) / (self.N + 1.0)
                if self.epsilon < 1e-8:
                    warnings.warn("Default epsilon is tiny because the cost values are not on a "
                                  "grid; pass epsilon explicitly (bound: OPT + N * epsilon).")
            else:
                self.epsilon = 1e-4
        else:
            self.epsilon = float(epsilon)

    def solve(self):
        iterations = 0
        if self.N == 1:
            self.x_to_y[0], self.y_to_x[0] = 0, 0
            return self.x_to_y, float(self.C[0, 0]), 0

        while np.any(self.x_to_y == -1):
            bids = self._bidding_phase()
            self._assignment_phase(bids)
            iterations += 1

        total_cost = float(sum(self.C[x, self.x_to_y[x]] for x in range(self.N)))
        return self.x_to_y, total_cost, iterations

    def _bidding_phase(self):
        bids = {}
        unassigned_x = np.where(self.x_to_y == -1)[0]

        for x in unassigned_x:
            eff_costs = self.C[x, :] - self.beta
            idx = np.argpartition(eff_costs, 1)
            y_star, y_second = idx[0], idx[1]
            if eff_costs[y_second] < eff_costs[y_star]:
                y_star, y_second = y_second, y_star

            alpha_prime = eff_costs[y_second]                          # SS13 Eq. (7)
            bid_value = self.C[x, y_star] - alpha_prime - self.epsilon  # SS13 Eq. (8)
            bids.setdefault(y_star, []).append((x, bid_value))

        return bids

    def _assignment_phase(self, bids):
        for y, y_bids in bids.items():
            best_x, min_bid = min(y_bids, key=lambda item: item[1])
            self.beta[y] = min_bid                                      # SS13 Eq. (9)

            old_x = self.y_to_x[y]
            if old_x != -1:
                self.x_to_y[old_x] = -1

            self.x_to_y[best_x] = y
            self.y_to_x[y] = best_x
