import numpy as np
import sys
from src.utils.data_structures import SparseNeighborhood

class AuctionOT:
    def __init__(self, cost_matrix, mu_X, mu_Y, epsilon=None, allowed_edges=None, initial_beta=None):
        self.C = np.array(cost_matrix, dtype=float)
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)
        self.N_X, self.N_Y = self.C.shape

        if epsilon is None:
            self.epsilon = 1.0 / (np.sum(self.mu_X) + 1.0)
        else:
            self.epsilon = epsilon

        # Track mass assignments cleanly
        self.mu = np.zeros((self.N_X, self.N_Y), dtype=float)
        
        # Track a SINGLE uniform price per destination
        if initial_beta is not None:
            self.beta = np.copy(initial_beta)
        else:
            self.beta = np.zeros(self.N_Y, dtype=float)

        if allowed_edges is not None:
            self.sparse = SparseNeighborhood(self.N_X, self.N_Y, allowed_edges)
        else:
            self.sparse = None

    def solve(self):
        iterations = 0
        while True:
            assigned_X = np.sum(self.mu, axis=1)
            unassigned_X = self.mu_X - assigned_X

            # Break if all mass is assigned (with float tolerance)
            if np.sum(unassigned_X[unassigned_X > 1e-8]) == 0:
                break

            bids = self._bidding_phase(unassigned_X)
            self._assignment_phase(bids)
            iterations += 1
            
            if iterations <= 50 or iterations % 500 == 0:
                unassigned_val = np.sum(unassigned_X[unassigned_X > 1e-8])
                print(f"iter {iterations}: unassigned={unassigned_val:.2f}", file=sys.stderr)

            if iterations > 20000:
                raise RuntimeError(f"AuctionOT did not converge after {iterations} iterations.")

        total_cost = np.sum(self.mu * self.C)
        return self.mu, total_cost, iterations

    def _bidding_phase(self, unassigned_X):
        bids = {y: [] for y in range(self.N_Y)}

        for x in range(self.N_X):
            mass_to_assign = unassigned_X[x]
            if mass_to_assign <= 1e-8:
                continue

            if self.sparse is not None:
                valid_y = self.sparse.get_allowed_y(x)
            else:
                valid_y = range(self.N_Y)

            if len(valid_y) == 0:
                continue

            # Find best and second best objects
            best_y = -1
            best_val = float('inf')
            second_val = float('inf')

            for y in valid_y:
                val = self.C[x, y] - self.beta[y]
                if val < best_val:
                    second_val = best_val
                    best_val = val
                    best_y = y
                elif val < second_val:
                    second_val = val

            if best_y == -1:
                continue

            # Fallback if there is only 1 valid target
            if second_val == float('inf'):
                second_val = best_val

            # Bid computation: drives beta downwards (making it more negative)
            bid_beta = self.C[x, best_y] - second_val - self.epsilon
            bids[best_y].append({'x': x, 'mass': mass_to_assign, 'bid_beta': bid_beta})

        return bids

    def _assignment_phase(self, bids):
        for y, y_bids in bids.items():
            if not y_bids:
                continue

            # Pool current owners (defending their mass at the current price) 
            # with the new incoming bids.
            pool = []
            for x in range(self.N_X):
                if self.mu[x, y] > 1e-8:
                    pool.append({'x': x, 'mass': self.mu[x, y], 'bid_beta': self.beta[y]})

            pool.extend(y_bids)

            # Sort ascending by beta (most negative/competitive bid wins)
            pool.sort(key=lambda b: b['bid_beta'])

            # Wipe the bin clean and refill it with the best bids
            self.mu[:, y] = 0.0
            remaining_capacity = self.mu_Y[y]
            worst_accepted_beta = self.beta[y]

            for bid in pool:
                if remaining_capacity <= 1e-8:
                    break
                grant = min(bid['mass'], remaining_capacity)
                if grant > 1e-8:
                    self.mu[bid['x'], y] += grant
                    remaining_capacity -= grant
                    worst_accepted_beta = bid['bid_beta']

            # Update the uniform market price for this destination
            self.beta[y] = worst_accepted_beta
