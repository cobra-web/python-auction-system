import numpy as np
import sys
from src.utils.data_structures import SparseNeighborhood

class AuctionOT:
    def __init__(self, cost_matrix, mu_X, mu_Y, epsilon=None, allowed_edges=None, initial_beta=None):
        self.C = np.array(cost_matrix, dtype=float)
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)
        self.N_X, self.N_Y = self.C.shape

        # Set Epsilon
        if epsilon is None:
            self.epsilon = 1e-3 # Default conservative start
        else:
            self.epsilon = epsilon

        # Coupling matrix (mass assignments)
        self.mu = np.zeros((self.N_X, self.N_Y), dtype=float)
        
        # Prices (Beta): The market price for each Y.
        # Initializing correctly is key to a warm start.
        if initial_beta is not None:
            self.beta = np.array(initial_beta, dtype=float)
        else:
            self.beta = np.zeros(self.N_Y, dtype=float)

        self.sparse = SparseNeighborhood(self.N_X, self.N_Y, allowed_edges) if allowed_edges else None

    def solve(self):
        iterations = 0
        # Main Auction Loop
        while True:
            # Check for unassigned mass
            assigned_X = np.sum(self.mu, axis=1)
            unassigned_X = self.mu_X - assigned_X

            # If all mass is assigned (within tolerance), convergence!
            if np.sum(unassigned_X[unassigned_X > 1e-8]) < 1e-8:
                break

            bids = self._bidding_phase(unassigned_X)
            self._assignment_phase(bids)
            
            iterations += 1
            if iterations % 500 == 0:
                print(f"iter {iterations}: unassigned={np.sum(unassigned_X[unassigned_X > 1e-8]):.2f}", file=sys.stderr)

            if iterations > 20000:
                raise RuntimeError(f"AuctionOT did not converge after {iterations} iterations.")

        total_cost = np.sum(self.mu * self.C)
        return self.mu, total_cost, iterations

    def _bidding_phase(self, unassigned_X):
        # Maps y -> list of bids
        bids = {y: [] for y in range(self.N_Y)}

        for x in range(self.N_X):
            mass = unassigned_X[x]
            if mass <= 1e-8: continue

            # Get target indices
            targets = self.sparse.get_allowed_y(x) if self.sparse else range(self.N_Y)
            
            # Find Best and Second-Best (for bid increment)
            best_y, second_y = -1, -1
            best_val, second_val = float('inf'), float('inf')

            for y in targets:
                # Value = Cost - Price
                val = self.C[x, y] - self.beta[y]
                if val < best_val:
                    second_val, second_y = best_val, best_y
                    best_val, best_y = val, y
                elif val < second_val:
                    second_val, second_y = val, y

            if best_y == -1: continue

            # Calculate price increment: beta must move by (cost_diff + eps)
            # This forces convergence
            inc = (second_val - best_val) if second_val != float('inf') else 0.0
            new_beta = self.beta[best_y] + inc + self.epsilon
            
            bids[best_y].append({'x': x, 'mass': mass, 'new_beta': new_beta})

        return bids

    def _assignment_phase(self, bids):
        for y, y_bids in bids.items():
            if not y_bids: continue

            # 1. Pool current owners + new bidders
            pool = []
            for x in range(self.N_X):
                if self.mu[x, y] > 1e-8:
                    pool.append({'x': x, 'mass': self.mu[x, y], 'bid_beta': self.beta[y]})
            
            for b in y_bids:
                pool.append({'x': b['x'], 'mass': b['mass'], 'bid_beta': b['new_beta']})

            # 2. Sort by price (highest price first, i.e., most competitive)
            pool.sort(key=lambda b: b['bid_beta'], reverse=True)

            # 3. Re-assign mass
            self.mu[:, y] = 0.0
            remaining_cap = self.mu_Y[y]
            
            # The new price for this Y bin is the price of the last accepted bid
            new_y_beta = self.beta[y] 

            for item in pool:
                if remaining_cap <= 1e-8: break
                
                grant = min(item['mass'], remaining_cap)
                self.mu[item['x'], y] += grant
                remaining_cap -= grant
                new_y_beta = item['bid_beta']

            self.beta[y] = new_y_beta
