import numpy as np
import sys
from src.utils.data_structures import SparseNeighborhood

class AuctionOT:
    def __init__(self, cost_matrix, mu_X, mu_Y, epsilon=None, allowed_edges=None, initial_beta=None):
        # 1. SCALE COSTS: Normalizing to [0, 1] is MANDATORY for auction stability
        self.C_raw = np.array(cost_matrix, dtype=float)
        self.max_c = np.max(self.C_raw)
        self.C = self.C_raw / (self.max_c if self.max_c > 0 else 1.0)
        
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)
        self.N_X, self.N_Y = self.C.shape
        
        self.epsilon = epsilon if epsilon is not None else 1e-3
        self.mu = np.zeros((self.N_X, self.N_Y), dtype=float)
        self.beta = np.array(initial_beta, dtype=float) if initial_beta is not None else np.zeros(self.N_Y, dtype=float)
        self.sparse = SparseNeighborhood(self.N_X, self.N_Y, allowed_edges) if allowed_edges else None

    def solve(self):
        iterations = 0
        while iterations < 20000:
            assigned_X = np.sum(self.mu, axis=1)
            unassigned_X = self.mu_X - assigned_X
            
            # Use a wider tolerance for mass check
            if np.sum(unassigned_X) < 1e-4:
                break
            
            # Find most "urgent" unassigned X
            x = np.argmax(unassigned_X)
            
            # Compute values for bidding
            vals = self.C[x, :] - self.beta
            
            # Only consider sparse neighborhood if it exists
            if self.sparse:
                allowed = self.sparse.get_allowed_y(x)
                if not allowed: # Fallback to dense if sparse is empty
                    targets = range(self.N_Y)
                else:
                    targets = allowed
            else:
                targets = range(self.N_Y)

            # Find best and second best
            best_y, best_val = -1, float('inf')
            second_val = float('inf')
            
            for y in targets:
                v = vals[y]
                if v < best_val:
                    second_val, best_val = best_val, v
                    best_y = y
                elif v < second_val:
                    second_val = v
            
            if best_y == -1: 
                # If no targets, the problem is infeasible or edges are too restricted
                iterations += 1
                continue

            # Update Price (Beta)
            # Ensure price increment is well-defined
            price_inc = max(self.epsilon, (second_val - best_val) + 1e-9)
            self.beta[best_y] -= price_inc 
            
            # Displacement
            for x_prime in range(self.N_X):
                if self.mu[x_prime, best_y] > 1e-8:
                    transfer = min(unassigned_X[x], self.mu[x_prime, best_y])
                    self.mu[x_prime, best_y] -= transfer
                    self.mu[x, best_y] += transfer
                    unassigned_X[x] -= transfer
                    if unassigned_X[x] <= 1e-8: break
            
            iterations += 1
            if iterations % 2000 == 0:
                print(f"  Auction progress: {np.sum(unassigned_X):.4f} mass left", file=sys.stderr)

        if np.sum(unassigned_X) >= 1e-4:
            raise RuntimeError("Did not converge")

        # Rescale cost for final return
        total_cost = np.sum(self.mu * self.C_raw)
        return self.mu, total_cost, iterations
