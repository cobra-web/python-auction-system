import numpy as np
import sys
from src.utils.data_structures import SparseNeighborhood

class AuctionOT:
    def __init__(self, cost_matrix, mu_X, mu_Y, epsilon=None, allowed_edges=None, initial_beta=None):
        # 1. Normalize costs to [0, 1] to prevent overflow
        self.C_raw = np.array(cost_matrix, dtype=float)
        self.max_c = np.max(self.C_raw)
        self.C = self.C_raw / (self.max_c if self.max_c > 0 else 1.0)
        
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)
        self.N_X, self.N_Y = self.C.shape
        
        self.epsilon = epsilon if epsilon is not None else 1e-3
        
        # Coupling matrix (mass assignments)
        self.mu = np.zeros((self.N_X, self.N_Y), dtype=float)
        
        # Single market price per destination
        if initial_beta is not None:
            self.beta = np.array(initial_beta, dtype=float)
        else:
            self.beta = np.zeros(self.N_Y, dtype=float)

        self.sparse = SparseNeighborhood(self.N_X, self.N_Y, allowed_edges) if allowed_edges else None

    def solve(self):
        iterations = 0
        # Iterate until all mass is assigned
        while iterations < 20000:
            assigned_X = np.sum(self.mu, axis=1)
            unassigned_X = self.mu_X - assigned_X
            
            # Use a robust tolerance
            if np.sum(unassigned_X[unassigned_X > 1e-6]) < 1e-6:
                break
            
            # Find most "urgent" unassigned X
            x = np.argmax(unassigned_X)
            
            # Compute bid values
            vals = self.C[x, :] - self.beta
            
            if self.sparse:
                targets = self.sparse.get_allowed_y(x)
            else:
                targets = range(self.N_Y)

            if not targets:
                # If X has no allowed targets, we are stuck
                iterations += 1
                continue

            # Find best and second best (standard auction logic)
            best_y, best_val = -1, float('inf')
            second_val = float('inf')
            
            for y in targets:
                v = vals[y]
                if v < best_val:
                    second_val = best_val
                    best_val = v
                    best_y = y
                elif v < second_val:
                    second_val = v
            
            # Update Price: Beta must adjust by the gap between best and 2nd best
            # This 'price increment' pushes the algorithm toward optimality
            price_inc = max(self.epsilon, (second_val - best_val) + 1e-9)
            self.beta[best_y] -= price_inc 
            
            # Displacement: Transfer mass from current owners to X
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

        if np.sum(unassigned_X[unassigned_X > 1e-6]) >= 1e-6:
            raise RuntimeError(f"Did not converge after {iterations} iterations.")

        return self.mu, np.sum(self.mu * self.C_raw), iterations
