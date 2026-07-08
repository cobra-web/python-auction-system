import numpy as np
import sys
from src.utils.data_structures import SparseNeighborhood

class AuctionOT:
    def __init__(self, cost_matrix, mu_X, mu_Y, epsilon=None, allowed_edges=None, initial_beta=None):
        self.C = np.array(cost_matrix, dtype=float)
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)
        self.N_X, self.N_Y = self.C.shape
        
        # Fundamental SS13 Theory: Epsilon must be large enough to allow displacement
        self.epsilon = epsilon if epsilon is not None else 1e-2 
        
        self.mu = np.zeros((self.N_X, self.N_Y), dtype=float)
        self.beta = np.array(initial_beta, dtype=float) if initial_beta is not None else np.zeros(self.N_Y, dtype=float)
        self.sparse = SparseNeighborhood(self.N_X, self.N_Y, allowed_edges) if allowed_edges else None

    def solve(self):
        iterations = 0
        # The key to convergence: while there is mass, force bids
        while True:
            unassigned_X = self.mu_X - np.sum(self.mu, axis=1)
            if np.sum(unassigned_X) < 1e-6:
                break
            
            # Find the unassigned X that has the most to gain
            x = np.argmax(unassigned_X)
            
            # Find Best and Second-Best to compute valid bid
            vals = self.C[x, :] - self.beta
            best_y = np.argmin(vals)
            best_val = vals[best_y]
            
            # Mask best to find second best
            vals[best_y] = np.inf
            second_val = np.min(vals)
            
            # Bid increment: This is the core SS13 dual update
            price_inc = max(self.epsilon, second_val - best_val + 1e-9)
            self.beta[best_y] -= price_inc # Lower price to attract mass
            
            # Assignment (Displacement)
            # Find all x' that currently own this y
            for x_prime in range(self.N_X):
                if self.mu[x_prime, best_y] > 0:
                    transfer = min(unassigned_X[x], self.mu[x_prime, best_y])
                    self.mu[x_prime, best_y] -= transfer
                    self.mu[x, best_y] += transfer
                    unassigned_X[x] -= transfer
                    if unassigned_X[x] <= 0: break
            
            iterations += 1
            if iterations > 100000: # Increased limit
                raise RuntimeError("Did not converge")
        return self.mu, np.sum(self.mu * self.C), iterations
