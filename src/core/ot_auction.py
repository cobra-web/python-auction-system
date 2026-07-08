import numpy as np
import sys
from src.utils.data_structures import SparseNeighborhood

class AuctionOT:
    def __init__(self, cost_matrix, mu_X, mu_Y, epsilon=None, allowed_edges=None, initial_beta=None):
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
        last_unassigned = float('inf')
        
        while iterations < 20000:
            unassigned_X = self.mu_X - np.sum(self.mu, axis=1)
            current_unassigned = np.sum(unassigned_X[unassigned_X > 1e-6])
            
            if current_unassigned < 1e-6:
                break
            
            # STALL DETECTION: If progress stops, relax the epsilon
            if abs(current_unassigned - last_unassigned) < 1e-8:
                self.epsilon *= 1.1 # Nudge the price harder to force movement
            last_unassigned = current_unassigned
            
            x = np.argmax(unassigned_X)
            
            # Calculate utility for all Y
            vals = self.C[x, :] - self.beta
            
            # Find best and second best
            best_y = np.argmin(vals)
            best_val = vals[best_y]
            vals[best_y] = np.inf
            second_val = np.min(vals)
            
            # Core SS13 Dual Update (Displacement via Price)
            price_inc = max(self.epsilon, (second_val - best_val) + 1e-12)
            self.beta[best_y] -= price_inc 
            
            # Displacement Phase: Force mass movement
            # We must displace mass from ALL X' currently holding Y
            for x_prime in range(self.N_X):
                if self.mu[x_prime, best_y] > 1e-9:
                    # How much can we displace?
                    can_take = min(unassigned_X[x], self.mu[x_prime, best_y])
                    self.mu[x_prime, best_y] -= can_take
                    self.mu[x, best_y] += can_take
                    unassigned_X[x] -= can_take
                    if unassigned_X[x] <= 1e-9: break
            
            iterations += 1
            if iterations % 5000 == 0:
                print(f"  Auction progress: {current_unassigned:.4f} mass left", file=sys.stderr)

        return self.mu, np.sum(self.mu * self.C_raw), iterations
