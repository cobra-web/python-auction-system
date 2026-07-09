import numpy as np
import sys
import math
from src.utils.data_structures import SparseNeighborhood


# ===== GLOBAL TOLERANCE PARAMETER =====
# Any mass amount below this is treated as zero (no bidding, no displacement)
TOL = 1e-7  # 0.1 micrograms if mass is in grams


class AuctionOT:
    
    def __init__(self, cost_matrix, mu_X, mu_Y, epsilon=None, allowed_edges=None, initial_beta=None):
        self.C_raw = np.array(cost_matrix, dtype=float)
        self.max_c = np.max(self.C_raw)
        self.C = self.C_raw / (self.max_c if self.max_c > 0 else 1.0)
        
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)
        self.N_X, self.N_Y = self.C.shape
        
        if epsilon is None:
            epsilon = 1e-3
        self.epsilon = epsilon
        
        self.mu = np.zeros((self.N_X, self.N_Y), dtype=float)
        
        if initial_beta is not None:
            self.beta = np.array(initial_beta, dtype=float)
        else:
            self.beta = np.zeros(self.N_Y, dtype=float)
        
        if allowed_edges is not None:
            self.sparse = SparseNeighborhood(self.N_X, self.N_Y, allowed_edges)
        else:
            self.sparse = None

    def _get_best_and_second_best_targets(self, x, valid_y_indices):
        best_y = None
        second_y = None
        best_val = np.inf
        second_val = np.inf
        
        for y in valid_y_indices:
            reduced_cost = self.C[x, y] - self.beta[y]
            
            if reduced_cost < best_val:
                second_val = best_val
                second_y = best_y
                best_val = reduced_cost
                best_y = y
            elif reduced_cost < second_val:
                second_val = reduced_cost
                second_y = y
        
        # If only one admissible target, second best is same as best
        if second_y is None:
            second_val = best_val
        
        return best_y, second_y, best_val, second_val

    def _get_current_sink_load(self, y):
        return np.sum(self.mu[:, y])

    def _calculate_available_capacity(self, y):
        current_load = self._get_current_sink_load(y)
        available = self.mu_Y[y] - current_load
        
        # Apply float tolerance: round to zero if below TOL
        if available < TOL:
            return 0.0
        
        return available

    def _displace_mass_from_others(self, x, y, amount_to_displace, TOL=1e-7):
        # NOTE: mass is tracked solely via self.mu. solve() recomputes
        # unassigned mass fresh each iteration as self.mu_X - sum(self.mu, axis=1),
        # so reducing self.mu[x_prime, y] here is itself the "refund" —
        # x_prime shows up as unassigned again on the next loop pass.
        actual_displaced = 0.0
        
        # 3. Strict Displacement Loop over all other owners
        for x_prime in range(self.mu.shape[0]):
            if x_prime == x:
                continue # Mathematical rule: A node must NEVER displace itself
                
            if self.mu[x_prime, y] > TOL:
                # Take only what we need, capped by what x_prime actually owns
                displaced = min(self.mu[x_prime, y], amount_to_displace)
                
                # 4. STRICT BANK TRANSACTION (Conservation of Mass)
                self.mu[x_prime, y] -= displaced
                
                amount_to_displace -= displaced
                actual_displaced += displaced
                
                # 5. Halt if we have successfully displaced all the mass we needed
                if amount_to_displace <= TOL:
                    break
                    
        return actual_displaced

    
    def _assign_mass_to_target(self, x, y, amount):
        TOL = 1e-7 # Strict tolerance to prevent float micro-bidding deadlocks
        
        if amount < TOL:
            return 0.0
            
        # 1. Calculate free space strictly respecting the capacity limit mu_Y
        current_mass_at_y = np.sum(self.mu[:, y])
        space_remaining = max(0.0, self.mu_Y[y] - current_mass_at_y)
        
        # 2. Determine how much we can just place in empty space vs. what needs displacement
        if amount <= space_remaining:
            # Simplest case: y has enough empty space to hold all of x's bid
            self.mu[x, y] += amount
            return amount
        else:
            # y is full (or will be). We fill the empty space, then displace the rest
            amount_to_displace = amount - space_remaining
            
            # Call the strict displacement loop
            actual_displaced = self._displace_mass_from_others(x, y, amount_to_displace, TOL)
            
            # Calculate the total amount x actually secured
            actual_assigned = space_remaining + actual_displaced
            
            # Finalize x's assignment strictly
            self.mu[x, y] += actual_assigned
            return actual_assigned

    
    def solve(self):
        iterations = 0
        max_iterations = 2000000
        
        # Track stagnation to break the plateau
        last_total_unassigned = float('inf')
        stagnation_counter = 0
        
        while iterations < max_iterations:
            assigned_X = np.sum(self.mu, axis=1)
            unassigned_X = self.mu_X - assigned_X
            total_unassigned = np.sum(unassigned_X)
            
            # Convergence check
            if total_unassigned < TOL:
                break

            # STAGNATION DETECTOR: If mass doesn't change, boost epsilon!
            if abs(total_unassigned - last_total_unassigned) < 1e-9:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_total_unassigned = total_unassigned
                
            if stagnation_counter > 50000:
                self.epsilon *= 1.5  # Aggressive boost to break the cycle
                stagnation_counter = 0
                print(f"  [!] Stagnation detected at iter {iterations}. Boosting epsilon to {self.epsilon:.6f}")

            # ... (rest of your auction logic remains exactly the same) ...
