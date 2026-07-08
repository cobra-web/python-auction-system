import numpy as np
import sys
from src.utils.data_structures import SparseNeighborhood


class AuctionOT:
    def __init__(self, cost_matrix, mu_X, mu_Y, epsilon=None, allowed_edges=None, initial_beta=None):

        self.C_raw = np.array(cost_matrix, dtype=float)
        self.max_c = np.max(self.C_raw)
        # Normalize costs to [0, 1] range for numerical stability
        self.C = self.C_raw / (self.max_c if self.max_c > 0 else 1.0)
        
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)
        self.N_X, self.N_Y = self.C.shape
        
        # ε must remain constant during solve()
        if epsilon is None:
            epsilon = 1e-3
        self.epsilon = epsilon
        
        # Coupling matrix: mu[i,j] = amount transported from source i to sink j
        self.mu = np.zeros((self.N_X, self.N_Y), dtype=float)
        
        if initial_beta is not None:
            self.beta = np.array(initial_beta, dtype=float)
        else:
            self.beta = np.zeros(self.N_Y, dtype=float)
        
        # Sparse neighborhood structure (for hierarchical methods)
        if allowed_edges is not None:
            self.sparse = SparseNeighborhood(self.N_X, self.N_Y, allowed_edges)
        else:
            self.sparse = None

    def solve(self):
        iterations = 0
        max_iterations = 100000
        
        while iterations < max_iterations:
            # Compute unassigned mass for each source
            assigned_X = np.sum(self.mu, axis=1)
            unassigned_X = self.mu_X - assigned_X
            
            # Check convergence: all mass assigned (up to numerical tolerance)
            total_unassigned = np.sum(unassigned_X)
            if total_unassigned < 1e-9:
                print(f"  Auction converged after {iterations} iterations")
                break
            
            # === BIDDING PHASE ===
            # Pick an unassigned source with the most mass to bid
            x = np.argmax(unassigned_X)
            mass_to_assign = unassigned_X[x]
            
            if mass_to_assign < 1e-9:
                iterations += 1
                continue
            
            # Get all admissible targets for this source
            if self.sparse is not None:
                valid_y_indices = self.sparse.get_allowed_y(x)
            else:
                valid_y_indices = list(range(self.N_Y))
            
            if len(valid_y_indices) == 0:
                print(f"  WARNING: Source {x} has no admissible targets!")
                break
            
            # Compute reduced costs: C[x, y] - beta[y]
            # We want MINIMUM reduced cost (best value)
            best_y = None
            second_y = None
            best_val = np.inf
            second_val = np.inf
            
            for y in valid_y_indices:
                reduced_cost = self.C[x, y] - self.beta[y]
                
                if reduced_cost < best_val:
                    # Shift best → second best, then update best
                    second_val = best_val
                    second_y = best_y
                    best_val = reduced_cost
                    best_y = y
                elif reduced_cost < second_val:
                    # Update second best only
                    second_val = reduced_cost
                    second_y = y
            
            # Safety check: if we couldn't find second best, set it to best
            # (happens when there's only one admissible target)
            if second_y is None:
                second_val = best_val
            
            price_increment = max(self.epsilon, (second_val - best_val) + 1e-12)
            self.beta[best_y] -= price_increment
            
            # Calculate available space at best_y
            current_supply_at_y = np.sum(self.mu[:, best_y])
            available_space_at_y = self.mu_Y[best_y] - current_supply_at_y
            
            # Determine how much mass we can actually assign to best_y
            mass_to_assign_to_y = min(mass_to_assign, available_space_at_y)
            
            if mass_to_assign_to_y < 1e-9:
                
                # Collect all sources holding best_y
                holders = []
                for x_prime in range(self.N_X):
                    if self.mu[x_prime, best_y] > 1e-9:
                        holders.append(x_prime)
                
                # Displace greedily: take from whoever has the most mass there
                for x_prime in holders:
                    mass_at_y = self.mu[x_prime, best_y]
                    can_displace = min(mass_to_assign, mass_at_y)
                    
                    self.mu[x_prime, best_y] -= can_displace
                    self.mu[x, best_y] += can_displace
                    mass_to_assign -= can_displace
                    
                    if mass_to_assign < 1e-9:
                        break
            else:
                # There is available space; assign what we can
                self.mu[x, best_y] += mass_to_assign_to_y
                mass_to_assign -= mass_to_assign_to_y
                
                # If x still has unassigned mass, we need to displace to make room
                if mass_to_assign > 1e-9:
                    holders = []
                    for x_prime in range(self.N_X):
                        if self.mu[x_prime, best_y] > 1e-9:
                            holders.append(x_prime)
                    
                    for x_prime in holders:
                        mass_at_y = self.mu[x_prime, best_y]
                        can_displace = min(mass_to_assign, mass_at_y)
                        
                        self.mu[x_prime, best_y] -= can_displace
                        self.mu[x, best_y] += can_displace
                        mass_to_assign -= can_displace
                        
                        if mass_to_assign < 1e-9:
                            break
            
            iterations += 1
            if iterations % 10000 == 0:
                print(f"  Iteration {iterations}: unassigned mass = {total_unassigned:.6f}", file=sys.stderr)
        
        if iterations >= max_iterations:
            print(f"  WARNING: Reached max iterations ({max_iterations})")
        
        # Compute final cost in original units
        total_cost = np.sum(self.mu * self.C_raw)
        
        return self.mu, total_cost, iterations
