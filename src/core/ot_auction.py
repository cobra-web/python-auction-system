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

    def _calculate_available_capacity(self, y, x_current):
        # Total mass currently held by y (from all sources)
        current_total_at_y = np.sum(self.mu[:, y])
        
        # Subtract mass that x_current already has at y
        # (because we can only fill up to total demand, not double-count)
        mass_from_x_current = self.mu[x_current, y]
        current_from_others = current_total_at_y - mass_from_x_current
        
        # Available = total demand - mass from others
        available = self.mu_Y[y] - current_from_others
        
        return max(0.0, available)

    def _assign_mass_to_best_target(self, x, best_y, mass_available):
        available_space = self._calculate_available_capacity(best_y, x)
        
        if available_space >= mass_available - 1e-9:
            # Case 1: Enough space for all residual mass
            self.mu[x, best_y] += mass_available
            return mass_available
        elif available_space > 1e-9:
            # Case 2: Partial space available
            self.mu[x, best_y] += available_space
            return available_space
        else:
            # Case 3: No space available; must displace other sources
            # CRITICAL: Only displace OTHER sources (x' != x)
            mass_to_displace = mass_available
            
            # Collect holders (excluding x itself)
            holders = []
            for x_prime in range(self.N_X):
                if x_prime != x and self.mu[x_prime, best_y] > 1e-9:
                    holders.append(x_prime)
            
            if len(holders) == 0:
                # No one else to displace; just fail gracefully
                # This shouldn't happen in balanced problems
                print(f"  WARNING: Source {x} cannot access sink {best_y} "
                      f"(full and no other holders)", file=sys.stderr)
                return 0.0
            
            # Displace from holders (take from whoever has most mass there)
            displaced_amount = 0.0
            for x_prime in sorted(holders, key=lambda xp: self.mu[xp, best_y], reverse=True):
                mass_at_y = self.mu[x_prime, best_y]
                can_displace = min(mass_to_displace, mass_at_y)
                
                self.mu[x_prime, best_y] -= can_displace
                self.mu[x, best_y] += can_displace
                mass_to_displace -= can_displace
                displaced_amount += can_displace
                
                if mass_to_displace < 1e-9:
                    break
            
            return displaced_amount

    def solve(self):
        iterations = 0
        max_iterations = 100000
        
        while iterations < max_iterations:
            # Compute unassigned mass for each source
            assigned_X = np.sum(self.mu, axis=1)
            unassigned_X = self.mu_X - assigned_X
            
            # Check convergence
            total_unassigned = np.sum(unassigned_X)
            if total_unassigned < 1e-9:
                print(f"  Auction converged after {iterations} iterations")
                break
            
            # Pick source with most unassigned mass
            x = np.argmax(unassigned_X)
            mass_to_assign = unassigned_X[x]
            
            if mass_to_assign < 1e-9:
                iterations += 1
                continue
            
            # Get admissible targets
            if self.sparse is not None:
                valid_y_indices = self.sparse.get_allowed_y(x)
            else:
                valid_y_indices = list(range(self.N_Y))
            
            if len(valid_y_indices) == 0:
                print(f"  ERROR: Source {x} has no admissible targets!", file=sys.stderr)
                break
            
            # === FIND BEST AND SECOND-BEST TARGETS ===
            best_y, second_y, best_val, second_val = self._get_best_and_second_best_targets(x, valid_y_indices)
            
            if best_y is None:
                print(f"  ERROR: Could not find best target for source {x}", file=sys.stderr)
                break
            
            # === UPDATE PRICE OF BEST TARGET (ε-CS) ===
            price_increment = max(self.epsilon, (second_val - best_val) + 1e-12)
            self.beta[best_y] -= price_increment
            
            # === ASSIGN MASS TO BEST TARGET ===
            mass_assigned_to_best = self._assign_mass_to_best_target(x, best_y, mass_to_assign)
            remaining_mass = mass_to_assign - mass_assigned_to_best
            
            # === HANDLE RESIDUAL MASS ===
            # If x still has mass after filling best_y, it must try 2nd-best target
            # This prevents infinite loops of x re-bidding for best_y
            if remaining_mass > 1e-9 and second_y is not None:
                # Only update price if we're actually assigning to second target
                # (don't double-update prices)
                second_val_current = self.C[x, second_y] - self.beta[second_y]
                
                # Find the third-best target (for computing price of second_y)
                third_val = np.inf
                for y in valid_y_indices:
                    if y != best_y and y != second_y:
                        rc = self.C[x, y] - self.beta[y]
                        if rc < third_val:
                            third_val = rc
                
                # If third_val not found, use second_val as fallback
                if third_val == np.inf:
                    third_val = second_val_current
                
                # Update price of second target (ε-CS)
                price_increment_second = max(self.epsilon, (third_val - second_val_current) + 1e-12)
                self.beta[second_y] -= price_increment_second
                
                # Assign residual mass to second target
                mass_assigned_to_second = self._assign_mass_to_best_target(x, second_y, remaining_mass)
                remaining_mass -= mass_assigned_to_second
            
            # === FINAL RESIDUAL CHECK ===
            # If there's still unassigned mass and we've exhausted top-2 targets,
            # find next-best and repeat (or force re-bidding on next iteration)
            if remaining_mass > 1e-9:
                # Find all targets not yet tried, sorted by reduced cost
                tried_targets = {best_y}
                if second_y is not None:
                    tried_targets.add(second_y)
                
                remaining_targets = [y for y in valid_y_indices if y not in tried_targets]
                if len(remaining_targets) > 0:
                    remaining_targets_sorted = sorted(
                        remaining_targets, 
                        key=lambda y: self.C[x, y] - self.beta[y]
                    )
                    # Try to assign to the next-best target
                    for y_next in remaining_targets_sorted:
                        if remaining_mass < 1e-9:
                            break
                        mass_assigned = self._assign_mass_to_best_target(x, y_next, remaining_mass)
                        remaining_mass -= mass_assigned
                # If we still have residual mass and no more targets, it will force
                # x to re-bid on next iteration (which is correct behavior for unbalanced)
            
            iterations += 1
            if iterations % 10000 == 0:
                print(f"  Iteration {iterations}: unassigned mass = {total_unassigned:.9f}", 
                      file=sys.stderr)
        
        if iterations >= max_iterations:
            print(f"  WARNING: Reached max iterations ({max_iterations})", file=sys.stderr)
            assigned_X = np.sum(self.mu, axis=1)
            unassigned_X = self.mu_X - assigned_X
            total_unassigned = np.sum(unassigned_X)
            print(f"  Final unassigned mass: {total_unassigned:.9f}", file=sys.stderr)
        
        # Compute final cost in original units
        total_cost = np.sum(self.mu * self.C_raw)
        
        return self.mu, total_cost, iterations
