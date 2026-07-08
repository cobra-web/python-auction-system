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

    def _get_current_sink_load(self, y):
        return np.sum(self.mu[:, y])

    def _calculate_available_capacity(self, y):
        current_load = self._get_current_sink_load(y)
        available = self.mu_Y[y] - current_load
        return max(0.0, available)

    def _displace_mass_from_others(self, y, x_current, mass_needed):
        mass_freed = 0.0
        mass_to_free = mass_needed
        
        # Collect all sources holding mass at y (excluding x_current)
        # Sort by descending mass to greedily take from largest holders first
        holders_with_mass = []
        for x_prime in range(self.N_X):
            if x_prime != x_current and self.mu[x_prime, y] > 1e-9:
                holders_with_mass.append((self.mu[x_prime, y], x_prime))
        
        # Sort by mass in descending order (take from largest holders first)
        holders_with_mass.sort(reverse=True, key=lambda item: item[0])
        
        # Displace mass greedily
        for mass_at_y, x_prime in holders_with_mass:
            if mass_to_free < 1e-9:
                break
            
            # Take as much as we need from this holder
            can_displace = min(mass_to_free, mass_at_y)
            
            # Remove from sink y
            self.mu[x_prime, y] -= can_displace
            mass_freed += can_displace
            mass_to_free -= can_displace
        
        return mass_freed

    def _assign_mass_to_target(self, x, y, mass_to_assign):
        # Step 1: Check available capacity (no displacement needed)
        available = self._calculate_available_capacity(y)
        
        if available >= mass_to_assign - 1e-9:
            # Case 1: Enough space; just add the mass
            self.mu[x, y] += mass_to_assign
            return mass_to_assign
        
        # Step 2: y is at or near capacity; try to displace from others
        if available > 1e-9:
            # Partial space: fill what we can
            self.mu[x, y] += available
            mass_remaining = mass_to_assign - available
        else:
            # No space: will need to displace to make ANY room
            mass_remaining = mass_to_assign
            available = 0.0
        
        # Step 3: Displace other sources to make more room
        # We need to free up space for mass_remaining
        mass_freed = self._displace_mass_from_others(y, x, mass_remaining)
        
        # Step 4: Assign the freed space
        space_after_displacement = mass_freed
        can_assign_after = min(mass_remaining, space_after_displacement)
        self.mu[x, y] += can_assign_after
        
        # Total assigned = what we put in + what we added after displacement
        total_assigned = available + can_assign_after
        
        # INVARIANT CHECK (debug only, remove in production)
        # current_sink_load = self._get_current_sink_load(y)
        # assert current_sink_load <= self.mu_Y[y] + 1e-9, \
        #     f"CAPACITY VIOLATION: sink {y} has {current_sink_load} but capacity is {self.mu_Y[y]}"
        
        return total_assigned

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
            
            # ===== FIND BEST AND SECOND-BEST TARGETS =====
            best_y, second_y, best_val, second_val = self._get_best_and_second_best_targets(x, valid_y_indices)
            
            if best_y is None:
                print(f"  ERROR: Could not find best target for source {x}", file=sys.stderr)
                break
            
            # ===== UPDATE PRICE OF BEST TARGET (ε-CS) =====
            price_increment = max(self.epsilon, (second_val - best_val) + 1e-12)
            self.beta[best_y] -= price_increment
            
            # ===== ASSIGN MASS TO BEST TARGET ===== 
            # CRITICAL: This method enforces capacity constraints
            mass_assigned_to_best = self._assign_mass_to_target(x, best_y, mass_to_assign)
            remaining_mass = mass_to_assign - mass_assigned_to_best
            
            # ===== HANDLE RESIDUAL MASS =====
            if remaining_mass > 1e-9 and second_y is not None:
                # Find third-best for price update
                third_val = np.inf
                for y in valid_y_indices:
                    if y != best_y and y != second_y:
                        rc = self.C[x, y] - self.beta[y]
                        if rc < third_val:
                            third_val = rc
                
                if third_val == np.inf:
                    third_val = second_val
                
                # Update price of second target
                second_val_current = self.C[x, second_y] - self.beta[second_y]
                price_increment_second = max(self.epsilon, (third_val - second_val_current) + 1e-12)
                self.beta[second_y] -= price_increment_second
                
                # Try to assign residual to second target
                mass_assigned_to_second = self._assign_mass_to_target(x, second_y, remaining_mass)
                remaining_mass -= mass_assigned_to_second
            
            # ===== MULTI-LEVEL RESIDUAL HANDLING =====
            if remaining_mass > 1e-9:
                tried_targets = {best_y}
                if second_y is not None:
                    tried_targets.add(second_y)
                
                remaining_targets = [y for y in valid_y_indices if y not in tried_targets]
                if len(remaining_targets) > 0:
                    remaining_targets_sorted = sorted(
                        remaining_targets, 
                        key=lambda y: self.C[x, y] - self.beta[y]
                    )
                    for y_next in remaining_targets_sorted:
                        if remaining_mass < 1e-9:
                            break
                        mass_assigned = self._assign_mass_to_target(x, y_next, remaining_mass)
                        remaining_mass -= mass_assigned
            
            iterations += 1
            if iterations % 10000 == 0:
                print(f"  Iteration {iterations}: unassigned mass = {total_unassigned:.9f}", file=sys.stderr)
        
        if iterations >= max_iterations:
            print(f"  WARNING: Reached max iterations ({max_iterations})", file=sys.stderr)
            assigned_X = np.sum(self.mu, axis=1)
            unassigned_X = self.mu_X - assigned_X
            total_unassigned = np.sum(unassigned_X)
            print(f"  Final unassigned mass: {total_unassigned:.9f}", file=sys.stderr)
        
        # ===== VERIFY CONSTRAINTS =====
        assigned_X = np.sum(self.mu, axis=1)
        assigned_Y = np.sum(self.mu, axis=0)
        
        supply_deficit = np.max(np.abs(assigned_X - self.mu_X))
        demand_deficit = np.max(np.abs(assigned_Y - self.mu_Y))
        
        if supply_deficit > 1e-6 or demand_deficit > 1e-6:
            print(f"  WARNING: Constraint violation detected!", file=sys.stderr)
            print(f"    Supply deficit: {supply_deficit:.2e}", file=sys.stderr)
            print(f"    Demand deficit: {demand_deficit:.2e}", file=sys.stderr)
        
        # Compute final cost in original units
        total_cost = np.sum(self.mu * self.C_raw)
        
        return self.mu, total_cost, iterations
