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

    def _displace_mass_from_others(self, y, x_current, mass_needed):
        mass_freed = 0.0
        mass_to_free = mass_needed
        
        # Collect all sources holding mass at y (excluding x_current)
        # Format: (mass_at_y, x_prime)
        holders_with_mass = []
        for x_prime in range(self.N_X):
            if x_prime != x_current:
                mass_held = self.mu[x_prime, y]
                # Only consider if mass is above tolerance
                if mass_held > TOL:
                    holders_with_mass.append((mass_held, x_prime))
        
        # Sort by mass in descending order (take from largest holders first - greedy)
        holders_with_mass.sort(reverse=True, key=lambda item: item[0])
        
        # Displace mass greedily from largest holders
        for mass_at_y, x_prime in holders_with_mass:
            if mass_to_free < TOL:
                # Tolerance check: stop if we need less than TOL
                break
            
            # How much can we take from x_prime at sink y?
            can_displace = min(mass_to_free, mass_at_y)
            
            # ===== STRICT MASS CONSERVATION =====
            # REMOVE from sink y
            self.mu[x_prime, y] -= can_displace
            
            # RETURN to x_prime as unassigned mass
            # (This is crucial: the displaced mass goes back to x_prime to be re-bid)
            # We track this implicitly: unassigned_X[x_prime] will be recomputed
            # in solve() as mu_X[x_prime] - sum(mu[x_prime, :])
            
            mass_freed += can_displace
            mass_to_free -= can_displace
            
            # Verify conservation within this step
            # (unassigned mass will be verified in solve() globally)
        
        return mass_freed

    def _assign_mass_to_target(self, x, y, mass_to_assign):
        # Apply tolerance: don't assign if amount is negligible
        if mass_to_assign < TOL:
            return 0.0
        
        # Step 1: Check available capacity (no displacement needed)
        available = self._calculate_available_capacity(y)
        
        if available >= mass_to_assign - TOL:
            # Case 1: Enough space; just add the mass
            self.mu[x, y] += mass_to_assign
            return mass_to_assign
        
        # Step 2: y is at or near capacity; need to displace
        # First, add what we can without displacement
        total_assigned = 0.0
        if available > TOL:
            self.mu[x, y] += available
            total_assigned += available
            mass_remaining = mass_to_assign - available
        else:
            mass_remaining = mass_to_assign
        
        # Step 3: Displace other sources to make room for the remaining mass
        if mass_remaining > TOL:
            mass_freed = self._displace_mass_from_others(y, x, mass_remaining)
            
            # Step 4: Assign as much of the freed space as we need
            can_assign_after_displacement = min(mass_remaining, mass_freed)
            if can_assign_after_displacement > TOL:
                self.mu[x, y] += can_assign_after_displacement
                total_assigned += can_assign_after_displacement
        
        return total_assigned

    def solve(self):
        iterations = 0
        max_iterations = 100000
        
        while iterations < max_iterations:
            # Compute unassigned mass for each source
            assigned_X = np.sum(self.mu, axis=1)
            unassigned_X = self.mu_X - assigned_X
            
            # Check convergence: all mass assigned (up to TOL)
            total_unassigned = np.sum(unassigned_X)
            if total_unassigned < TOL:
                print(f"  Auction converged after {iterations} iterations")
                break
            
            # Pick source with most unassigned mass
            x = np.argmax(unassigned_X)
            mass_to_assign = unassigned_X[x]
            
            # Skip if this source has negligible unassigned mass (below tolerance)
            if mass_to_assign < TOL:
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
            
            # ===== UPDATE PRICE OF BEST TARGET (ε-COMPLEMENTARY SLACKNESS) =====
            # CRITICAL: Price increment must be at least epsilon
            # If (second_val - best_val) is negative due to float error, use epsilon
            price_gap = second_val - best_val
            price_increment = max(self.epsilon, price_gap + 1e-12)
            
            # CRITICAL: Prices ALWAYS increase (beta gets MORE NEGATIVE)
            self.beta[best_y] -= price_increment
            
            # ===== ASSIGN MASS TO BEST TARGET =====
            mass_assigned_to_best = self._assign_mass_to_target(x, best_y, mass_to_assign)
            remaining_mass = mass_to_assign - mass_assigned_to_best
            
            # ===== HANDLE RESIDUAL MASS: TRY SECOND-BEST =====
            if remaining_mass > TOL and second_y is not None:
                # Find third-best target (for price update of second_y)
                third_val = np.inf
                for y in valid_y_indices:
                    if y != best_y and y != second_y:
                        rc = self.C[x, y] - self.beta[y]
                        if rc < third_val:
                            third_val = rc
                
                # If no third target exists, use second_val as fallback
                if third_val == np.inf:
                    third_val = second_val
                
                # Update price of second target
                second_val_current = self.C[x, second_y] - self.beta[second_y]
                price_gap_second = third_val - second_val_current
                price_increment_second = max(self.epsilon, price_gap_second + 1e-12)
                self.beta[second_y] -= price_increment_second
                
                # Try to assign residual to second target
                mass_assigned_to_second = self._assign_mass_to_target(x, second_y, remaining_mass)
                remaining_mass -= mass_assigned_to_second
            
            # ===== MULTI-TARGET RESIDUAL HANDLING =====
            if remaining_mass > TOL:
                tried_targets = {best_y}
                if second_y is not None:
                    tried_targets.add(second_y)
                
                remaining_targets = [y for y in valid_y_indices if y not in tried_targets]
                if len(remaining_targets) > 0:
                    # Sort by reduced cost
                    remaining_targets_sorted = sorted(
                        remaining_targets, 
                        key=lambda y: self.C[x, y] - self.beta[y]
                    )
                    
                    # Try each remaining target
                    for y_next in remaining_targets_sorted:
                        if remaining_mass < TOL:
                            break
                        
                        # Update price for this target
                        rc_current = self.C[x, y_next] - self.beta[y_next]
                        # Find next-best after y_next
                        next_best = np.inf
                        for y_other in remaining_targets_sorted:
                            if y_other != y_next:
                                rc_other = self.C[x, y_other] - self.beta[y_other]
                                if rc_other < next_best:
                                    next_best = rc_other
                        
                        if next_best == np.inf:
                            next_best = rc_current
                        
                        price_inc = max(self.epsilon, (next_best - rc_current) + 1e-12)
                        self.beta[y_next] -= price_inc
                        
                        # Assign residual
                        mass_assigned = self._assign_mass_to_target(x, y_next, remaining_mass)
                        remaining_mass -= mass_assigned
            
            iterations += 1
            if iterations % 10000 == 0:
                print(f"  Iteration {iterations}: total unassigned mass = {total_unassigned:.9f}", 
                      file=sys.stderr)
        
        # ===== CONVERGENCE CHECK AND DIAGNOSTICS =====
        if iterations >= max_iterations:
            print(f"  WARNING: Reached max iterations ({max_iterations})", file=sys.stderr)
            assigned_X = np.sum(self.mu, axis=1)
            unassigned_X = self.mu_X - assigned_X
            total_unassigned = np.sum(unassigned_X)
            print(f"  Final unassigned mass: {total_unassigned:.9f}", file=sys.stderr)
        
        # ===== VERIFY CONSERVATION AND CONSTRAINTS =====
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
