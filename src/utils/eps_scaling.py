import numpy as np

class EpsScalingManager:
    
    def __init__(self, solver_class, cost_matrix, theta=5.0, target_eps=None, initial_beta=None, **solver_kwargs):
        self.solver_class = solver_class
        self.C_original = np.array(cost_matrix, dtype=float)  # Store original for cost computation
        self.C_normalized = self.C_original.copy()  # Will be normalized for solver
        
        # Normalize cost matrix for numerical stability
        max_c = np.max(np.abs(self.C_original))
        if max_c > 0:
            self.C_normalized = self.C_original / max_c
        
        self.theta = theta
        self.solver_kwargs = solver_kwargs
        self.N = self.C_original.shape[0]
        self.initial_beta = initial_beta
        
        # Compute target epsilon from cost distribution
        if target_eps is None:
            unique_costs = np.unique(self.C_normalized)
            if len(unique_costs) > 1:
                delta_c = np.min(np.diff(unique_costs))
            else:
                delta_c = 1.0
            
            self.target_eps = delta_c / float(self.N)
        else:
            self.target_eps = target_eps
        
        # Initialize epsilon for first scaling phase
        C_max = np.max(self.C_normalized)
        self.start_eps = max(C_max / 2.0, self.target_eps * self.theta)

    def solve(self):
        current_eps = self.start_eps
        best_beta = self.initial_beta
        final_assignment = None
        total_iterations = 0
        
        print(f"Starting ε-scaling. Start ε: {current_eps:.6f}, Target ε: {self.target_eps:.6f}")
        
        # ε-scaling loop: progressively decrease epsilon
        while current_eps >= self.target_eps:
            print(f"  -> Solving for ε = {current_eps:.6f}")
            
            # Create solver with normalized cost matrix
            solver = self.solver_class(
                self.C_normalized, 
                epsilon=current_eps, 
                **self.solver_kwargs
            )
            
            # Warm-start with previous beta (if available)
            if best_beta is not None:
                self._inject_beta(solver, best_beta)
            
            # Run auction with fixed epsilon
            result = solver.solve()
            
            # Unpack result
            if len(result) == 3:
                final_assignment, _, iters = result
            else:
                final_assignment, _, iters = result[:3]
            
            total_iterations += iters
            
            # Extract dual variables for next epsilon phase
            best_beta = self._extract_beta(solver)
            
            # Check if we're done with this epsilon
            if current_eps <= self.target_eps:
                break
            
            # Reduce epsilon for next iteration
            current_eps = max(current_eps / self.theta, self.target_eps)
        
        print(f"ε-scaling complete in {total_iterations} total iterations.")
        
        # Compute final cost using ORIGINAL cost matrix (not normalized)
        true_cost = self._calculate_final_cost(final_assignment)
        
        return final_assignment, true_cost, total_iterations, best_beta

    def _inject_beta(self, solver, old_beta):
        solver.beta = np.copy(old_beta)

    def _extract_beta(self, solver):
        """Extract dual variables from solver for next phase"""
        return np.copy(solver.beta)

    def _calculate_final_cost(self, assignment):
        if assignment.ndim == 1:
            # 1D assignment (permutation)
            return sum(self.C_original[x, assignment[x]] for x in range(self.N))
        else:
            # 2D coupling matrix
            return np.sum(assignment * self.C_original)
