import numpy as np


class EpsScalingManager:
    
    def __init__(self, solver_class, cost_matrix, theta=5.0, target_eps=None, initial_beta=None, **solver_kwargs):
        self.solver_class = solver_class
        self.C_original = np.array(cost_matrix, dtype=float)  # Store original for cost computation
        
        # FIX: Save max_c as an instance attribute so we can use it for tolerances
        self.max_c = np.max(np.abs(self.C_original))
        if self.max_c == 0:
            self.max_c = 1.0  # Prevent division by zero
            
        # Normalize cost matrix for numerical stability
        self.C_normalized = self.C_original / self.max_c
        
        self.theta = theta
        self.solver_kwargs = solver_kwargs
        self.N_X, self.N_Y = self.C_original.shape
        self.initial_beta = initial_beta
        
        # FIX: Compute target epsilon robustly to avoid floating-point stalls
        if target_eps is None:
            relative_tolerance = 1e-4 * self.max_c
            theoretical_bound = 1.0 / float(self.N_X + 1)
            
            # Use the larger of the two to prevent microscopic epsilons
            self.target_eps = max(relative_tolerance, theoretical_bound)
        else:
            self.target_eps = target_eps
        
        # Initialize epsilon for first scaling phase
        C_max_normalized = np.max(self.C_normalized)
        self.start_eps = max(C_max_normalized / 2.0, self.target_eps * self.theta)

    def solve(self):
        current_eps = self.start_eps
        best_beta = self.initial_beta
        final_assignment = None
        total_iterations = 0
        
        print(f"Starting eps-scaling. Start eps: {current_eps:.6f}, Target eps: {self.target_eps:.6f}")
        
        # eps-scaling loop: progressively decrease epsilon
        while current_eps >= self.target_eps:
            print(f"  -> Solving for eps = {current_eps:.6f}")
            
            # Create solver with normalized cost matrix
            # CRITICAL: Pass solver_kwargs which contain mu_X, mu_Y, allowed_edges, etc.
            try:
                solver = self.solver_class(
                    self.C_normalized, 
                    epsilon=current_eps, 
                    **self.solver_kwargs
                )
            except TypeError as e:
                print(f"ERROR creating solver: {e}")
                print(f"  solver_class: {self.solver_class}")
                print(f"  C_normalized shape: {self.C_normalized.shape}")
                print(f"  solver_kwargs keys: {self.solver_kwargs.keys()}")
                for key, val in self.solver_kwargs.items():
                    if isinstance(val, np.ndarray):
                        print(f"    {key}: array of shape {val.shape}, dtype {val.dtype}")
                    else:
                        print(f"    {key}: {type(val).__name__} = {val}")
                raise
            
            # Warm-start with previous beta (if available)
            if best_beta is not None:
                self._inject_beta(solver, best_beta)
            
            # Run auction with fixed epsilon
            result = solver.solve()
            
            # Unpack result (always returns 3-tuple: (mu, cost, iterations))
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
        
        print(f"eps-scaling complete in {total_iterations} total iterations.")
        
        # Compute final cost using ORIGINAL cost matrix (not normalized)
        true_cost = self._calculate_final_cost(final_assignment)
        
        return final_assignment, true_cost, total_iterations, best_beta

    def _inject_beta(self, solver, old_beta):
        solver.beta = np.copy(old_beta)

    def _extract_beta(self, solver):
        return np.copy(solver.beta)

    def _calculate_final_cost(self, assignment):
        if assignment is None:
            raise ValueError("Assignment is None; solver failed to produce output")
        
        # Handle both 1D (permutation) and 2D (coupling) assignments
        if assignment.ndim == 1:
            # 1D assignment (permutation): assignment[x] = y for each x
            cost = sum(self.C_original[x, int(assignment[x])] for x in range(self.N_X))
        elif assignment.ndim == 2:
            # 2D assignment (coupling matrix)
            if assignment.shape != self.C_original.shape:
                raise ValueError(f"Assignment shape {assignment.shape} doesn't match cost shape {self.C_original.shape}")
            cost = np.sum(assignment * self.C_original)
        else:
            raise ValueError(f"Assignment has unexpected shape: {assignment.shape}")
        
        return float(cost)
