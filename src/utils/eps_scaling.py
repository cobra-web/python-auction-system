import numpy as np

class EpsScalingManager:
    
    def __init__(self, solver_class, cost_matrix, theta=5.0, target_eps=None, initial_beta=None, **solver_kwargs):
        self.solver_class = solver_class
        self.C_original = np.array(cost_matrix, dtype=float) 
        
        self.max_c = np.max(np.abs(self.C_original))
        if self.max_c == 0:
            self.max_c = 1.0
            
        self.C_normalized = self.C_original / self.max_c
        self.theta = theta
        self.solver_kwargs = solver_kwargs
        self.N_X, self.N_Y = self.C_original.shape
        
        self.initial_beta_normalized = initial_beta / self.max_c if initial_beta is not None else None
        
        # Keep internal eps entirely in normalized [0,1] space
        if target_eps is None:
            self.target_eps_normalized = 1e-4
        else:
            self.target_eps_normalized = target_eps / self.max_c
            
        self.target_eps_absolute = self.target_eps_normalized * self.max_c
        
        # CRITICAL FIX: Do not nuke the warm-start! 
        # If we have an initial beta, run only a single phase at target_eps.
        if self.initial_beta_normalized is not None:
            self.start_eps = self.target_eps_normalized
        else:
            self.start_eps = max(1.0 / 2.0, self.target_eps_normalized * self.theta)

    def solve(self):
        current_eps = self.start_eps
        best_beta = self.initial_beta_normalized
        final_assignment = None
        total_iterations = 0
        
        while current_eps >= self.target_eps_normalized:
            solver = self.solver_class(
                self.C_normalized, 
                epsilon=current_eps, 
                **self.solver_kwargs
            )
            
            if best_beta is not None:
                solver.beta = np.copy(best_beta)
            
            result = solver.solve()
            
            if len(result) == 3:
                final_assignment, _, iters = result
            else:
                final_assignment, _, iters = result[:3]
            
            total_iterations += iters
            best_beta = np.copy(solver.beta)
            
            if current_eps <= self.target_eps_normalized:
                break
            
            current_eps = max(current_eps / self.theta, self.target_eps_normalized)
        
        true_cost = self._calculate_final_cost(final_assignment)
        
        # Return absolute prices to the Multiscale solver
        return final_assignment, true_cost, total_iterations, best_beta * self.max_c

    def _calculate_final_cost(self, assignment):
        if assignment is None:
            raise ValueError("Assignment is None")
        if assignment.ndim == 1:
            cost = sum(self.C_original[x, int(assignment[x])] for x in range(self.N_X))
        elif assignment.ndim == 2:
            cost = np.sum(assignment * self.C_original)
        return float(cost)
