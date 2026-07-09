import numpy as np

class EpsScalingManager:
    
    def __init__(self, solver_class, cost_matrix, theta=5.0, target_eps=None, initial_beta=None, **solver_kwargs):
        self.solver_class = solver_class
        self.C_original = np.array(cost_matrix, dtype=float) 
        
        self.max_c = np.max(np.abs(self.C_original))
        if self.max_c == 0: self.max_c = 1.0
            
        self.C_normalized = self.C_original / self.max_c
        self.theta = theta
        self.solver_kwargs = solver_kwargs
        self.N_X, self.N_Y = self.C_original.shape
        
        if target_eps is None:
            self.target_eps_normalized = 1e-4
        else:
            self.target_eps_normalized = target_eps / self.max_c
            
        self.target_eps_absolute = self.target_eps_normalized * self.max_c
        
        # CRITICAL FIX: Always start epsilon scaling from a large value (C_max/2). 
        # This allows the auction to take "giant leaps" to instantly balance the restricted sparse graph.
        self.start_eps = max(1.0 / 2.0, self.target_eps_normalized * self.theta)

    def solve(self):
        current_eps = self.start_eps
        
        # Discard dual warm-starts; rely on structural warm-starts (N_guess)
        best_beta = None 
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
        return final_assignment, true_cost, total_iterations, best_beta * self.max_c

    def _calculate_final_cost(self, assignment):
        if assignment is None:
            raise ValueError("Assignment is None")
        if assignment.ndim == 1:
            cost = sum(self.C_original[x, int(assignment[x])] for x in range(self.N_X))
        elif assignment.ndim == 2:
            cost = np.sum(assignment * self.C_original)
        return float(cost)
