import numpy as np

class EpsScalingManager:
    
    def __init__(self, solver_class, cost_matrix, theta=5.0, target_eps=None, initial_beta=None, **solver_kwargs):
        self.solver_class = solver_class
        self.C_original = np.array(cost_matrix, dtype=float) 
        
        self.max_c = np.max(np.abs(self.C_original))
        if self.max_c == 0:
            self.max_c = 1.0
            
        # Normalize for internal solver
        self.C_normalized = self.C_original / self.max_c
        self.theta = theta
        self.solver_kwargs = solver_kwargs
        self.N_X, self.N_Y = self.C_original.shape
        
        # Scale initial beta down for the normalized internal solver
        self.initial_beta_normalized = initial_beta / self.max_c if initial_beta is not None else None
        
        # Epsilon setup strictly in normalized [0, 1] space
        if target_eps is None:
            self.target_eps_normalized = 1e-4
        else:
            self.target_eps_normalized = target_eps / self.max_c
            
        # EXPOSE absolute target eps for external consistency checks
        self.target_eps_absolute = self.target_eps_normalized * self.max_c
        
        C_max_normalized = 1.0
        self.start_eps = max(C_max_normalized / 2.0, self.target_eps_normalized * self.theta)

    def solve(self):
        current_eps = self.start_eps
        best_beta = self.initial_beta_normalized
        final_assignment = None
        total_iterations = 0
        
        print(f"Starting eps-scaling. Start eps: {current_eps:.6f}, Target eps: {self.target_eps_normalized:.6f}")
        
        while current_eps >= self.target_eps_normalized:
            # current_eps is ALREADY normalized here
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
        
        # Lift beta back to original absolute scale before returning to the Multiscale solver
        return final_assignment, true_cost, total_iterations, best_beta * self.max_c

    def _calculate_final_cost(self, assignment):
        if assignment is None:
            raise ValueError("Assignment is None")
        
        if assignment.ndim == 1:
            cost = sum(self.C_original[x, int(assignment[x])] for x in range(self.N_X))
        elif assignment.ndim == 2:
            cost = np.sum(assignment * self.C_original)
        else:
            raise ValueError("Assignment has unexpected shape")
        
        return float(cost)
