import numpy as np

class EpsScalingManager:
    def __init__(self, solver_class, cost_matrix, theta=5.0, target_eps=None, initial_beta=None, **solver_kwargs):
        self.solver_class = solver_class
        self.C = np.array(cost_matrix)
        self.theta = theta
        self.solver_kwargs = solver_kwargs
        self.N = self.C.shape[0]
        self.initial_beta = initial_beta
        
        if target_eps is None:
            unique_costs = np.unique(self.C)
            if len(unique_costs) > 1:
                delta_c = np.min(np.diff(unique_costs))
            else:
                delta_c = 1.0 
                
            self.target_eps = delta_c / float(self.N)
        else:
            self.target_eps = target_eps
            
        C_max = np.max(self.C)
        self.start_eps = max(C_max / 2.0, self.target_eps * self.theta)

    def solve(self):
        current_eps = self.start_eps
        best_beta = self.initial_beta
        final_assignment = None
        total_iterations = 0
        
        print(f"Starting e-scaling. Target eps: {self.target_eps:.6f}")
        
        while current_eps >= self.target_eps:
            print(f"  -> Solving for eps = {current_eps:.6f}")
            
            solver = self.solver_class(self.C, epsilon=current_eps, **self.solver_kwargs)
            
            if best_beta is not None:
                self._inject_beta(solver, best_beta)
                
            result = solver.solve()
            
            if len(result) == 3:
                final_assignment, _, iters = result
            
            total_iterations += iters
            best_beta = self._extract_beta(solver)
            
            if current_eps <= self.target_eps:
                break
                
            current_eps = max(current_eps / self.theta, self.target_eps)
            
        print(f"e-scaling complete in {total_iterations} total iterations.")
        
        true_cost = self._calculate_final_cost(final_assignment)
        
        return final_assignment, true_cost, total_iterations, best_beta

    def _inject_beta(self, solver, old_beta):
        solver.beta = np.copy(old_beta)

    def _extract_beta(self, solver):
        return np.copy(solver.beta)

    def _calculate_final_cost(self, assignment):
        if assignment.ndim == 1: 
            return sum(self.C[x, assignment[x]] for x in range(self.N))
        else:                    
            return np.sum(assignment * self.C)
