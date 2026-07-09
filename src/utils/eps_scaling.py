import numpy as np

class EpsScalingManager:

    @staticmethod
    def compute_target_eps_absolute(cost_matrix, target_eps=None):
        """
        Pure function version of the target_eps_absolute calculation below.
        Depends only on the cost matrix and the requested target_eps -- NOT
        on allowed_edges, mu_X/mu_Y, or any solver state -- so it can be
        computed for a generation's cost matrix before a neighborhood has
        been chosen, e.g. to certify a sparse neighborhood before ever
        constructing/solving with it.
        """
        C = np.asarray(cost_matrix, dtype=float)
        max_c = np.max(np.abs(C))
        if max_c == 0:
            max_c = 1.0
        target_eps_normalized = 1e-4 if target_eps is None else target_eps / max_c
        return target_eps_normalized * max_c

    def __init__(self, solver_class, cost_matrix, theta=5.0, target_eps=None, initial_beta=None, **solver_kwargs):
        self.solver_class = solver_class
        self.C_original = np.array(cost_matrix, dtype=float) 
        
        self.max_c = np.max(np.abs(self.C_original))
        if self.max_c == 0: self.max_c = 1.0
            
        self.C_normalized = self.C_original / self.max_c
        self.theta = theta
        self.solver_kwargs = solver_kwargs
        self.N_X, self.N_Y = self.C_original.shape
        
        self.initial_beta_normalized = initial_beta / self.max_c if initial_beta is not None else None
        
        self.target_eps_absolute = self.compute_target_eps_absolute(self.C_original, target_eps)
        self.target_eps_normalized = self.target_eps_absolute / self.max_c
        
        # WARM START FIX: Give the solver a short 2-phase adjustment window 
        # to adapt to newly added edges without resetting all the way to 0.5.
        if self.initial_beta_normalized is not None:
            self.start_eps = max(self.target_eps_normalized * (self.theta ** 2), self.target_eps_normalized)
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
        return final_assignment, true_cost, total_iterations, best_beta * self.max_c

    def _calculate_final_cost(self, assignment):
        if assignment is None:
            raise ValueError("Assignment is None")
        if assignment.ndim == 1:
            cost = sum(self.C_original[x, int(assignment[x])] for x in range(self.N_X))
        elif assignment.ndim == 2:
            cost = np.sum(assignment * self.C_original)
        return float(cost)
