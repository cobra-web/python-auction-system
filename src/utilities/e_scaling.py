import numpy as np

class EpsScalingManager:
    def __init__(self, solver_class, cost_matrix, theta=5.0, target_eps=None, **solver_kwargs):
        """
        Parameters:
        - solver_class: The class of the solver to use (e.g., AuctionLAP or AuctionOT).
        - cost_matrix: The dense C matrix.
        - theta: The factor by which epsilon is divided each iteration (typically 4 to 10).
        - target_eps: The threshold for global optimality. If None, it calculates it automatically.
        - solver_kwargs: Additional arguments needed to init the solver (like mu_X, mu_Y).
        """
        self.solver_class = solver_class
        self.C = np.array(cost_matrix)
        self.theta = theta
        self.solver_kwargs = solver_kwargs
        
        self.N = self.C.shape[0]
        
        # 1. Target epsilon: Must be < min_diff / |X| for global optimality.
        if target_eps is None:
            # Safe default for integer costs or general robust optimality
            self.target_eps = 1.0 / (self.N + 1.0)
        else:
            self.target_eps = target_eps
            
        # 2. Starting epsilon: Usually related to the maximum cost variation C
        C_max = np.max(self.C)
        self.start_eps = max(C_max / 2.0, self.target_eps * self.theta)

    def solve(self):
        """
        Runs the successive auction solves, passing dual variables forward.
        """
        current_eps = self.start_eps
        best_beta = None
        final_assignment = None
        total_iterations = 0
        
        print(f"Starting e-scaling. Target eps: {self.target_eps:.4f}")
        
        while current_eps >= self.target_eps:
            print(f"  -> Solving for eps = {current_eps:.4f}")
            
            # Instantiate the solver for this iteration
            solver = self.solver_class(self.C, epsilon=current_eps, **self.solver_kwargs)
            
            # WARM START: Inject prices (beta) from the previous iteration
            if best_beta is not None:
                self._inject_beta(solver, best_beta)
                
            # Solve the problem
            result = solver.solve()
            
            # Result unpacking depends on whether it's LAP or OT
            if len(result) == 3:
                final_assignment, _, iters = result
            
            total_iterations += iters
            
            # Extract the duals to pass to the next loop
            best_beta = self._extract_beta(solver)
            
            # Decrease epsilon
            # We break manually to ensure we run EXACTLY at target_eps for the final run
            if current_eps == self.target_eps:
                break
                
            current_eps = max(current_eps / self.theta, self.target_eps)
            
        print(f"e-scaling complete in {total_iterations} total iterations.")
        
        # Recalculate true cost using the final assignment
        true_cost = self._calculate_final_cost(final_assignment)
        return final_assignment, true_cost, total_iterations, best_beta

    def _inject_beta(self, solver, old_beta):
        """Helper to inject prices depending on the solver type."""
        if hasattr(solver, 'y_atoms'):
            # It's the OT solver: update the empty state price
            for y in range(solver.N_Y):
                if len(solver.y_atoms[y]) > 0:
                    solver.y_atoms[y][0]['beta'] = old_beta[y]
        else:
            # It's the LAP solver
            solver.beta = np.copy(old_beta)

    def _extract_beta(self, solver):
        """Helper to extract prices depending on the solver type."""
        if hasattr(solver, 'y_atoms'):
            # Extract the lowest price for each sink y
            return np.array([min(atom['beta'] for atom in atoms) for atoms in solver.y_atoms])
        else:
            return np.copy(solver.beta)

    def _calculate_final_cost(self, assignment):
        """Helper to compute cost without the epsilon penalties."""
        if assignment.ndim == 1: # LAP (1D array)
            return sum(self.C[x, assignment[x]] for x in range(self.N))
        else:                    # OT (Coupling Matrix)
            return np.sum(assignment * self.C)
