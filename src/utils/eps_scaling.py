import numpy as np

class EpsScalingManager:
    """epsilon-scaling driver for the auction OT solver with a safe floor."""

    def __init__(self, solver_class, cost_matrix, theta=5.0, target_eps=None,
                 min_eps=1e-4, initial_beta=None, **solver_kwargs):
        self.solver_class = solver_class
        self.C_original = np.array(cost_matrix, dtype=float)

        max_c = np.max(np.abs(self.C_original))
        self.scale = max_c if max_c > 0 else 1.0
        self.C_normalized = self.C_original / self.scale

        self.theta = float(theta)
        self.min_eps = float(min_eps)  # Guard against continuous mass Zeno's paradox
        self.solver_kwargs = solver_kwargs
        self.N_X, self.N_Y = self.C_original.shape
        self.initial_beta = initial_beta

        if target_eps is None:
            unique_costs = np.unique(self.C_normalized)
            if len(unique_costs) > 1:
                delta_c = float(np.min(np.diff(unique_costs)))
            else:
                delta_c = 1.0
            self.target_eps = delta_c / float(self.N_X + 1)
        else:
            self.target_eps = float(target_eps)

        C_max = float(np.max(self.C_normalized))
        self.start_eps = max(C_max / 2.0, self.target_eps * self.theta)

    def solve(self):
        current_eps = self.start_eps
        best_beta = self.initial_beta
        final_assignment = None
        total_iterations = 0

        # Enforce the safe mathematical floor
        effective_target = max(self.target_eps, self.min_eps)

        print(f"Starting eps-scaling. Start eps: {current_eps:.6e}, "
              f"Target eps: {self.target_eps:.6e}, Safe Floor: {effective_target:.6e}")

        while current_eps >= effective_target:
            print(f"  -> Solving for eps = {current_eps:.6e}")

            solver = self.solver_class(
                self.C_normalized,
                epsilon=current_eps,
                normalize=False,
                **self.solver_kwargs,
            )

            if best_beta is not None:
                self._inject_beta(solver, best_beta)

            mu, _, iters = solver.solve()
            final_assignment = mu
            total_iterations += iters

            best_beta = self._extract_beta(solver)

            if current_eps <= effective_target:
                break
            current_eps = max(current_eps / self.theta, effective_target)

        print(f"eps-scaling complete in {total_iterations} total iterations.")
        true_cost = self._calculate_final_cost(final_assignment)
        return final_assignment, true_cost, total_iterations, best_beta

    def _inject_beta(self, solver, old_beta):
        # We pass the 1D extraction back as the baseline beta_diamond
        solver.beta_diamond = np.copy(old_beta)

    def _extract_beta(self, solver):
        # Extract the effective 1D beta for the warm start of the next phase
        return solver.get_effective_beta()

    def _calculate_final_cost(self, assignment):
        if assignment is None:
            raise ValueError("Assignment is None; solver failed to produce output.")
        cost = float(np.sum(assignment * self.C_original))
        return float(cost)
