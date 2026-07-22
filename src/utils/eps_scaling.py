import numpy as np

class EpsScalingManager:
    """epsilon-scaling driver for the auction OT solver with a safe floor."""

    def __init__(self, solver_class, X_pts, Y_pts, mu_X, mu_Y, theta=5.0, target_eps=None,
                 min_eps=1e-4, initial_beta=None, **solver_kwargs):
        self.solver_class = solver_class
        self.X_pts = np.array(X_pts, dtype=float)
        self.Y_pts = np.array(Y_pts, dtype=float)
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)

        self.theta = float(theta)
        self.min_eps = float(min_eps)  
        self.solver_kwargs = solver_kwargs
        
        self.N_X = len(self.X_pts)
        self.N_Y = len(self.Y_pts)
        self.initial_beta = initial_beta

        if target_eps is None:
            self.target_eps = 1.0 / float(self.N_X + 1)
        else:
            self.target_eps = float(target_eps)

        C_max = 1.0
        self.start_eps = max(C_max / 2.0, self.target_eps * self.theta)

    def solve(self):
        current_eps = self.start_eps
        best_beta = self.initial_beta
        final_assignment = None
        final_cost = 0.0
        total_iterations = 0

        effective_target = max(self.target_eps, self.min_eps)

        print(f"Starting eps-scaling. Start eps: {current_eps:.6e}, "
              f"Target eps: {self.target_eps:.6e}, Safe Floor: {effective_target:.6e}")

        while current_eps >= effective_target:
            print(f"  -> Solving for eps = {current_eps:.6e}")

            # Intercept and force normalize to True to prevent kwarg clashes 
            # and guarantee safe epsilon-scaling math.
            safe_kwargs = self.solver_kwargs.copy()
            if "normalize" not in safe_kwargs:
                safe_kwargs["normalize"] = True

            solver = self.solver_class(
                X_pts=self.X_pts,
                Y_pts=self.Y_pts,
                mu_X=self.mu_X,
                mu_Y=self.mu_Y,
                epsilon=current_eps,
                **safe_kwargs
            )

            if best_beta is not None:
                self._inject_beta(solver, best_beta)

            mu, cost, iters = solver.solve()
            final_assignment = mu
            final_cost = cost
            total_iterations += iters

            best_beta = self._extract_beta(solver)

            if current_eps <= effective_target:
                break
            current_eps = max(current_eps / self.theta, effective_target)

        print(f"eps-scaling complete in {total_iterations} total iterations.")
        return final_assignment, final_cost, total_iterations, best_beta

    def _inject_beta(self, solver, old_beta):
        solver.beta_diamond = np.copy(old_beta)

    def _extract_beta(self, solver):
        return solver.get_effective_beta()
