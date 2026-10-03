import numpy as np


class EpsScalingManager:
    """Epsilon-scaling wrapper around an auction solver (Bertsekas 1979/1988).

    All costs handed to the solver are normalised by max_c, so C_max = 1 here. This is only
    correct if the caller passes the GLOBAL max_c (the same value on every level).

    Accuracy: a final eps-CS plan is optimal up to M * eps * C_max (M = total mass, C_max = 1 in
    normalised units, i.e. M * eps * max_c in raw cost units). Therefore the default target is
    eps = accuracy / M. Exactness on real-valued costs cannot be certified by a finite eps.
    """

    def __init__(self, solver_class, X_pts, Y_pts, mu_X, mu_Y, theta=5.0, target_eps=None,
                 min_eps=1e-12, start_eps=None, initial_beta=None, accuracy=1e-3,
                 verbose=False, **solver_kwargs):
        self.solver_class = solver_class
        self.X_pts = np.array(X_pts, dtype=float)
        self.Y_pts = np.array(Y_pts, dtype=float)
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)

        self.theta = float(theta)
        self.min_eps = max(float(min_eps), 1e-15)
        self.solver_kwargs = solver_kwargs
        self.verbose = verbose

        self.N_X = len(self.X_pts)
        self.N_Y = len(self.Y_pts)
        self.initial_beta = initial_beta

        total_mass = max(float(np.sum(self.mu_X)), 1.0)
        if target_eps is None:
            self.target_eps = float(accuracy) / total_mass
        else:
            self.target_eps = float(target_eps)

        C_max = 1.0  # costs are normalised by max_c inside the solver

        self.effective_target = max(self.target_eps, self.min_eps)

        if start_eps is not None:
            self.start_eps = float(start_eps)
        elif self.initial_beta is None:
            # cold start: full ladder from C_max / 2 down to the target
            self.start_eps = max(C_max / 2.0, self.effective_target * self.theta)
        else:
            # warm start: a few repair steps above the target
            self.start_eps = min(C_max / 2.0, self.effective_target * (self.theta ** 3))
        self.start_eps = max(self.start_eps, self.effective_target)

    def solve(self):
        current_eps = self.start_eps
        best_beta = self.initial_beta
        final_assignment = None
        final_cost = 0.0
        total_iterations = 0

        if self.verbose:
            print(f"Starting eps-scaling. Start eps: {current_eps:.6e}, "
                  f"Target eps: {self.target_eps:.6e}, Floor: {self.effective_target:.6e}")

        while True:
            if self.verbose:
                print(f"  -> Solving for eps = {current_eps:.6e}")

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

            if current_eps <= self.effective_target:
                break
            current_eps = max(current_eps / self.theta, self.effective_target)

        if self.verbose:
            print(f"eps-scaling complete in {total_iterations} total iterations.")
        self.final_eps = current_eps
        return final_assignment, final_cost, total_iterations, best_beta

    def _inject_beta(self, solver, old_beta):
        solver.beta_diamond = np.copy(old_beta)

    def _extract_beta(self, solver):
        return solver.get_effective_beta()
