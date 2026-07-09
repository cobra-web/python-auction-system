import numpy as np


class EpsScalingManager:
    """epsilon-scaling driver for the auction OT solver.

    epsilon-scaling (Bertsekas/Eckstein; referenced in Schmitzer 2013, Sec. 5 as
    the remedy for the runtime's sensitivity to the cost range C, replacing the
    factor C by log(|X|*C) in the complexity) solves a sequence of auctions with
    a geometrically DECREASING epsilon, warm-starting each phase with the dual
    variable beta of the previous, coarser phase. Large epsilon avoids the
    "price haggling" plateaus; the warm start means the fine phases start already
    close to optimal.
    """

    def __init__(self, solver_class, cost_matrix, theta=5.0, target_eps=None,
                 initial_beta=None, **solver_kwargs):
        self.solver_class = solver_class
        self.C_original = np.array(cost_matrix, dtype=float)  # for TRUE final cost

        # --- Normalise ONCE, here. ------------------------------------------------
        # BUG FIX: the solver also normalised internally, so cost was divided by
        # its max twice and the epsilon schedule (computed on THIS normalised
        # matrix) no longer matched the scale the solver ran on. We now normalise
        # once and pass normalize=False to the solver so there is a single,
        # consistent cost scale for C, beta and epsilon.
        max_c = np.max(np.abs(self.C_original))
        self.scale = max_c if max_c > 0 else 1.0
        self.C_normalized = self.C_original / self.scale

        self.theta = float(theta)
        self.solver_kwargs = solver_kwargs
        self.N_X, self.N_Y = self.C_original.shape
        self.initial_beta = initial_beta

        # --- Target epsilon ------------------------------------------------------
        # Schmitzer's exactness threshold (Sec. 3): epsilon < delta_c / |X|, where
        # delta_c is the smallest gap between two distinct cost values. We take a
        # value strictly below that bound so the final coupling is exact for
        # integer / grid-quantised costs. For genuinely continuous costs delta_c
        # can be tiny; then you converge to an epsilon-approximate solution and
        # may prefer to pass target_eps explicitly.
        if target_eps is None:
            unique_costs = np.unique(self.C_normalized)
            if len(unique_costs) > 1:
                delta_c = float(np.min(np.diff(unique_costs)))
            else:
                delta_c = 1.0
            # strictly below delta_c / N_X
            self.target_eps = delta_c / float(self.N_X + 1)
        else:
            self.target_eps = float(target_eps)

        # First (coarsest) epsilon: large enough to skip haggling but at least a
        # few theta-steps above target so scaling actually happens.
        C_max = float(np.max(self.C_normalized))
        self.start_eps = max(C_max / 2.0, self.target_eps * self.theta)

    def solve(self):
        current_eps = self.start_eps
        best_beta = self.initial_beta
        final_assignment = None
        total_iterations = 0

        print(f"Starting eps-scaling. Start eps: {current_eps:.6e}, "
              f"Target eps: {self.target_eps:.6e}")

        while current_eps >= self.target_eps:
            print(f"  -> Solving for eps = {current_eps:.6e}")

            # Build the solver on the ONCE-normalised matrix. normalize=False
            # prevents the double-normalisation described above.
            solver = self.solver_class(
                self.C_normalized,
                epsilon=current_eps,
                normalize=False,
                **self.solver_kwargs,
            )

            # Warm start: inject the previous phase's beta (same cost scale, so
            # it is a valid dual for the finer epsilon).
            if best_beta is not None:
                self._inject_beta(solver, best_beta)

            mu, _, iters = solver.solve()
            final_assignment = mu
            total_iterations += iters

            # Carry duals forward to the next, finer phase.
            best_beta = self._extract_beta(solver)

            if current_eps <= self.target_eps:
                break
            current_eps = max(current_eps / self.theta, self.target_eps)

        print(f"eps-scaling complete in {total_iterations} total iterations.")

        true_cost = self._calculate_final_cost(final_assignment)
        return final_assignment, true_cost, total_iterations, best_beta

    # ----------------------------------------------------------------- warm start
    def _inject_beta(self, solver, old_beta):
        solver.beta = np.copy(old_beta)

    def _extract_beta(self, solver):
        return np.copy(solver.beta)

    # ------------------------------------------------------------- final true cost
    def _calculate_final_cost(self, assignment):
        if assignment is None:
            raise ValueError("Assignment is None; solver failed to produce output.")

        if assignment.ndim == 1:
            cost = sum(self.C_original[x, int(assignment[x])]
                       for x in range(self.N_X))
        elif assignment.ndim == 2:
            if assignment.shape != self.C_original.shape:
                raise ValueError(
                    f"Assignment shape {assignment.shape} doesn't match cost "
                    f"shape {self.C_original.shape}"
                )
            # True cost uses the ORIGINAL (un-normalised) cost matrix.
            cost = float(np.sum(assignment * self.C_original))
        else:
            raise ValueError(f"Assignment has unexpected shape: {assignment.shape}")

        return float(cost)
