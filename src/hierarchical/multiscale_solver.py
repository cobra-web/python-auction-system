import sys
from collections import defaultdict

import numpy as np

from src.core.ot_auction import AuctionOT
from src.utils.eps_scaling import EpsScalingManager
from src.hierarchical.consistency import ConsistencyChecker


class HierarchicalMultiscaleSolver:
    """Hierarchical multiscale auction for discrete OT (Schmitzer-Schnoerr 2013).

    Level d solves the problem on the active cells at depth d (mass = cell mass, location = cell
    centroid, cost = squared centroid distance / max_c) restricted to a sparse neighbourhood N_d.
    N_d is induced from the support of the coarser plan; the consistency check then adds every pair
    that would violate eps-CS in the full problem on that level. The loop repeats until nothing is
    added.

    Exactness: the final level must consist of singleton cells, then it is the original problem
    with the true point costs and no mass splitting inside cells takes place. The solver therefore
    REQUIRES trees whose leaves hold exactly one point (build with max_points_per_cell=1 and a
    sufficient max_allowed_depth). With leaves of several points the final plan could violate the
    marginals, because spreading a cell mass uniformly over its points is only correct for
    uniform point masses.

    The final plan is eps-optimal: cost <= OPT + M * eps * max_c with M the total mass.
    """

    def __init__(self, tree_X, tree_Y, mu_X, mu_Y, max_c=1.0, target_eps=None, min_eps=1e-12,
                 accuracy=1e-3, verbose=False):
        self.tree_X = tree_X
        self.tree_Y = tree_Y
        self.X_pts = self.tree_X.points
        self.Y_pts = self.tree_Y.points
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)

        self.max_c = float(max_c)
        total_mass = max(float(self.mu_X.sum()), 1.0)
        self.target_eps = float(target_eps) if target_eps is not None else accuracy / total_mass
        self.min_eps = min_eps
        self.verbose = verbose

        self.max_depth = max(self.tree_X.max_depth, self.tree_Y.max_depth)
        self.last_N_guess = []
        self.last_mu = None
        self.last_beta = None

        self._check_singleton_leaves()

    def _check_singleton_leaves(self):
        for name, tree in (("X", self.tree_X), ("Y", self.tree_Y)):
            leaves = tree.get_active_cells_at_depth(self.max_depth)
            bad = [c for c in leaves if len(c.point_indices) != 1]
            if bad:
                raise ValueError(
                    f"Tree {name} has {len(bad)} leaf cells with more than one point. The final level "
                    f"must be at point resolution. Build the trees with max_points_per_cell=1 and "
                    f"a larger max_allowed_depth.")

    def _build_coarsened_problem(self, depth):
        cells_X = self.tree_X.get_active_cells_at_depth(depth)
        cells_Y = self.tree_Y.get_active_cells_at_depth(depth)

        mu_X_hat = np.array([self.mu_X[c.point_indices].sum() for c in cells_X])
        mu_Y_hat = np.array([self.mu_Y[c.point_indices].sum() for c in cells_Y])
        X_pts_hat = np.array([self.X_pts[c.point_indices].mean(axis=0) for c in cells_X])
        Y_pts_hat = np.array([self.Y_pts[c.point_indices].mean(axis=0) for c in cells_Y])
        return X_pts_hat, Y_pts_hat, mu_X_hat, mu_Y_hat, cells_X, cells_Y

    @staticmethod
    def _induce_sparse_neighborhood(mu_hat_dict, cells_X_coarse, cells_Y_coarse,
                                    cells_X_fine, cells_Y_fine, tol=1e-9):
        """Children pairs of every coarse pair carrying mass.

        Feasible: the product split pi(a, b) = mu_hat(A, B) * m(a) * m(b) / (m(A) m(B)) is a
        coupling of the fine marginals supported on this set.
        """
        cell_to_idx_X = {cell: idx for idx, cell in enumerate(cells_X_fine)}
        cell_to_idx_Y = {cell: idx for idx, cell in enumerate(cells_Y_fine)}

        allowed = set()
        for i in mu_hat_dict:
            for j, mass in mu_hat_dict[i].items():
                if mass > tol:
                    pa, pb = cells_X_coarse[i], cells_Y_coarse[j]
                    ch_a = pa.children if pa.children else [pa]
                    ch_b = pb.children if pb.children else [pb]
                    for ca in ch_a:
                        for cb in ch_b:
                            if ca in cell_to_idx_X and cb in cell_to_idx_Y:
                                allowed.add((cell_to_idx_X[ca], cell_to_idx_Y[cb]))
        return sorted(allowed)

    def _log(self, msg):
        if self.verbose:
            sys.stderr.write(msg + "\n")

    def solve(self):
        # Depth 0: root against root (a single pair).
        cX_pts, cY_pts, c_mu_X, c_mu_Y, cX, cY = self._build_coarsened_problem(0)
        manager = EpsScalingManager(
            AuctionOT, X_pts=cX_pts, Y_pts=cY_pts, mu_X=c_mu_X, mu_Y=c_mu_Y,
            normalize=False, max_c=self.max_c, target_eps=self.target_eps,
            min_eps=self.min_eps)
        current_mu, _, _, final_beta = manager.solve()

        checker = ConsistencyChecker(self.tree_X, self.tree_Y, initial_sparse_N=[], max_c=self.max_c)

        for d in range(0, self.max_depth):
            fX_pts, fY_pts, f_mu_X, f_mu_Y, fX, fY = self._build_coarsened_problem(d + 1)
            N_guess = self._induce_sparse_neighborhood(current_mu, cX, cY, fX, fY)
            checker.N_set = set(N_guess)
            self._log(f"[Depth {d + 1}] initial induced edges: {len(N_guess)}")

            # warm start: children inherit the dual of their parent
            cell_to_idx_Y_coarse = {cell: idx for idx, cell in enumerate(cY)}
            beta_level = np.zeros(len(fY), dtype=float)
            for i, fine_cell in enumerate(fY):
                lookup = fine_cell.parent if fine_cell.parent in cell_to_idx_Y_coarse else fine_cell
                if lookup in cell_to_idx_Y_coarse:
                    beta_level[i] = final_beta[cell_to_idx_Y_coarse[lookup]]

            # cost the solver uses on this level: squared centroid distance / max_c
            def cost_fn(x, y, fX_pts=fX_pts, fY_pts=fY_pts):
                diff = fX_pts[x] - fY_pts[y]
                return float(np.dot(diff, diff)) / self.max_c

            iteration = 1
            while True:
                hybrid = EpsScalingManager(
                    AuctionOT, X_pts=fX_pts, Y_pts=fY_pts, mu_X=f_mu_X, mu_Y=f_mu_Y,
                    allowed_edges=N_guess, initial_beta=beta_level, normalize=False,
                    max_c=self.max_c, target_eps=self.target_eps, min_eps=self.min_eps)
                current_mu, _, _, final_beta = hybrid.solve()
                beta_level = final_beta
                eps = hybrid.target_eps

                # alpha(x) = min over the neighbourhood of (solver cost - beta), solver cost
                edges = np.asarray(N_guess, dtype=int)
                xs, ys = edges[:, 0], edges[:, 1]
                net = np.sum((fX_pts[xs] - fY_pts[ys]) ** 2, axis=1) / self.max_c - final_beta[ys]
                alpha = np.full(len(f_mu_X), np.inf)  # no edge: never prune
                np.minimum.at(alpha, xs, net)
                alpha_prime = alpha + eps

                new_edges = checker.run_consistency_check(
                    alpha_prime, final_beta, target_depth=d + 1, cost_fn=cost_fn)
                self._log(f"  -> loop {iteration}: checker added {len(new_edges)} edges, "
                          f"active {len(checker.N_set)}")
                iteration += 1
                if not new_edges:
                    break
                N_guess = sorted(checker.N_set)

            self.last_N_guess = N_guess
            cX, cY = fX, fY

        self.last_mu = current_mu
        self.last_beta = final_beta

        final_X = self.tree_X.get_active_cells_at_depth(self.max_depth)
        final_Y = self.tree_Y.get_active_cells_at_depth(self.max_depth)
        plan = []
        for i in current_mu:
            for j, mass in current_mu[i].items():
                if mass > 1e-9:
                    plan.append((final_X[i].point_indices[0], final_Y[j].point_indices[0], mass))
        return plan
