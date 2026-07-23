import sys
import numpy as np
from collections import defaultdict
from src.core.ot_auction import AuctionOT
from src.utils.eps_scaling import EpsScalingManager
from src.hierarchical.consistency import ConsistencyChecker

class HierarchicalMultiscaleSolver:
    def __init__(self, tree_X, tree_Y, mu_X, mu_Y, max_c=1.0, target_eps=None, min_eps=1e-4):
        self.tree_X = tree_X
        self.tree_Y = tree_Y
        self.X_pts = self.tree_X.points
        self.Y_pts = self.tree_Y.points
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)
        
        self.max_c = max_c
        self.target_eps = target_eps
        self.min_eps = min_eps

        self.max_depth = max(self.tree_X.max_depth, self.tree_Y.max_depth)
        self.last_N_guess = []

    def _build_coarsened_problem(self, depth):
        cells_X = self.tree_X.get_active_cells_at_depth(depth)
        cells_Y = self.tree_Y.get_active_cells_at_depth(depth)

        num_a = len(cells_X)
        num_b = len(cells_Y)

        mu_X_hat = np.zeros(num_a, dtype=float)
        mu_Y_hat = np.zeros(num_b, dtype=float)

        X_pts_hat = np.zeros((num_a, self.tree_X.dimensions))
        Y_pts_hat = np.zeros((num_b, self.tree_Y.dimensions))

        for i, cell_a in enumerate(cells_X):
            mu_X_hat[i] = np.sum(self.mu_X[cell_a.point_indices])
            X_pts_hat[i] = np.mean(self.X_pts[cell_a.point_indices], axis=0)

        for j, cell_b in enumerate(cells_Y):
            mu_Y_hat[j] = np.sum(self.mu_Y[cell_b.point_indices])
            Y_pts_hat[j] = np.mean(self.Y_pts[cell_b.point_indices], axis=0)

        return X_pts_hat, Y_pts_hat, mu_X_hat, mu_Y_hat, cells_X, cells_Y

    def _induce_sparse_neighborhood(self, mu_hat_dict, cells_X_coarse, cells_Y_coarse, cells_X_fine, cells_Y_fine):
        cell_to_idx_X = {cell: idx for idx, cell in enumerate(cells_X_fine)}
        cell_to_idx_Y = {cell: idx for idx, cell in enumerate(cells_Y_fine)}

        allowed_edges = []
        for i in mu_hat_dict:
            for j, mass in mu_hat_dict[i].items():
                if mass > 1e-6:
                    parent_a = cells_X_coarse[i]
                    parent_b = cells_Y_coarse[j]

                    children_a = parent_a.children if parent_a.children else [parent_a]
                    children_b = parent_b.children if parent_b.children else [parent_b]

                    for child_a in children_a:
                        for child_b in children_b:
                            if child_a in cell_to_idx_X and child_b in cell_to_idx_Y:
                                allowed_edges.append((cell_to_idx_X[child_a], cell_to_idx_Y[child_b]))

        return allowed_edges

    def solve(self):
        # Base case: Solve depth 0 (Root vs Root)
        cX_pts, cY_pts, c_mu_X, c_mu_Y, cX, cY = self._build_coarsened_problem(0)
        
        manager = EpsScalingManager(
            AuctionOT, X_pts=cX_pts, Y_pts=cY_pts, mu_X=c_mu_X, mu_Y=c_mu_Y, 
            normalize=False, max_c=self.max_c, min_eps=self.min_eps
        )
        current_mu, _, _, final_beta = manager.solve()

        # Instantiate checker with the shared global scale
        checker = ConsistencyChecker(self.tree_X, self.tree_Y, initial_sparse_N=[], max_c=self.max_c)

        # Top-down loop
        for d in range(0, self.max_depth):
            fX_pts, fY_pts, f_mu_X, f_mu_Y, fX, fY = self._build_coarsened_problem(d + 1)
            N_guess = self._induce_sparse_neighborhood(current_mu, cX, cY, fX, fY)

            checker.N_set = set(N_guess)
            sys.stderr.write(f"[Depth {d+1}] Initial induced edges: {len(N_guess)}\n")

            cell_to_idx_Y_coarse = {cell: idx for idx, cell in enumerate(cY)}
            current_beta_for_level = np.zeros(len(fY), dtype=float)

            for i, fine_cell in enumerate(fY):
                lookup_cell = fine_cell.parent if fine_cell.parent in cell_to_idx_Y_coarse else fine_cell
                if lookup_cell in cell_to_idx_Y_coarse:
                    parent_idx = cell_to_idx_Y_coarse[lookup_cell]
                    current_beta_for_level[i] = final_beta[parent_idx]

            iteration_count = 1
            current_start_eps = None #Loop 1 starts fresh at 0.5

            while True:
                is_final_level = (d + 1 == self.max_depth)
                level_target_eps = self.target_eps if is_final_level else None
                
                hybrid_manager = EpsScalingManager(
                    AuctionOT, X_pts=fX_pts, Y_pts=fY_pts, mu_X=f_mu_X, mu_Y=f_mu_Y, 
                    allowed_edges=N_guess,
                    initial_beta=current_beta_for_level,
                    normalize=False,
                    max_c=self.max_c,
                    target_eps=level_target_eps,
                    min_eps=self.min_eps
                    start_eps=current_start_eps
                )
                current_mu, total_cost, total_iters, final_beta = hybrid_manager.solve()
                current_beta_for_level = final_beta

                current_start_eps = hybrid_manager.target_eps

                ys_by_x = defaultdict(list)
                for xp, y in N_guess:
                    ys_by_x[xp].append(y)

                alpha = np.zeros(len(f_mu_X))
                for x in range(len(f_mu_X)):
                    valid_ys = ys_by_x[x]
                    if len(valid_ys) > 0:
                        # --- FIX: Normalize bounding box distances to match solver scale ---
                        min_dists = [checker._bbox_min_sq_dist(fX[x].bbox, fY[y].bbox) / self.max_c for y in valid_ys]
                        alpha[x] = np.min(np.array(min_dists) - final_beta[valid_ys])
                    else:
                        alpha[x] = 0.0

                target_eps = hybrid_manager.target_eps
                alpha_prime = alpha + target_eps

                prev_len = len(checker.N_set)
                checker.run_consistency_check(alpha_prime, final_beta, target_depth=d+1)
                added = len(checker.N_set) - prev_len
                N_guess = list(checker.N_set)
                
                sys.stderr.write(f"  -> Loop {iteration_count}: Checker added {added} edges. Total active: {len(N_guess)}\n")
                iteration_count += 1

                if added == 0:
                    break

            self.last_N_guess = N_guess
            cX, cY = fX, fY

        sparse_assignments = []
        final_X = self.tree_X.get_active_cells_at_depth(self.max_depth)
        final_Y = self.tree_Y.get_active_cells_at_depth(self.max_depth)
        
        for i in current_mu:
            for j, mass in current_mu[i].items():
                if mass > 1e-8:
                    x_indices = final_X[i].point_indices
                    y_indices = final_Y[j].point_indices
                    
                    mass_per_pair = mass / (len(x_indices) * len(y_indices))
                    for orig_x in x_indices:
                        for orig_y in y_indices:
                            sparse_assignments.append((orig_x, orig_y, mass_per_pair))

        return sparse_assignments
