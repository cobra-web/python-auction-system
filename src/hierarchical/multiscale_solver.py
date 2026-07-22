import numpy as np
from src.core.ot_auction import AuctionOT
from src.utils.eps_scaling import EpsScalingManager
from src.hierarchical.consistency import ConsistencyChecker

class HierarchicalMultiscaleSolver:
    def __init__(self, tree_X, tree_Y, mu_X, mu_Y):
        self.tree_X = tree_X
        self.tree_Y = tree_Y
        self.X_pts = self.tree_X.points
        self.Y_pts = self.tree_Y.points
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)

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
            # Use the true mathematical center of the data points
            X_pts_hat[i] = np.mean(self.X_pts[cell_a.point_indices], axis=0)

        for j, cell_b in enumerate(cells_Y):
            mu_Y_hat[j] = np.sum(self.mu_Y[cell_b.point_indices])
            # Use the true mathematical center of the data points
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

                    # If the coarse cell is a leaf, it acts as its own child at the fine level
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
            AuctionOT, X_pts=cX_pts, Y_pts=cY_pts, mu_X=c_mu_X, mu_Y=c_mu_Y, normalize=False
        )
        current_mu, _, _, final_beta = manager.solve()

        checker = ConsistencyChecker(self.tree_X, self.tree_Y, initial_sparse_N=[])

        # Top-down loop
        for d in range(0, self.max_depth):
            fX_pts, fY_pts, f_mu_X, f_mu_Y, fX, fY = self._build_coarsened_problem(d + 1)
            N_guess = self._induce_sparse_neighborhood(current_mu, cX, cY, fX, fY)

            checker.N_set = set(N_guess)

            cell_to_idx_Y_coarse = {cell: idx for idx, cell in enumerate(cY)}
            current_beta_for_level = np.zeros(len(fY), dtype=float)

            for i, fine_cell in enumerate(fY):
                lookup_cell = fine_cell.parent if fine_cell.parent in cell_to_idx_Y_coarse else fine_cell
                if lookup_cell in cell_to_idx_Y_coarse:
                    parent_idx = cell_to_idx_Y_coarse[lookup_cell]
                    current_beta_for_level[i] = final_beta[parent_idx]

            while True:
                hybrid_manager = EpsScalingManager(
                    AuctionOT, X_pts=fX_pts, Y_pts=fY_pts, mu_X=f_mu_X, mu_Y=f_mu_Y, 
                    allowed_edges=N_guess,
                    initial_beta=current_beta_for_level,
                    normalize=False
                )
                current_mu, total_cost, total_iters, final_beta = hybrid_manager.solve()
                current_beta_for_level = final_beta

                alpha = np.zeros(len(f_mu_X))
                for x in range(len(f_mu_X)):
                    valid_ys = [y for (x_prime, y) in N_guess if x_prime == x]
                    if len(valid_ys) > 0:
                        min_dists = [checker._bbox_min_sq_dist(fX[x].bbox, fY[y].bbox) for y in valid_ys]
                        alpha[x] = np.min(np.array(min_dists) - final_beta[valid_ys])
                    else:
                        alpha[x] = 0.0

                target_eps = hybrid_manager.target_eps
                alpha_prime = alpha + target_eps

                prev_len = len(checker.N_set)
                checker.run_consistency_check(alpha_prime, final_beta, target_depth=d+1)
                added = len(checker.N_set) - prev_len
                N_guess = list(checker.N_set)

                if added == 0:
                    break

            self.last_N_guess = N_guess
            
            # Setup coarse variables for the next level down
            cX, cY = fX, fY

        # Reconstruction to sparse assignment list
        sparse_assignments = []
        # The final assignment is at max_depth
        final_X = self.tree_X.get_active_cells_at_depth(self.max_depth)
        final_Y = self.tree_Y.get_active_cells_at_depth(self.max_depth)
        
        for i in current_mu:
            for j, mass in current_mu[i].items():
                if mass > 0:
                    # Note: You still need logic to distribute mass if len(point_indices) > 1
                    # This safely grabs the first point as a placeholder.
                    orig_x = final_X[i].point_indices[0] 
                    orig_y = final_Y[j].point_indices[0]
                    sparse_assignments.append((orig_x, orig_y, mass))

        return sparse_assignments
