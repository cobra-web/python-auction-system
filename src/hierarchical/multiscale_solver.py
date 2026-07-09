import numpy as np
from src.core.ot_auction import AuctionOT, TOL
from src.utils.eps_scaling import EpsScalingManager
from src.hierarchical.consistency import ConsistencyChecker

class HierarchicalMultiscaleSolver:
    
    def __init__(self, tree_X, tree_Y, cost_matrix, mu_X, mu_Y):
        self.tree_X = tree_X
        self.tree_Y = tree_Y
        self.C = np.array(cost_matrix, dtype=float)
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)
        self.g = self.tree_X.g

    def _build_coarsened_problem(self, gen):
        cells_X = self.tree_X.generations[gen]
        cells_Y = self.tree_Y.generations[gen]

        num_a = len(cells_X)
        num_b = len(cells_Y)

        mu_X_hat = np.zeros(num_a, dtype=float)
        mu_Y_hat = np.zeros(num_b, dtype=float)

        for i, cell_a in enumerate(cells_X):
            mu_X_hat[i] = np.sum(self.mu_X[cell_a.point_indices])
        for j, cell_b in enumerate(cells_Y):
            mu_Y_hat[j] = np.sum(self.mu_Y[cell_b.point_indices])

        C_hat = np.zeros((num_a, num_b))
        for i, cell_a in enumerate(cells_X):
            for j, cell_b in enumerate(cells_Y):
                idx_a = cell_a.point_indices
                idx_b = cell_b.point_indices
                C_hat[i, j] = np.min(self.C[np.ix_(idx_a, idx_b)])

        return C_hat, mu_X_hat, mu_Y_hat

    def _induce_sparse_neighborhood(self, coarse_mu, gen_coarse):
        gen_fine = gen_coarse - 1
        cells_X_coarse = self.tree_X.generations[gen_coarse]
        cells_Y_coarse = self.tree_Y.generations[gen_coarse]
        cells_X_fine = self.tree_X.generations[gen_fine]
        cells_Y_fine = self.tree_Y.generations[gen_fine]
        
        cell_to_idx_X = {cell: idx for idx, cell in enumerate(cells_X_fine)}
        cell_to_idx_Y = {cell: idx for idx, cell in enumerate(cells_Y_fine)}
        
        allowed_edges = []
        for i in range(len(cells_X_coarse)):
            for j in range(len(cells_Y_coarse)):
                if coarse_mu[i, j] > 1e-9:
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
        coarsest_gen = self.g - 1
        print(f"Starting Hierarchical Multiscale Solve (depth g={self.g})")
        
        checker = ConsistencyChecker(self.tree_X, self.tree_Y, self.C, initial_sparse_N=[])

        C_hat, mu_X_hat, mu_Y_hat = self._build_coarsened_problem(coarsest_gen)
        manager = EpsScalingManager(AuctionOT, C_hat, mu_X=mu_X_hat, mu_Y=mu_Y_hat)
        current_mu, _, _, current_prices = manager.solve()

        for gen in range(coarsest_gen - 1, -1, -1):
            print(f"\n--- Refining to Generation {gen} ---")

            N_guess = self._induce_sparse_neighborhood(current_mu, gen + 1)
            C_fine, mu_X_fine, mu_Y_fine = self._build_coarsened_problem(gen)

            cells_Y_fine = self.tree_Y.generations[gen]
            cells_Y_coarse = self.tree_Y.generations[gen + 1]
            cell_to_idx_Y_coarse = {cell: idx for idx, cell in enumerate(cells_Y_coarse)}

            current_beta_for_level = np.zeros(len(cells_Y_fine), dtype=float)
            for i, fine_cell in enumerate(cells_Y_fine):
                parent_cell = fine_cell.parent
                if parent_cell in cell_to_idx_Y_coarse:
                    current_beta_for_level[i] = current_prices[cell_to_idx_Y_coarse[parent_cell]]

            checker.N_set = set(N_guess)
            checker.c_hat_cache.clear()

            consistency_iterations = 0
            while consistency_iterations < 100:
                consistency_iterations += 1

                hybrid_manager = EpsScalingManager(
                    AuctionOT, C_fine, mu_X=mu_X_fine, mu_Y=mu_Y_fine,
                    allowed_edges=N_guess,
                    initial_beta=current_beta_for_level,
                    target_eps=None
                )
                current_mu, total_cost, total_iters, final_prices = hybrid_manager.solve()
                current_beta_for_level = final_prices.copy()

                target_eps_absolute = hybrid_manager.target_eps_absolute

                # CRITICAL FIX: Invert positive auction prices to negative OT duals
                beta_dual = -final_prices

                alpha = np.full(len(mu_X_fine), np.inf, dtype=float)
                for x in range(len(mu_X_fine)):
                    valid_ys = [y for (x_prime, y) in N_guess if x_prime == x]
                    if len(valid_ys) > 0:
                        alpha[x] = np.min(C_fine[x, valid_ys] - beta_dual[valid_ys])

                alpha_prime = alpha + target_eps_absolute + 1e-9

                prev_len = len(checker.N_set)
                
                # Pass true OT duals (beta_dual) into consistency check
                checker.run_consistency_check(alpha_prime, beta_dual, target_gen=gen)
                added = len(checker.N_set) - prev_len
                N_guess = list(checker.N_set)

                if added == 0:
                    print(f"Generation {gen}: verified with {len(N_guess)} active pairs")
                    break
                else:
                    print(f"  [Consistency Iter {consistency_iterations}] Found {added} new edges")

                if len(N_guess) > len(mu_X_fine) * len(mu_Y_fine) * 0.5:
                    print(f"  WARNING: Neighborhood too dense; stopping expansion")
                    break

            self.last_N_guess = N_guess
            current_prices = final_prices

        print("\nReconstructing solution...")
        cells_X_fine = self.tree_X.generations[0]
        cells_Y_fine = self.tree_Y.generations[0]
        reconstructed_mu = np.zeros_like(self.C)

        for i, cell_a in enumerate(cells_X_fine):
            idx_a = cell_a.point_indices
            total_a = np.sum(self.mu_X[idx_a])
            if total_a < TOL: continue
            weights_a = self.mu_X[idx_a] / total_a

            for j, cell_b in enumerate(cells_Y_fine):
                cell_mass = current_mu[i, j]
                if cell_mass < TOL: continue
                idx_b = cell_b.point_indices
                total_b = np.sum(self.mu_Y[idx_b])
                if total_b < TOL: continue
                weights_b = self.mu_Y[idx_b] / total_b

                reconstructed_mu[np.ix_(idx_a, idx_b)] += cell_mass * np.outer(weights_a, weights_b)

        return reconstructed_mu
