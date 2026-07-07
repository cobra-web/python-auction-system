import numpy as np
from src.core.ot_auction import AuctionOT
from src.utils.eps_scaling import EpsScalingManager
from src.hierarchical.consistency import ConsistencyChecker


class HierarchicalMultiscaleSolver:
    def __init__(self, tree_X, tree_Y, cost_matrix, mu_X, mu_Y):
        self.tree_X = tree_X
        self.tree_Y = tree_Y
        self.C = cost_matrix
        self.mu_X = mu_X
        self.mu_Y = mu_Y

        assert self.tree_X.g == self.tree_Y.g, "Hierarchical partitions must have equal depth."
        self.g = self.tree_X.g

    def _build_coarsened_problem(self, gen):
        # Eq. 16
        cells_X = self.tree_X.generations[gen]
        cells_Y = self.tree_Y.generations[gen]

        num_a = len(cells_X)
        num_b = len(cells_Y)

        mu_X_hat = np.zeros(num_a, dtype=int)
        mu_Y_hat = np.zeros(num_b, dtype=int)

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

    def _induce_sparse_neighborhood(self, mu_hat, gen_coarse):
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
                if mu_hat[i, j] > 0:
                    parent_a = cells_X_coarse[i]
                    parent_b = cells_Y_coarse[j]

                    children_a = parent_a.children if (parent_a.children and parent_a.children[0] in cell_to_idx_X) else [parent_a]
                    children_b = parent_b.children if (parent_b.children and parent_b.children[0] in cell_to_idx_Y) else [parent_b]

                    for child_a in children_a:
                        for child_b in children_b:
                            if child_a in cell_to_idx_X and child_b in cell_to_idx_Y:
                                allowed_edges.append((cell_to_idx_X[child_a], cell_to_idx_Y[child_b]))

        return allowed_edges

    def solve(self):
        coarsest_gen = self.g - 1
        print(f"Starting Multiscale Solve. Root Generation: {coarsest_gen}")

        # One checker for the whole solve: cell ids are global and unique across
        # the tree, so c_hat_cache stays valid across generations instead of
        # being rebuilt from scratch at every refinement step.
        checker = ConsistencyChecker(self.tree_X, self.tree_Y, self.C, initial_sparse_N=[])

        C_fine, mu_X_fine, mu_Y_fine = self._build_coarsened_problem(coarsest_gen)

        manager = EpsScalingManager(AuctionOT, C_fine, mu_X=mu_X_fine, mu_Y=mu_Y_fine)
        current_mu, _, _, final_beta = manager.solve()
        final_alpha = None

        for gen in range(coarsest_gen - 1, -1, -1):
            print(f"\n--- Refining to Generation {gen} ---")

            N_guess = self._induce_sparse_neighborhood(current_mu, gen + 1)
            C_fine, mu_X_fine, mu_Y_fine = self._build_coarsened_problem(gen)

            checker.N_set = set(N_guess)
            checker.c_hat_cache = {k: v for k, v in checker.c_hat_cache.items()}  # keep cache, drop nothing

            prev_alpha = None
            prev_beta = None

            # Section 4.2
            while True:
                solve_kwargs = dict(mu_X=mu_X_fine, mu_Y=mu_Y_fine, allowed_edges=N_guess)
                if prev_alpha is not None:
                    # Warm-start: only new edges from the last expansion round
                    # need fresh bidding, the rest should already be near
                    # complementary-slack. This assumes AuctionOT/EpsScalingManager
                    # accept initial dual arrays - wire these kwargs to match
                    # whatever your AuctionOT constructor exposes.
                    solve_kwargs["initial_alpha"] = prev_alpha
                    solve_kwargs["initial_beta"] = prev_beta

                hybrid_manager = EpsScalingManager(AuctionOT, C_fine, **solve_kwargs)
                current_mu, total_cost, total_iters, final_beta = hybrid_manager.solve()

                alpha = np.zeros(len(mu_X_fine))
                for x in range(len(mu_X_fine)):
                    assigned_ys = np.where(current_mu[x] > 0)[0]
                    if len(assigned_ys) > 0:
                        alpha[x] = np.min(C_fine[x, assigned_ys] - final_beta[assigned_ys])

                target_eps = hybrid_manager.target_eps
                alpha_prime = alpha + target_eps

                rebid_candidates = checker.run_consistency_check(alpha_prime, final_beta, start_gen=gen)
                new_edges = [(x, y) for (x, y) in checker.N_set if x in rebid_candidates]
                # run_consistency_check already merged new_edges into checker.N_set;
                # N_guess needs to reflect that so the next EpsScalingManager call
                # actually bids over them.
                added = len(checker.N_set) - len(N_guess)
                N_guess = list(checker.N_set)

                if added == 0:
                    print(f"Generation {gen} solved and verified. Cost: {total_cost:.4f}")
                    break
                else:
                    print(f"  [Boundary Violation] Found {added} missing edges. Expanding neighborhood...")
                    prev_alpha, prev_beta = alpha, final_beta

            self.last_N_guess = N_guess

        print("\nMultiscale optimization complete. Reconstructing original array alignments...")

        # restore original matrix coordinates
        orig_idx_X = [cell.point_indices[0] for cell in self.tree_X.generations[0]]
        orig_idx_Y = [cell.point_indices[0] for cell in self.tree_Y.generations[0]]

        reconstructed_mu = np.zeros_like(self.C)
        for i, orig_x in enumerate(orig_idx_X):
            for j, orig_y in enumerate(orig_idx_Y):
                reconstructed_mu[orig_x, orig_y] = current_mu[i, j]

        return reconstructed_mu

