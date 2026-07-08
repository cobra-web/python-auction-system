import numpy as np
from src.core.ot_auction import AuctionOT
from src.utils.eps_scaling import EpsScalingManager
from src.hierarchical.consistency import ConsistencyChecker


class HierarchicalMultiscaleSolver:
    def __init__(self, tree_X, tree_Y, cost_matrix, mu_X, mu_Y):
        self.tree_X = tree_X
        self.tree_Y = tree_Y
        self.C = np.array(cost_matrix, dtype=float)  # STORE ORIGINAL
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)

        assert self.tree_X.g == self.tree_Y.g, "Hierarchical partitions must have equal depth."
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

        # Compute coarse-level costs: minimum cost within each coarse cell
        # This ensures costs are lower bounds (tighter pruning)
        C_hat = np.zeros((num_a, num_b))
        for i, cell_a in enumerate(cells_X):
            for j, cell_b in enumerate(cells_Y):
                idx_a = cell_a.point_indices
                idx_b = cell_b.point_indices
                C_hat[i, j] = np.min(self.C[np.ix_(idx_a, idx_b)])

        return C_hat, mu_X_hat, mu_Y_hat

    def solve(self):
        coarsest_gen = self.g - 1
        print(f"Starting Hierarchical Multiscale Solve (depth g={self.g})")
        print(f"Root generation (coarsest): {coarsest_gen}")

        # Initialize consistency checker for all generations
        checker = ConsistencyChecker(self.tree_X, self.tree_Y, self.C, initial_sparse_N=[])

        # ===== COARSEST LEVEL: Solve OT on coarsest partitions =====
        C_hat, mu_X_hat, mu_Y_hat = self._build_coarsened_problem(coarsest_gen)

        # Use ε-scaling manager on coarsest level
        manager = EpsScalingManager(AuctionOT, C_hat, mu_X=mu_X_hat, mu_Y=mu_Y_hat)
        current_mu, _, _, final_beta = manager.solve()
        current_beta = final_beta

        # ===== REFINEMENT LOOP: Refine downward through generations =====
        for gen in range(coarsest_gen - 1, -1, -1):
            print(f"\n--- Refining to Generation {gen} ---")

            # Lift coarse solution to candidate edges at fine level
            N_guess = self._induce_sparse_neighborhood(current_mu, gen + 1)
            C_fine, mu_X_fine, mu_Y_fine = self._build_coarsened_problem(gen)

            # Initialize β for fine level using parent's β values
            cells_Y_fine = self.tree_Y.generations[gen]
            cells_Y_coarse = self.tree_Y.generations[gen + 1]
            cell_to_idx_Y_coarse = {cell: idx for idx, cell in enumerate(cells_Y_coarse)}

            current_beta_for_level = np.zeros(len(cells_Y_fine), dtype=float)
            for i, fine_cell in enumerate(cells_Y_fine):
                parent_cell = fine_cell.parent
                if parent_cell in cell_to_idx_Y_coarse:
                    parent_idx = cell_to_idx_Y_coarse[parent_cell]
                    current_beta_for_level[i] = current_beta[parent_idx]

            # Update consistency checker for this level
            checker.N_set = set(N_guess)
            checker.c_hat_cache.clear()

            # ===== CONSISTENCY LOOP (Section 4.2 of SS13) =====
            consistency_iterations = 0
            while consistency_iterations < 100:
                consistency_iterations += 1

                # Solve OT at this level with warm-start
                hybrid_manager = EpsScalingManager(
                    AuctionOT, C_fine, mu_X=mu_X_fine, mu_Y=mu_Y_fine,
                    allowed_edges=N_guess,
                    initial_beta=current_beta_for_level,
                    target_eps=None
                )
                current_mu, total_cost, total_iters, final_beta = hybrid_manager.solve()
                current_beta_for_level = final_beta.copy()

                # Extract dual certificates from solution
                alpha = np.full(len(mu_X_fine), np.inf, dtype=float)
                for x in range(len(mu_X_fine)):
                    valid_ys = [y for (x_prime, y) in N_guess if x_prime == x]
                    if len(valid_ys) > 0:
                        alpha[x] = np.min(C_fine[x, valid_ys] - final_beta[valid_ys])

                # Add ε-slack for consistency check
                target_eps = hybrid_manager.target_eps
                alpha_prime = alpha + target_eps

                # ===== CONSISTENCY CHECK: Expand N_guess if needed =====
                prev_len = len(checker.N_set)
                checker.run_consistency_check(alpha_prime, final_beta, target_gen=gen)
                added = len(checker.N_set) - prev_len
                N_guess = list(checker.N_set)

                if added == 0:
                    print(f"Generation {gen}: verified with {len(N_guess)} active pairs "
                          f"({100.0*len(N_guess)/(len(mu_X_fine)*len(mu_Y_fine)):.1f}% density)")
                    break
                else:
                    print(f"  [Consistency Iter {consistency_iterations}] Found {added} new edges "
                          f"(N_guess: {prev_len} → {len(N_guess)})")

                # Safety: stop if neighborhood grows too dense
                if len(N_guess) > len(mu_X_fine) * len(mu_Y_fine) * 0.5:
                    print(f"  WARNING: Neighborhood at {100.0*len(N_guess)/(len(mu_X_fine)*len(mu_Y_fine)):.1f}% density; stopping expansion")
                    break

            self.last_N_guess = N_guess
            current_beta = final_beta

        print("\nMultiscale optimization complete. Reconstructing solution at finest scale...")

        # Reconstruct solution in original (finest-scale) coordinates
        orig_idx_X = [cell.point_indices[0] for cell in self.tree_X.generations[0]]
        orig_idx_Y = [cell.point_indices[0] for cell in self.tree_Y.generations[0]]

        reconstructed_mu = np.zeros_like(self.C)
        for i, orig_x in enumerate(orig_idx_X):
            for j, orig_y in enumerate(orig_idx_Y):
                reconstructed_mu[orig_x, orig_y] = current_mu[i, j]

        # CRITICAL: Return coupling in original coordinates
        # Cost will be computed by benchmark using original C
        return reconstructed_mu
