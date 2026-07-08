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
        """Build coarsened OT problem at generation gen (Eq. 16)"""
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
                # Minimum cost within each coarse cell (lower bound)
                C_hat[i, j] = np.min(self.C[np.ix_(idx_a, idx_b)])

        return C_hat, mu_X_hat, mu_Y_hat

    def _induce_sparse_neighborhood(self, mu_hat, gen_coarse):
        """Lift solution from coarse level to fine level (Eq. 17)"""
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
                if mu_hat[i, j] > 1e-9:  # Only lift active coarse edges
                    parent_a = cells_X_coarse[i]
                    parent_b = cells_Y_coarse[j]

                    # Get children (or self if leaf)
                    children_a = parent_a.children if (parent_a.children and len(parent_a.children) > 0) else [parent_a]
                    children_b = parent_b.children if (parent_b.children and len(parent_b.children) > 0) else [parent_b]

                    for child_a in children_a:
                        for child_b in children_b:
                            if child_a in cell_to_idx_X and child_b in cell_to_idx_Y:
                                allowed_edges.append((cell_to_idx_X[child_a], cell_to_idx_Y[child_b]))

        return allowed_edges

    def solve(self):
        """Main hierarchical solver (SS13 Algorithm 1)"""
        coarsest_gen = self.g - 1
        print(f"Starting Multiscale Solve. Root Generation: {coarsest_gen}")

        # Initialize consistency checker
        checker = ConsistencyChecker(self.tree_X, self.tree_Y, self.C, initial_sparse_N=[])

        # ===== COARSEST LEVEL: Solve OT on coarsest partitions =====
        C_hat, mu_X_hat, mu_Y_hat = self._build_coarsened_problem(coarsest_gen)

        manager = EpsScalingManager(AuctionOT, C_hat, mu_X=mu_X_hat, mu_Y=mu_Y_hat)
        current_mu, _, _, final_beta = manager.solve()

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
                    # Initialize with parent's dual value (conservative warm start)
                    current_beta_for_level[i] = final_beta[parent_idx]

            # Update consistency checker for this level
            checker.N_set = set(N_guess)
            checker.c_hat_cache.clear()  # Clear cache for new level

            # ===== SECTION 4.2: Consistency loop at current level =====
            consistency_iterations = 0
            while consistency_iterations < 100:  # Safeguard against infinite loops
                consistency_iterations += 1

                # Solve OT with warm-start β and allowed edges N_guess
                hybrid_manager = EpsScalingManager(
                    AuctionOT, C_fine, mu_X=mu_X_fine, mu_Y=mu_Y_fine,
                    allowed_edges=N_guess,
                    initial_beta=current_beta_for_level,
                    target_eps=None  # Let manager compute appropriate target_eps
                )
                current_mu, total_cost, total_iters, final_beta = hybrid_manager.solve()
                current_beta_for_level = final_beta.copy()

                # Extract dual certificates α' from solution
                # CRITICAL FIX: α(x) = min_y [C(x,y) - β(y)] over ALL allowed edges
                alpha = np.full(len(mu_X_fine), np.inf, dtype=float)
                for x in range(len(mu_X_fine)):
                    valid_ys = [y for (x_prime, y) in N_guess if x_prime == x]
                    if len(valid_ys) > 0:
                        alpha[x] = np.min(C_fine[x, valid_ys] - final_beta[valid_ys])
                    # else: alpha[x] remains inf (node was not allowed to bid)

                # Add ε-slack for consistency check (Eq. 4.5-4.6 in SS13)
                target_eps = hybrid_manager.target_eps
                alpha_prime = alpha + target_eps

                # ===== CONSISTENCY CHECK: Expand N_guess if needed =====
                prev_len = len(checker.N_set)
                checker.run_consistency_check(alpha_prime, final_beta, target_gen=gen)
                added = len(checker.N_set) - prev_len
                N_guess = list(checker.N_set)

                if added == 0:
                    print(f"Generation {gen} solved and verified in {consistency_iterations} consistency iterations. Cost: {total_cost:.4f}")
                    break
                else:
                    print(f"  [Consistency Iteration {consistency_iterations}] Found {added} missing edges. Expanding from {prev_len} to {len(N_guess)}...")

                # Safety: if neighborhood is growing too large, stop expanding
                if len(N_guess) > len(mu_X_fine) * len(mu_Y_fine) * 0.5:
                    print(f"  WARNING: Neighborhood grew to {len(N_guess) / (len(mu_X_fine) * len(mu_Y_fine)) * 100:.1f}% density. Stopping expansion.")
                    break

            self.last_N_guess = N_guess

        print("\nMultiscale optimization complete. Reconstructing original array alignments...")

        # Reconstruct solution in original coordinates
        orig_idx_X = [cell.point_indices[0] for cell in self.tree_X.generations[0]]
        orig_idx_Y = [cell.point_indices[0] for cell in self.tree_Y.generations[0]]

        reconstructed_mu = np.zeros_like(self.C)
        for i, orig_x in enumerate(orig_idx_X):
            for j, orig_y in enumerate(orig_idx_Y):
                reconstructed_mu[orig_x, orig_y] = current_mu[i, j]

        return reconstructed_mu

