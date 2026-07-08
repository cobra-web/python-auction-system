import numpy as np
from src.core.ot_auction import AuctionOT, TOL
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

    def _induce_sparse_neighborhood(self, coarse_mu, gen_coarse):
        gen_fine = gen_coarse - 1
        
        cells_X_coarse = self.tree_X.generations[gen_coarse]
        cells_Y_coarse = self.tree_Y.generations[gen_coarse]
        
        cells_X_fine = self.tree_X.generations[gen_fine]
        cells_Y_fine = self.tree_Y.generations[gen_fine]
        
        # Create mapping from fine-level cell to index
        cell_to_idx_X = {cell: idx for idx, cell in enumerate(cells_X_fine)}
        cell_to_idx_Y = {cell: idx for idx, cell in enumerate(cells_Y_fine)}
        
        allowed_edges = []
        
        # Iterate over all coarse-level pairs
        for i in range(len(cells_X_coarse)):
            for j in range(len(cells_Y_coarse)):
                # Only expand pairs with nonzero mass in the coarse solution
                if coarse_mu[i, j] > 1e-9:
                    parent_a = cells_X_coarse[i]
                    parent_b = cells_Y_coarse[j]
                    
                    # Get children of coarse cells (or self if no children)
                    if parent_a.children and len(parent_a.children) > 0:
                        children_a = parent_a.children
                    else:
                        children_a = [parent_a]
                    
                    if parent_b.children and len(parent_b.children) > 0:
                        children_b = parent_b.children
                    else:
                        children_b = [parent_b]
                    
                    # Expand to all combinations of children at fine level
                    for child_a in children_a:
                        for child_b in children_b:
                            # Check that children are actually in the fine level
                            if child_a in cell_to_idx_X and child_b in cell_to_idx_Y:
                                fine_x = cell_to_idx_X[child_a]
                                fine_y = cell_to_idx_Y[child_b]
                                allowed_edges.append((fine_x, fine_y))
        
        return allowed_edges


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
        
        # FIX: Speichere beta der gröbsten Ebene unskaliert (Original-Metrik)
        max_c_coarsest = np.max(np.abs(C_hat)) if np.max(np.abs(C_hat)) > 0 else 1.0
        current_beta = final_beta * max_c_coarsest

        # ===== REFINEMENT LOOP: Refine downward through generations =====
        for gen in range(coarsest_gen - 1, -1, -1):
            print(f"\n--- Refining to Generation {gen} ---")

            # Lift coarse solution to candidate edges at fine level
            N_guess = self._induce_sparse_neighborhood(current_mu, gen + 1)
            C_fine, mu_X_fine, mu_Y_fine = self._build_coarsened_problem(gen)
            
            # FIX: Bestimme den Normierungsfaktor der aktuellen feinen Ebene
            max_c_fine = np.max(np.abs(C_fine)) if np.max(np.abs(C_fine)) > 0 else 1.0

            # Initialize β for fine level using parent's β values
            cells_Y_fine = self.tree_Y.generations[gen]
            cells_Y_coarse = self.tree_Y.generations[gen + 1]
            cell_to_idx_Y_coarse = {cell: idx for idx, cell in enumerate(cells_Y_coarse)}

            current_beta_for_level_unscaled = np.zeros(len(cells_Y_fine), dtype=float)
            for i, fine_cell in enumerate(cells_Y_fine):
                parent_cell = fine_cell.parent
                if parent_cell in cell_to_idx_Y_coarse:
                    parent_idx = cell_to_idx_Y_coarse[parent_cell]
                    # Parent's dual-Wert liegt in unskalierter Metrik vor
                    current_beta_for_level_unscaled[i] = current_beta[parent_idx]

            # FIX: Skaliere beta für den EpsScalingManager dieser Ebene herunter
            current_beta_for_level = current_beta_for_level_unscaled / max_c_fine

            # Update consistency checker for this level
            checker.N_set = set(N_guess)
            checker.c_hat_cache.clear()

            # ===== CONSISTENCY LOOP (Section 4.2 of SS13) =====
            consistency_iterations = 0
            while consistency_iterations < 100:
                consistency_iterations += 1

                # Solve OT at this level with warm-start (erwartet normiertes beta)
                hybrid_manager = EpsScalingManager(
                    AuctionOT, C_fine, mu_X=mu_X_fine, mu_Y=mu_Y_fine,
                    allowed_edges=N_guess,
                    initial_beta=current_beta_for_level,
                    target_eps=None
                )
                current_mu, total_cost, total_iters, final_beta = hybrid_manager.solve()
                current_beta_for_level = final_beta.copy()

                # FIX: Rekonstruiere unnormierte Variablen für den Konsistenz-Checker
                final_beta_unnormalized = final_beta * max_c_fine
                target_eps_unnormalized = hybrid_manager.target_eps * max_c_fine

                # Extract dual certificates using unnormalized coordinates
                alpha = np.full(len(mu_X_fine), np.inf, dtype=float)
                for x in range(len(mu_X_fine)):
                    valid_ys = [y for (x_prime, y) in N_guess if x_prime == x]
                    if len(valid_ys) > 0:
                        alpha[x] = np.min(C_fine[x, valid_ys] - final_beta_unnormalized[valid_ys])

                # Add ε-slack (alles im unnormierten Raum)
                alpha_prime = alpha + target_eps_unnormalized

                # ===== CONSISTENCY CHECK: Expand N_guess if needed =====
                prev_len = len(checker.N_set)
                
                # FIX: Übergabe des unnormierten final_beta passend zu alpha_prime und checker.C
                checker.run_consistency_check(alpha_prime, final_beta_unnormalized, target_gen=gen)
                added = len(checker.N_set) - prev_len
                N_guess = list(checker.N_set)

                if added == 0:
                    print(f"Generation {gen}: verified with {len(N_guess)} active pairs "
                          f"({100.0*len(N_guess)/(len(mu_X_fine)*len(mu_Y_fine)):.1f}% density)")
                    break
                else:
                    print(f"  [Consistency Iter {consistency_iterations}] Found {added} new edges "
                          f"(N_guess: {prev_len} -> {len(N_guess)})")

                # Safety: stop if neighborhood grows too dense
                if len(N_guess) > len(mu_X_fine) * len(mu_Y_fine) * 0.5:
                    print(f"  WARNING: Neighborhood at {100.0*len(N_guess)/(len(mu_X_fine)*len(mu_Y_fine)):.1f}% density; stopping expansion")
                    break

            self.last_N_guess = N_guess
            # FIX: Speichere das unnormierte beta für die nächste Iteration der äußeren Schleife
            current_beta = final_beta * max_c_fine

        print("\nMultiscale optimization complete. Reconstructing solution at finest scale...")

        # Reconstruct solution in original (finest-scale) coordinates.
        cells_X_fine = self.tree_X.generations[0]
        cells_Y_fine = self.tree_Y.generations[0]

        reconstructed_mu = np.zeros_like(self.C)

        for i, cell_a in enumerate(cells_X_fine):
            idx_a = cell_a.point_indices
            total_a = np.sum(self.mu_X[idx_a])
            if total_a < TOL:
                continue
            weights_a = self.mu_X[idx_a] / total_a

            for j, cell_b in enumerate(cells_Y_fine):
                cell_mass = current_mu[i, j]
                if cell_mass < TOL:
                    continue

                idx_b = cell_b.point_indices
                total_b = np.sum(self.mu_Y[idx_b])
                if total_b < TOL:
                    continue
                weights_b = self.mu_Y[idx_b] / total_b

                reconstructed_mu[np.ix_(idx_a, idx_b)] += cell_mass * np.outer(weights_a, weights_b)

        return reconstructed_mu
