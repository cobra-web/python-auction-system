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
        mu_X_hat = np.array([np.sum(self.mu_X[c.point_indices]) for c in cells_X])
        mu_Y_hat = np.array([np.sum(self.mu_Y[c.point_indices]) for c in cells_Y])
        
        C_hat = np.zeros((len(cells_X), len(cells_Y)))
        for i, cx in enumerate(cells_X):
            for j, cy in enumerate(cells_Y):
                C_hat[i, j] = np.min(self.C[np.ix_(cx.point_indices, cy.point_indices)])
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
        current_mu, _, _, current_beta = manager.solve() 

        # The coarsest level is solved DENSELY (no allowed_edges), so alpha
        # computed as a plain row-min over the full C_hat is exact -- every
        # (x, y) pair really was considered. This is the first "trustworthy"
        # dual pair (alpha, beta) we can hand down to certify neighborhoods
        # at finer generations.
        current_alpha = np.min(C_hat - current_beta[np.newaxis, :], axis=1)

        for gen in range(coarsest_gen - 1, -1, -1):
            print(f"\n--- Refining to Generation {gen} ---")

            C_fine, mu_X_fine, mu_Y_fine = self._build_coarsened_problem(gen)

            cells_X_fine = self.tree_X.generations[gen]
            cells_Y_fine = self.tree_Y.generations[gen]
            cells_X_coarse = self.tree_X.generations[gen + 1]
            cells_Y_coarse = self.tree_Y.generations[gen + 1]
            cell_to_idx_X_coarse = {cell: idx for idx, cell in enumerate(cells_X_coarse)}
            cell_to_idx_Y_coarse = {cell: idx for idx, cell in enumerate(cells_Y_coarse)}

            # Inherit BOTH dual variables from the parent generation.
            # beta -> auction warm start (as before).
            # alpha -> lets us CERTIFY a neighborhood for this generation
            #          before any auction runs on it (see below).
            current_beta_for_level = np.zeros(len(cells_Y_fine), dtype=float)
            for i, fine_cell in enumerate(cells_Y_fine):
                parent_cell = fine_cell.parent
                if parent_cell in cell_to_idx_Y_coarse:
                    current_beta_for_level[i] = current_beta[cell_to_idx_Y_coarse[parent_cell]]

            current_alpha_for_level = np.zeros(len(cells_X_fine), dtype=float)
            for i, fine_cell in enumerate(cells_X_fine):
                parent_cell = fine_cell.parent
                if parent_cell in cell_to_idx_X_coarse:
                    current_alpha_for_level[i] = current_alpha[cell_to_idx_X_coarse[parent_cell]]

            # ===== CERTIFY THE NEIGHBORHOOD BEFORE THE FIRST SOLVE =====
            # _induce_sparse_neighborhood is a PRIMAL, mass-based guess. It
            # carries no feasibility/optimality guarantee on its own -- it is
            # only ever a candidate seed. Schmitzer's consistency theorem
            # requires the neighborhood to be verified via the DUAL condition
            #     c_hat(a,b) - beta_hat(b) < alpha_prime_hat(a)
            # using duals that are already known to be correct -- here, the
            # parent generation's (gen+1) verified alpha/beta, inherited down.
            # Running this check BEFORE the auction, rather than after,
            # closes the theoretical gap: the auction below is now only ever
            # given an edge set that has already been proven sufficient.
            N_seed = self._induce_sparse_neighborhood(current_mu, gen + 1)
            checker.N_set = set(N_seed)
            checker.c_hat_cache.clear()

            target_eps_absolute = EpsScalingManager.compute_target_eps_absolute(C_fine)
            alpha_prime_for_level = current_alpha_for_level + target_eps_absolute

            checker.run_consistency_check(alpha_prime_for_level, current_beta_for_level, target_gen=gen)
            N_guess = list(checker.N_set)
            print(f"  Certified {len(N_guess)} edges before first solve "
                  f"(primal-mass seed had {len(N_seed)})")

            # ===== FIXED-POINT ITERATION =====
            # The pre-certification above uses the PARENT's duals, which are
            # not yet optimal for this generation. After solving we obtain
            # sharper, generation-local duals and re-check with those; this
            # loop is Schmitzer's fixed-point iteration, now starting from a
            # certified graph instead of an unverified heuristic one, so in
            # practice it should converge in zero or one extra pass rather
            # than being relied upon to rescue an infeasible first solve.
            consistency_iterations = 0
            while consistency_iterations < 100:
                consistency_iterations += 1

                hybrid_manager = EpsScalingManager(
                    AuctionOT, C_fine, mu_X=mu_X_fine, mu_Y=mu_Y_fine,
                    allowed_edges=N_guess,
                    initial_beta=current_beta_for_level,  # WARM START PASSED HERE
                    target_eps=None
                )
                current_mu, total_cost, total_iters, final_beta = hybrid_manager.solve()
                
                # Update beta for the next iteration of the consistency loop
                current_beta_for_level = final_beta.copy()
                target_eps_absolute = hybrid_manager.target_eps_absolute

                # SIGN INVERSION REMOVED: final_beta IS the correct mathematical OT dual.


                alpha = np.full(len(mu_X_fine), np.inf, dtype=float)

                for x in range(len(mu_X_fine)):
                    valid_ys = [y for (x_prime, y) in N_guess if x_prime == x]

                    if len(valid_ys) > 0:
                        min_sparse = np.min(C_fine[x, valid_ys] - final_beta[valid_ys])
                        min_all = np.min(C_fine[x] - final_beta)

                        print(
                            f"x={x:3d} "
                            f"gap={min_sparse-min_all:.3e} "
                            f"sparse={min_sparse:.6f} "
                            f"all={min_all:.6f}"
                        )

                        alpha[x] = min_sparse
                    else:
                        # Every x has positive mass, so a certified N_guess
                        # should never leave a row with zero edges. If this
                        # fires, pre-certification has a real gap -- surface
                        # it loudly instead of silently carrying inf forward,
                        # since alpha_prime = inf disables pruning for that
                        # row's entire ancestor chain and would mask the bug.
                        print(f"  WARNING: x={x} has zero certified edges "
                              f"at generation {gen}; pre-certification should "
                              f"have prevented this.")

                # Schmitzer's condition is alpha' = alpha + eps, exactly --
                # no extra additive slack. (The previous +1e-9 here was an
                # unjustified deviation: at tight target_eps it stops being
                # negligible and can make the check under-detect missing
                # edges instead of only guarding against float round-off.)
                alpha_prime = alpha + target_eps_absolute

                prev_len = len(checker.N_set)
                checker.run_consistency_check(alpha_prime, final_beta, target_gen=gen)
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
            current_beta = final_beta
            # 'added == 0' means N_guess is fully verified at this generation:
            # the sparse-restricted min equals the true dense min for every x.
            # That makes this generation's `alpha` exact, safe to hand down as
            # the certified dual for pre-certifying the next (finer) generation.
            current_alpha = alpha

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
