import numpy as np
import copy
from src.core.ot_auction import AuctionOT
from src.utils.eps_scaling import EpsScalingManager
from src.hierarchical.consistency import ConsistencyChecker

class HierarchicalMultiscaleSolver:
    def __init__(self, tree_X, tree_Y, cost_matrix, mu_X, mu_Y):
        self.tree_X = tree_X
        self.tree_Y = tree_Y
        self.C_raw = np.array(cost_matrix, dtype=float)
        
        self.max_c = np.max(self.C_raw)
        self.C = self.C_raw / (self.max_c if self.max_c > 0 else 1.0)
        
        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)

        self._align_tree_depths()
        self.g = self.tree_X.g
        
        self.last_N_guess = []

    def _align_tree_depths(self):
        max_g = max(self.tree_X.g, self.tree_Y.g)
        
        for tree in [self.tree_X, self.tree_Y]:
            while tree.g < max_g:
                current_depth = tree.g
                leaf_generation = tree.generations[-1]
                next_generation = []
                
                for cell in leaf_generation:
                    virtual_child = copy.copy(cell)
                    virtual_child.generation = current_depth
                    virtual_child.parent = cell
                    virtual_child.children = []
                    
                    cell.children = [virtual_child]
                    next_generation.append(virtual_child)
                
                tree.generations.append(next_generation)
                tree.g += 1

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
                if mu_hat[i, j] > 1e-6:
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
        
        checker = ConsistencyChecker(self.tree_X, self.tree_Y, self.C, initial_sparse_N=[])

        C_fine, mu_X_fine, mu_Y_fine = self._build_coarsened_problem(coarsest_gen)

        manager = EpsScalingManager(
            AuctionOT, C_fine, mu_X=mu_X_fine, mu_Y=mu_Y_fine, normalize=False
        )
        current_mu, _, _, final_beta = manager.solve()

        for gen in range(coarsest_gen - 1, -1, -1):
            N_guess = self._induce_sparse_neighborhood(current_mu, gen + 1)
            C_fine, mu_X_fine, mu_Y_fine = self._build_coarsened_problem(gen)

            checker.N_set = set(N_guess)

            cells_Y_fine = self.tree_Y.generations[gen]
            cells_Y_coarse = self.tree_Y.generations[gen + 1]
            cell_to_idx_Y_coarse = {cell: idx for idx, cell in enumerate(cells_Y_coarse)}

            current_beta_for_level = np.zeros(len(cells_Y_fine), dtype=float)

            for i, fine_cell in enumerate(cells_Y_fine):
                parent_cell = fine_cell.parent
                if parent_cell in cell_to_idx_Y_coarse:
                    parent_idx = cell_to_idx_Y_coarse[parent_cell]
                    current_beta_for_level[i] = final_beta[parent_idx]

            while True:
                hybrid_manager = EpsScalingManager(
                    AuctionOT, C_fine, mu_X=mu_X_fine, mu_Y=mu_Y_fine, 
                    allowed_edges=N_guess,
                    initial_beta=current_beta_for_level,
                    normalize=False
                )
                current_mu, total_cost, total_iters, final_beta = hybrid_manager.solve()
                current_beta_for_level = final_beta

                alpha = np.zeros(len(mu_X_fine))
                for x in range(len(mu_X_fine)):
                    valid_ys = [y for (x_prime, y) in N_guess if x_prime == x]
                    if len(valid_ys) > 0:
                        alpha[x] = np.min(C_fine[x, valid_ys] - final_beta[valid_ys])
                    else:
                        alpha[x] = 0.0

                target_eps = hybrid_manager.target_eps
                alpha_prime = alpha + target_eps

                prev_len = len(checker.N_set)
                checker.run_consistency_check(alpha_prime, final_beta, target_gen=gen)
                added = len(checker.N_set) - prev_len
                N_guess = list(checker.N_set)

                if added == 0:
                    break

            self.last_N_guess = N_guess

        reconstructed_mu = np.zeros_like(self.C_raw)
        
        cells_X_finest = self.tree_X.generations[0]
        cells_Y_finest = self.tree_Y.generations[0]
        
        for i, cell_x in enumerate(cells_X_finest):
            for j, cell_y in enumerate(cells_Y_finest):
                if current_mu[i, j] > 0:
                    assert len(cell_x.point_indices) == 1, f"Cell X {i} has {len(cell_x.point_indices)} points!"
                    assert len(cell_y.point_indices) == 1, f"Cell Y {j} has {len(cell_y.point_indices)} points!"
                    
                    orig_x = cell_x.point_indices[0]
                    orig_y = cell_y.point_indices[0]
                    reconstructed_mu[orig_x, orig_y] = current_mu[i, j]

        return reconstructed_mu
