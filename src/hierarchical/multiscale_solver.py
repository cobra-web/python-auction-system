import numpy as np
from src.core.ot_auction import AuctionOT
from src.utils.eps_scaling import EpsScalingManager

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
        """
        Eq. 16: Constructs the coarsened Optimal Transport problem at generation 'n'.
        Aggregates mass allocations and establishes the bounding costs.
        """
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
        """
        Projects the active coupling supports down to the child cells 
        at the next finer generation level. Fixes the KeyError tracking bug.
        """
        gen_fine = gen_coarse - 1
        cells_X_coarse = self.tree_X.generations[gen_coarse]
        cells_Y_coarse = self.tree_Y.generations[gen_coarse]
        
        cells_X_fine = self.tree_X.generations[gen_fine]
        cells_Y_fine = self.tree_Y.generations[gen_fine]
        
        # Create an inverse lookup dictionary to map a child cell object 
        # to its exact row/column index in the fine matrix sequence
        cell_to_idx_X = {cell: idx for idx, cell in enumerate(cells_X_fine)}
        cell_to_idx_Y = {cell: idx for idx, cell in enumerate(cells_Y_fine)}
        
        allowed_edges = []
        
        for i in range(len(cells_X_coarse)):
            for j in range(len(cells_Y_coarse)):
                if mu_hat[i, j] > 0:
                    parent_a = cells_X_coarse[i]
                    parent_b = cells_Y_coarse[j]
                    
                    # Allow all children of parent_a to trade with children of parent_b
                    for child_a in parent_a.children:
                        for child_b in parent_b.children:
                            idx_a = cell_to_idx_X[child_a]
                            idx_b = cell_to_idx_Y[child_b]
                            allowed_edges.append((idx_a, idx_b))
                            
        return allowed_edges

    def solve(self):
        """
        Master loop descending down the structural layers.
        """
        coarsest_gen = self.g - 1
        print(f"Starting Multiscale Solve. Root Generation: {coarsest_gen}")
        
        # 1. Base initialization step at the widest coarsened layer
        C_fine, mu_X_fine, mu_Y_fine = self._build_coarsened_problem(coarsest_gen)
        
        manager = EpsScalingManager(
            AuctionOT, 
            C_fine, 
            mu_X=mu_X_fine, 
            mu_Y=mu_Y_fine
        )
        current_mu, _, _ = manager.solve()
        
        # 2. Refine downwards sequentially
        for gen in range(coarsest_gen - 1, -1, -1):
            print(f"\n--- Refining to Generation {gen} ---")
            
            # Map neighbor positions matching the current dimensional metrics
            N_guess = self._induce_sparse_neighborhood(current_mu, gen + 1)
            
            # Always build coarsened structures based on the node configurations
            C_fine, mu_X_fine, mu_Y_fine = self._build_coarsened_problem(gen)
                
            hybrid_manager = EpsScalingManager(
                AuctionOT,
                C_fine,
                mu_X=mu_X_fine,
                mu_Y=mu_Y_fine,
                allowed_edges=N_guess
            )
            current_mu, total_cost, total_iters = hybrid_manager.solve()
            print(f"Generation {gen} solved. Cost: {total_cost:.4f} ({total_iters} total iterations)")
            
        print("\nMultiscale optimization complete. Reconstructing original array alignments...")
        
        # 3. Restore the original matrix coordinates
        # Generation 0 cells correspond to unique spatial points. 
        # Map indices back to the input vector layout.
        orig_idx_X = [cell.point_indices[0] for cell in self.tree_X.generations[0]]
        orig_idx_Y = [cell.point_indices[0] for cell in self.tree_Y.generations[0]]
        
        reconstructed_mu = np.zeros_like(self.C)
        for i, orig_x in enumerate(orig_idx_X):
            for j, orig_y in enumerate(orig_idx_Y):
                reconstructed_mu[orig_x, orig_y] = current_mu[i, j]
                
        return reconstructed_mu
