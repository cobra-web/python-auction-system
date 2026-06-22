from src.core.ot_auction import AuctionOT
from src.utils.eps_scaling import EpsScalingManager
from src.hierarchical.consistency import ConsistencyChecker
import numpy as np


class HierarchicalMultiscaleSolver:
    def __init__(self, tree_X, tree_Y, cost_matrix, mu_X, mu_Y):
        self.tree_X = tree_X
        self.tree_Y = tree_Y
        self.C = cost_matrix
        self.mu_X = mu_X
        self.mu_Y = mu_Y
        
        # Ensure trees have the same depth (g)
        assert self.tree_X.g == self.tree_Y.g, "Hierarchical partitions must have equal depth."
        self.g = self.tree_X.g

    def _build_coarsened_problem(self, gen):
        """
        Eq. 16: Constructs the coarsened Optimal Transport problem at generation 'n'.
        Aggregates mass and computes \hat{c}(a,b).
        """
        cells_X = self.tree_X.generations[gen]
        cells_Y = self.tree_Y.generations[gen]
        
        num_a = len(cells_X)
        num_b = len(cells_Y)
        
        # Aggregate mass distributions for the coarsened problem
        mu_X_hat = np.zeros(num_a, dtype=int)
        mu_Y_hat = np.zeros(num_b, dtype=int)
        
        for i, cell_a in enumerate(cells_X):
            mu_X_hat[i] = np.sum(self.mu_X[cell_a.point_indices])
            
        for j, cell_b in enumerate(cells_Y):
            mu_Y_hat[j] = np.sum(self.mu_Y[cell_b.point_indices])
            
        # Coarsened cost matrix \hat{c}
        C_hat = np.zeros((num_a, num_b))
        for i, cell_a in enumerate(cells_X):
            for j, cell_b in enumerate(cells_Y):
                # Using Eq. 14 minimum bounding logic
                idx_a = cell_a.point_indices
                idx_b = cell_b.point_indices
                C_hat[i, j] = np.min(self.C[np.ix_(idx_a, idx_b)])
                
        return C_hat, mu_X_hat, mu_Y_hat

    def _induce_sparse_neighborhood(self, mu_hat, gen_coarse):
        """
        "The support of \hat{\mu}' is picked as initial guess for \hat{N} 
        when solving the refined problem." 
        Projects the coarse solution down one generation to their children.
        """
        cells_X = self.tree_X.generations[gen_coarse]
        cells_Y = self.tree_Y.generations[gen_coarse]
        
        N_fine_guess = set()
        
        # Iterate over the coarse coupling matrix
        for i in range(len(cells_X)):
            for j in range(len(cells_Y)):
                if mu_hat[i, j] > 0:
                    # If mass was moved between cell_a and cell_b, 
                    # allow their children to trade at the finer scale.
                    parent_a = cells_X[i]
                    parent_b = cells_Y[j]
                    
                    for child_a in parent_a.children:
                        for child_b in parent_b.children:
                            # We track the tree node IDs for the solver
                            N_fine_guess.add((child_a.id, child_b.id))
                            
        return list(N_fine_guess)

    # --- 2. THE NEW SOLVE METHOD REPLACES THE OLD ONE ENTIRELY ---
    def solve(self, start_generation):
        """
        The master multiscale loop in perfect harmony.
        Starts at a coarse scale, solves densely (WITH e-scaling), 
        and recursively refines down to gen 0.
        """
        print(f"Starting Multiscale Solve at Coarse Generation {start_generation}")
        
        # Direct dense solution at the coarsest scale
        C_coarse, mu_X_coarse, mu_Y_coarse = self._build_coarsened_problem(start_generation)
        
        # HARMONY POINT 1: Use the EpsScalingManager for the dense coarse solve
        coarse_manager = EpsScalingManager(
            AuctionOT, 
            C_coarse, 
            mu_X=mu_X_coarse, 
            mu_Y=mu_Y_coarse
        )
        current_mu, _, _ = coarse_manager.solve()
        
        # Recursively solve at finer scales
        for gen in range(start_generation - 1, -1, -1):
            print(f"Refining to Generation {gen}...")
            
            # Extract initial guess \hat{N} from the coarser generation
            N_guess = self._induce_sparse_neighborhood(current_mu, gen + 1)
            
            # Build the specific problem for this generation
            C_gen, mu_X_gen, mu_Y_gen = self._build_coarsened_problem(gen)
            
            # HARMONY POINT 2: Wrap the Hybrid execution in the scaling manager
            hybrid_manager = EpsScalingManager(
                AuctionOT, 
                C_gen, 
                mu_X=mu_X_gen, 
                mu_Y=mu_Y_gen,
                allowed_edges=N_guess,           
                consistency_checker=ConsistencyChecker(self.tree_X, self.tree_Y, C_gen, N_guess) 
            )
            
            current_mu, _, _ = hybrid_manager.solve()
            
            print(f"Generation {gen} solved. Sparse subset expanded as needed.")
            
        print("Finest scale (Generation 0) solved. Global optimality guaranteed.")
        return current_mu
