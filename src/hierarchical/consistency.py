import numpy as np

class ConsistencyChecker:
    """
    Executes the Hierarchical Consistency Check Phase as defined in Section 4.
    Ensures global optimality while operating on a sparse neighborhood subset.
    """
    def __init__(self, tree_X, tree_Y, cost_matrix, initial_sparse_N):
        self.tree_X = tree_X
        self.tree_Y = tree_Y
        self.C = cost_matrix
        
        # Track the active sparse neighborhood \hat{N} using a set for O(1) lookups
        self.N_set = set(initial_sparse_N)
        
        # Storage for hierarchical extensions
        self.alpha_prime_hat = {}
        self.beta_hat = {}

    def _compute_extensions(self, alpha_prime, beta):
        """
        Eq. 13: Computes \hat{\alpha}' and \hat{\beta} bottom-up from the leaves to the root.
        """
        # Propagate alpha_prime_hat up tree_X
        for gen in range(self.tree_X.g):
            for cell in self.tree_X.generations[gen]:
                if gen == 0: # Leaves (Singletons)
                    x = cell.point_indices[0]
                    self.alpha_prime_hat[cell.id] = alpha_prime[x]
                else:        # Parents
                    self.alpha_prime_hat[cell.id] = max(
                        self.alpha_prime_hat[child.id] for child in cell.children
                    )
                    
        # Propagate beta_hat up tree_Y
        for gen in range(self.tree_Y.g):
            for cell in self.tree_Y.generations[gen]:
                if gen == 0: # Leaves (Singletons)
                    y = cell.point_indices[0]
                    self.beta_hat[cell.id] = beta[y]
                else:        # Parents
                    self.beta_hat[cell.id] = max(
                        self.beta_hat[child.id] for child in cell.children
                    )

    def _c_hat(self, cell_a, cell_b):
        """
        Eq. 14: \hat{c}(a,b) = min_{x \in a, y \in b} c(x,y)
        Note: For very large problems (scenario 'P2H-LB' in the paper), this explicit 
        subgrid minimum should be replaced by bounding-box lower bounds to save time.
        """
        idx_a = cell_a.point_indices
        idx_b = cell_b.point_indices
        return np.min(self.C[np.ix_(idx_a, idx_b)])

    def run_consistency_check(self, alpha_prime, beta, start_gen=None):
        """
        Checks the inequalities. If violated, recursively dives finer.
        Returns a list of 'x' elements that need to re-bid.
        """
        self._compute_extensions(alpha_prime, beta)
        
        # Usually start checking at the coarsest level (root)
        if start_gen is None:
            start_gen = self.tree_X.g - 1
            
        new_edges = []
        for cell_a in self.tree_X.generations[start_gen]:
            for cell_b in self.tree_Y.generations[start_gen]:
                new_edges.extend(self._check_recursive(cell_a, cell_b))
                
        # Update sparse neighborhood and return the affected 'x' nodes
        rebid_candidates = set()
        for x, y in new_edges:
            self.N_set.add((x, y))
            rebid_candidates.add(x)
            
        return list(rebid_candidates)

    def _check_recursive(self, cell_a, cell_b):
        """
        The core recursive logic checking \hat{c} - \hat{\beta} >= \hat{\alpha}'
        """
        c_hat_val = self._c_hat(cell_a, cell_b)
        a_prime_hat = self.alpha_prime_hat[cell_a.id]
        b_hat = self.beta_hat[cell_b.id]
        
        # Condition from text: If true, no deeper check is needed[cite: 195].
        if c_hat_val - b_hat >= a_prime_hat:
            return []
            
        # If violated and we are at the finest scale (Generation 0) [cite: 197]
        if cell_a.generation == 0 and cell_b.generation == 0:
            x = cell_a.point_indices[0]
            y = cell_b.point_indices[0]
            
            # If not already in our sparse set, we found a missing link! [cite: 198]
            if (x, y) not in self.N_set:
                return [(x, y)]
            return []
            
        # If violated and we are at a coarse scale, recurse into children 
        found_edges = []
        for child_a in cell_a.children:
            for child_b in cell_b.children:
                found_edges.extend(self._check_recursive(child_a, child_b))
                
        return found_edges
