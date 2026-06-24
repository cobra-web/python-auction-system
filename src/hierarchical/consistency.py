import numpy as np

class ConsistencyChecker:
    def __init__(self, tree_X, tree_Y, cost_matrix, initial_sparse_N):
        self.tree_X = tree_X
        self.tree_Y = tree_Y
        self.C = cost_matrix
        
        #track N hat using a set for O(1) lookups
        self.N_set = set(initial_sparse_N)
        
        self.alpha_prime_hat = {}
        self.beta_hat = {}

    def _compute_extensions(self, alpha_prime, beta):
        #Eq 13
        #propagate alpha_prime_hat up tree_X
        for gen in range(self.tree_X.g):
            for cell in self.tree_X.generations[gen]:
                if gen == 0: #leaves
                    x = cell.point_indices[0]
                    self.alpha_prime_hat[cell.id] = alpha_prime[x]
                else:        #parents
                    self.alpha_prime_hat[cell.id] = max(
                        self.alpha_prime_hat[child.id] for child in cell.children
                    )
                    
        #propagate beta_hat up tree_Y
        for gen in range(self.tree_Y.g):
            for cell in self.tree_Y.generations[gen]:
                if gen == 0: #leaves
                    y = cell.point_indices[0]
                    self.beta_hat[cell.id] = beta[y]
                else:        #parents
                    self.beta_hat[cell.id] = max(
                        self.beta_hat[child.id] for child in cell.children
                    )

    def _c_hat(self, cell_a, cell_b):
        #Eq 14
        idx_a = cell_a.point_indices
        idx_b = cell_b.point_indices
        return np.min(self.C[np.ix_(idx_a, idx_b)])

    def run_consistency_check(self, alpha_prime, beta, start_gen=None):
        self._compute_extensions(alpha_prime, beta)
        
        #starts checking at the coarsest level/root
        if start_gen is None:
            start_gen = self.tree_X.g - 1
            
        new_edges = []
        for cell_a in self.tree_X.generations[start_gen]:
            for cell_b in self.tree_Y.generations[start_gen]:
                new_edges.extend(self._check_recursive(cell_a, cell_b))
                
        #updates sparse neighborhood and returns the affected 'x' nodes
        rebid_candidates = set()
        for x, y in new_edges:
            self.N_set.add((x, y))
            rebid_candidates.add(x)
            
        return list(rebid_candidates)

    def _check_recursive(self, cell_a, cell_b):
        c_hat_val = self._c_hat(cell_a, cell_b)
        a_prime_hat = self.alpha_prime_hat[cell_a.id]
        b_hat = self.beta_hat[cell_b.id]
        
        if c_hat_val - b_hat >= a_prime_hat:
            return []
            
        #if violated and we are at gen 0
        if cell_a.generation == 0 and cell_b.generation == 0:
            x = cell_a.point_indices[0]
            y = cell_b.point_indices[0]
            
            #if not already in our sparse set, we found a missing link
            if (x, y) not in self.N_set:
                return [(x, y)]
            return []
            
        #if violated and we are at a coarse scale, recurse into children 
        found_edges = []
        for child_a in cell_a.children:
            for child_b in cell_b.children:
                found_edges.extend(self._check_recursive(child_a, child_b))
                
        return found_edges
