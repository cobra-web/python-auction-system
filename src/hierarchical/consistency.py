import numpy as np

class ConsistencyChecker:
    def __init__(self, tree_X, tree_Y, initial_sparse_N=None, max_c=1.0):
        self.tree_X = tree_X
        self.tree_Y = tree_Y
        self.max_c = float(max_c)
        self.N_set = set(initial_sparse_N) if initial_sparse_N is not None else set()
        self.c_hat_cache = {}

    def _compute_extensions(self, alpha_prime, beta, target_depth):
        """Compute extended dual variables upwards from the target active depth"""
        cells_X_target = self.tree_X.get_active_cells_at_depth(target_depth)
        cells_Y_target = self.tree_Y.get_active_cells_at_depth(target_depth)

        cell_to_idx_X = {cell: idx for idx, cell in enumerate(cells_X_target)}
        cell_to_idx_Y = {cell: idx for idx, cell in enumerate(cells_Y_target)}

        self.alpha_prime_hat = {}
        self.beta_hat = {}

        for cell in cells_X_target:
            self.alpha_prime_hat[cell.id] = alpha_prime[cell_to_idx_X[cell]]
            
        for cell in cells_Y_target:
            self.beta_hat[cell.id] = beta[cell_to_idx_Y[cell]]

        # Recursively map max dual variables up to the root
        def _propagate_up_X(node):
            if node.id in self.alpha_prime_hat:
                return self.alpha_prime_hat[node.id]
            if not node.children:
                return -float('inf') 
            val = max(_propagate_up_X(child) for child in node.children)
            self.alpha_prime_hat[node.id] = val
            return val

        def _propagate_up_Y(node):
            if node.id in self.beta_hat:
                return self.beta_hat[node.id]
            if not node.children:
                return -float('inf')
            val = max(_propagate_up_Y(child) for child in node.children)
            self.beta_hat[node.id] = val
            return val

        _propagate_up_X(self.tree_X.cells[0]) 
        _propagate_up_Y(self.tree_Y.cells[0])

        self._cell_to_idx_X = cell_to_idx_X
        self._cell_to_idx_Y = cell_to_idx_Y

    def _bbox_min_sq_dist(self, bbox_a, bbox_b):
        min_a, max_a = bbox_a
        min_b, max_b = bbox_b
        dist = 0.0
        for i in range(len(min_a)):
            d = max(0.0, min_a[i] - max_b[i], min_b[i] - max_a[i])
            dist += d * d
        return dist

    def _c_hat(self, cell_a, cell_b):
        cache_key = (cell_a.id, cell_b.id)
        if cache_key in self.c_hat_cache:
            return self.c_hat_cache[cache_key]
        
        # --- FIX: Divide by self.max_c to match normalized solver units ---
        val = self._bbox_min_sq_dist(cell_a.bbox, cell_b.bbox) / self.max_c
        self.c_hat_cache[cache_key] = val
        return val

    def run_consistency_check(self, alpha_prime, beta, target_depth):
        self._compute_extensions(alpha_prime, beta, target_depth)

        # Always initiate check from the root (depth 0)
        new_edges = self._check_recursive(self.tree_X.cells[0], self.tree_Y.cells[0], alpha_prime, beta, target_depth)

        for x, y in new_edges:
            self.N_set.add((x, y))

        return new_edges

    def _check_recursive(self, cell_a, cell_b, alpha_prime, beta, target_depth):
        c_hat_val = self._c_hat(cell_a, cell_b)
        a_prime_hat = self.alpha_prime_hat[cell_a.id]
        b_hat = self.beta_hat[cell_b.id]

        if c_hat_val - b_hat >= a_prime_hat:
            return []

        cell_a_is_terminal = (cell_a.depth == target_depth) or (not cell_a.children)
        cell_b_is_terminal = (cell_b.depth == target_depth) or (not cell_b.children)

        if cell_a_is_terminal and cell_b_is_terminal:
            x = self._cell_to_idx_X[cell_a]
            y = self._cell_to_idx_Y[cell_b]
            found_edges = []

            if (x, y) not in self.N_set:
                if c_hat_val - beta[y] < alpha_prime[x]:
                    found_edges.append((x, y))

            return found_edges

        found_edges = []
        if not cell_a_is_terminal and not cell_b_is_terminal:
            for child_a in cell_a.children:
                for child_b in cell_b.children:
                    found_edges.extend(self._check_recursive(child_a, child_b, alpha_prime, beta, target_depth))
        elif not cell_a_is_terminal:
            for child_a in cell_a.children:
                found_edges.extend(self._check_recursive(child_a, cell_b, alpha_prime, beta, target_depth))
        elif not cell_b_is_terminal:
            for child_b in cell_b.children:
                found_edges.extend(self._check_recursive(cell_a, child_b, alpha_prime, beta, target_depth))

        return found_edges
