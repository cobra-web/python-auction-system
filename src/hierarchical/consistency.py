import numpy as np

class ConsistencyChecker:
    def __init__(self, tree_X, tree_Y, initial_sparse_N):
        self.tree_X = tree_X
        self.tree_Y = tree_Y
        self.N_set = set(initial_sparse_N)
        self.alpha_prime_hat = {}
        self.beta_hat = {}
        self.c_hat_cache = {}

    def _compute_extensions(self, alpha_prime, beta, target_gen):
        cell_to_idx_X = {cell: idx for idx, cell in enumerate(self.tree_X.generations[target_gen])}
        cell_to_idx_Y = {cell: idx for idx, cell in enumerate(self.tree_Y.generations[target_gen])}

        for gen in range(target_gen, self.tree_X.g):
            for cell in self.tree_X.generations[gen]:
                if gen == target_gen:
                    self.alpha_prime_hat[cell.id] = alpha_prime[cell_to_idx_X[cell]]
                else:
                    if cell.children:
                        self.alpha_prime_hat[cell.id] = max(
                            self.alpha_prime_hat[child.id] for child in cell.children
                        )

        for gen in range(target_gen, self.tree_Y.g):
            for cell in self.tree_Y.generations[gen]:
                if gen == target_gen:
                    self.beta_hat[cell.id] = beta[cell_to_idx_Y[cell]]
                else:
                    if cell.children:
                        self.beta_hat[cell.id] = max(
                            self.beta_hat[child.id] for child in cell.children
                        )

        self._cell_to_idx_X = cell_to_idx_X
        self._cell_to_idx_Y = cell_to_idx_Y

    def _bbox_min_sq_dist(self, bbox_a, bbox_b):
        """Calculates exact min squared euclidean distance between two bboxes"""
        min_a, max_a = bbox_a
        min_b, max_b = bbox_b
        dist = 0.0
        for i in range(len(min_a)):
            d = max(0.0, min_a[i] - max_b[i], min_b[i] - max_a[i])
            dist += d * d
        return dist

    def _c_hat(self, cell_a, cell_b):
        """Compute extended cost (Eq. 12) via bounding boxes dynamically"""
        cache_key = (cell_a.id, cell_b.id)
        if cache_key in self.c_hat_cache:
            return self.c_hat_cache[cache_key]

        val = self._bbox_min_sq_dist(cell_a.bbox, cell_b.bbox)
        self.c_hat_cache[cache_key] = val
        return val

    def run_consistency_check(self, alpha_prime, beta, target_gen, start_gen=None):
        self._compute_extensions(alpha_prime, beta, target_gen)

        if start_gen is None:
            start_gen = self.tree_X.g - 1

        new_edges = []
        for cell_a in self.tree_X.generations[start_gen]:
            for cell_b in self.tree_Y.generations[start_gen]:
                new_edges.extend(self._check_recursive(cell_a, cell_b, alpha_prime, beta, target_gen))

        for x, y in new_edges:
            self.N_set.add((x, y))

        return new_edges

    def _check_recursive(self, cell_a, cell_b, alpha_prime, beta, target_gen):
        c_hat_val = self._c_hat(cell_a, cell_b)
        a_prime_hat = self.alpha_prime_hat[cell_a.id]
        b_hat = self.beta_hat[cell_b.id]

        if c_hat_val - b_hat >= a_prime_hat:
            return []

        cell_a_is_terminal = (cell_a.generation == target_gen) or (not cell_a.children)
        cell_b_is_terminal = (cell_b.generation == target_gen) or (not cell_b.children)

        if cell_a_is_terminal and cell_b_is_terminal:
            x = self._cell_to_idx_X[cell_a]
            y = self._cell_to_idx_Y[cell_b]
            found_edges = []

            if (x, y) not in self.N_set:
                if c_hat_val - beta[y] < alpha_prime[x]:
                    found_edges.append((x, y))

            return found_edges

        found_edges = []
        if cell_a.children and cell_b.children:
            for child_a in cell_a.children:
                for child_b in cell_b.children:
                    found_edges.extend(self._check_recursive(child_a, child_b, alpha_prime, beta, target_gen))
        elif cell_a.children:
            for child_a in cell_a.children:
                found_edges.extend(self._check_recursive(child_a, cell_b, alpha_prime, beta, target_gen))
        elif cell_b.children:
            for child_b in cell_b.children:
                found_edges.extend(self._check_recursive(cell_a, child_b, alpha_prime, beta, target_gen))

        return found_edges
