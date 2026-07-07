import numpy as np

class ConsistencyChecker:
    def __init__(self, tree_X, tree_Y, cost_matrix, initial_sparse_N):
        self.tree_X = tree_X
        self.tree_Y = tree_Y
        self.C = cost_matrix

        self.N_set = set(initial_sparse_N)

        self.alpha_prime_hat = {}
        self.beta_hat = {}

        self.c_hat_cache = {}

    def _compute_extensions(self, alpha_prime, beta, target_gen):
        # Eq 13, generalized: base case is cells at target_gen (indexed by
        # position within that generation's cell list), not always true
        # leaves. alpha_prime/beta are sized to that generation's cell count.
        cell_to_idx_X = {cell: idx for idx, cell in enumerate(self.tree_X.generations[target_gen])}
        cell_to_idx_Y = {cell: idx for idx, cell in enumerate(self.tree_Y.generations[target_gen])}

        for gen in range(target_gen, self.tree_X.g):
            for cell in self.tree_X.generations[gen]:
                if gen == target_gen:
                    self.alpha_prime_hat[cell.id] = alpha_prime[cell_to_idx_X[cell]]
                else:
                    self.alpha_prime_hat[cell.id] = max(
                        self.alpha_prime_hat[child.id] for child in cell.children
                    )

        for gen in range(target_gen, self.tree_Y.g):
            for cell in self.tree_Y.generations[gen]:
                if gen == target_gen:
                    self.beta_hat[cell.id] = beta[cell_to_idx_Y[cell]]
                else:
                    self.beta_hat[cell.id] = max(
                        self.beta_hat[child.id] for child in cell.children
                    )

        self._cell_to_idx_X = cell_to_idx_X
        self._cell_to_idx_Y = cell_to_idx_Y

    def _c_hat(self, cell_a, cell_b):
        cache_key = (cell_a.id, cell_b.id)
        if cache_key in self.c_hat_cache:
            return self.c_hat_cache[cache_key]

        if cell_a.generation == 0 and cell_b.generation == 0:
            idx_a = cell_a.point_indices
            idx_b = cell_b.point_indices
            val = np.min(self.C[np.ix_(idx_a, idx_b)])
        else:
            val = float('inf')
            for child_a in cell_a.children:
                for child_b in cell_b.children:
                    val = min(val, self._c_hat(child_a, child_b))

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

        rebid_candidates = set()
        for x, y in new_edges:
            self.N_set.add((x, y))
            rebid_candidates.add(x)

        return list(rebid_candidates)

    def _check_recursive(self, cell_a, cell_b, alpha_prime, beta, target_gen):
        c_hat_val = self._c_hat(cell_a, cell_b)
        a_prime_hat = self.alpha_prime_hat[cell_a.id]
        b_hat = self.beta_hat[cell_b.id]

        if c_hat_val - b_hat >= a_prime_hat:
            return []

        if cell_a.generation == target_gen and cell_b.generation == target_gen:
            x = self._cell_to_idx_X[cell_a]
            y = self._cell_to_idx_Y[cell_b]
            found_edges = []
            if (x, y) not in self.N_set:
                if c_hat_val - beta[y] < alpha_prime[x]:
                    found_edges.append((x, y))
            return found_edges

        found_edges = []
        for child_a in cell_a.children:
            for child_b in cell_b.children:
                found_edges.extend(self._check_recursive(child_a, child_b, alpha_prime, beta, target_gen))

        return found_edges
