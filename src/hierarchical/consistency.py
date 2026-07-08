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
        """Compute extended dual variables (Eq. 13) from target_gen upward"""
        cell_to_idx_X = {cell: idx for idx, cell in enumerate(self.tree_X.generations[target_gen])}
        cell_to_idx_Y = {cell: idx for idx, cell in enumerate(self.tree_Y.generations[target_gen])}

        # Extend α' upward: α'̂(a) = max over descendants of α'(leaf)
        for gen in range(target_gen, self.tree_X.g):
            for cell in self.tree_X.generations[gen]:
                if gen == target_gen:
                    self.alpha_prime_hat[cell.id] = alpha_prime[cell_to_idx_X[cell]]
                else:
                    # Extend: take max over all children (upward)
                    if cell.children:
                        self.alpha_prime_hat[cell.id] = max(
                            self.alpha_prime_hat[child.id] for child in cell.children
                        )

        # Extend β upward: β̂(b) = max over descendants of β(leaf)
        for gen in range(target_gen, self.tree_Y.g):
            for cell in self.tree_Y.generations[gen]:
                if gen == target_gen:
                    self.beta_hat[cell.id] = beta[cell_to_idx_Y[cell]]
                else:
                    # Extend: take max over all children (upward)
                    if cell.children:
                        self.beta_hat[cell.id] = max(
                            self.beta_hat[child.id] for child in cell.children
                        )

        self._cell_to_idx_X = cell_to_idx_X
        self._cell_to_idx_Y = cell_to_idx_Y

    def _c_hat(self, cell_a, cell_b):
        """Compute extended cost (Eq. 12): minimum cost within coarse cells"""
        cache_key = (cell_a.id, cell_b.id)
        if cache_key in self.c_hat_cache:
            return self.c_hat_cache[cache_key]

        if not cell_a.children and not cell_b.children:
            # Both are true leaves in the tree (regardless of nominal generation
            # number -- a branch may stop splitting before generation 0 and get
            # replicated downward). Use their actual point sets directly.
            idx_a = cell_a.point_indices
            idx_b = cell_b.point_indices
            val = np.min(self.C[np.ix_(idx_a, idx_b)])
        else:
            # Coarser level: extend downward (min over all descendant pairs)
            val = float('inf')
            if cell_a.children and cell_b.children:
                for child_a in cell_a.children:
                    for child_b in cell_b.children:
                        val = min(val, self._c_hat(child_a, child_b))
            elif cell_a.children:
                for child_a in cell_a.children:
                    val = min(val, self._c_hat(child_a, cell_b))
            elif cell_b.children:
                for child_b in cell_b.children:
                    val = min(val, self._c_hat(cell_a, child_b))

        self.c_hat_cache[cache_key] = val
        return val

    def run_consistency_check(self, alpha_prime, beta, target_gen, start_gen=None):
        """
        Run consistency check (SS13 Section 4.1-4.2):
        Find pairs (x,y) at target_gen where slackness is violated.
        """
        self._compute_extensions(alpha_prime, beta, target_gen)

        if start_gen is None:
            start_gen = self.tree_X.g - 1

        new_edges = []
        # Start from coarsest level and recursively refine
        for cell_a in self.tree_X.generations[start_gen]:
            for cell_b in self.tree_Y.generations[start_gen]:
                new_edges.extend(self._check_recursive(cell_a, cell_b, alpha_prime, beta, target_gen))

        # Add discovered edges to N
        for x, y in new_edges:
            self.N_set.add((x, y))

        return new_edges

    def _check_recursive(self, cell_a, cell_b, alpha_prime, beta, target_gen):
        """
        Recursive consistency check (SS13 Eq. 4.5-4.6):
        At coarser levels, prune if ĉ(a,b) - β̂(b) ≥ α'̂(a)
        At target level, check individual pairs if ĉ(a,b) - β(b) < α'(a)
        """
        c_hat_val = self._c_hat(cell_a, cell_b)
        a_prime_hat = self.alpha_prime_hat[cell_a.id]
        b_hat = self.beta_hat[cell_b.id]

        # === PRUNING RULE: If cost exceeds price at coarse level, skip entire subtree ===
        if c_hat_val - b_hat >= a_prime_hat:
            return []

        # === BASE CASE: reached target generation, OR this branch stopped
        # splitting earlier (childless) and can't be refined any further ===
        cell_a_is_terminal = (cell_a.generation == target_gen) or (not cell_a.children)
        cell_b_is_terminal = (cell_b.generation == target_gen) or (not cell_b.children)

        if cell_a_is_terminal and cell_b_is_terminal:
            x = self._cell_to_idx_X[cell_a]
            y = self._cell_to_idx_Y[cell_b]
            found_edges = []

            # Add edge if it's not already in N and violates complementary slackness
            if (x, y) not in self.N_set:
                # CRITICAL FIX: Check if ĉ - β < α' (violation requiring edge)
                if c_hat_val - beta[y] < alpha_prime[x]:
                    found_edges.append((x, y))

            return found_edges

        # === RECURSIVE CASE: Expand to children ===
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
