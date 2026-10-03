import numpy as np


class ConsistencyChecker:
    """Hierarchical consistency check (Schmitzer-Schnoerr 2013, Sec. 4).

    Given the dual solution (alpha', beta) of the sparse problem on the active cells at
    target_depth, find all pairs (a, b) outside the neighbourhood N for which

        cost(a, b) - beta(b) < alpha'(a)

    i.e. pairs that would violate eps-CS in the full problem. Interior node pairs are pruned with
    the cell lower bound c_hat(A, B) = min squared distance between the bounding boxes and the
    maximal duals of the subtrees:

        c_hat(A, B) - beta_hat(B) >= alpha'_hat(A)   =>   no violating pair below (A, B).

    This is sound because c_hat(A, B) <= cost(a, b) for every pair of descendants (the centroid or
    point of a cell lies inside its bounding box), beta_hat(B) >= beta(b) and alpha'_hat(A) >= alpha'(a).

    The decision at a terminal pair uses cost_fn, the cost the solver really used on this level
    (centroid cost, or the point cost on the final level), NOT the bounding box bound.
    """

    def __init__(self, tree_X, tree_Y, initial_sparse_N=None, max_c=1.0):
        self.tree_X = tree_X
        self.tree_Y = tree_Y
        self.max_c = float(max_c)
        self.N_set = set(initial_sparse_N) if initial_sparse_N is not None else set()
        self.c_hat_cache = {}

    def _compute_extensions(self, alpha_prime, beta, target_depth):
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

        def _propagate_up(node, table):
            if node.id in table:
                return table[node.id]
            if not node.children:
                return -float('inf')
            val = max(_propagate_up(child, table) for child in node.children)
            table[node.id] = val
            return val

        _propagate_up(self.tree_X.cells[0], self.alpha_prime_hat)
        _propagate_up(self.tree_Y.cells[0], self.beta_hat)

        self._cell_to_idx_X = cell_to_idx_X
        self._cell_to_idx_Y = cell_to_idx_Y

    @staticmethod
    def _bbox_min_sq_dist(bbox_a, bbox_b):
        min_a, max_a = bbox_a
        min_b, max_b = bbox_b
        dist = 0.0
        for i in range(len(min_a)):
            d = max(0.0, min_a[i] - max_b[i], min_b[i] - max_a[i])
            dist += d * d
        return dist

    def _c_hat(self, cell_a, cell_b):
        key = (cell_a.id, cell_b.id)
        val = self.c_hat_cache.get(key)
        if val is None:
            val = self._bbox_min_sq_dist(cell_a.bbox, cell_b.bbox) / self.max_c
            self.c_hat_cache[key] = val
        return val

    def run_consistency_check(self, alpha_prime, beta, target_depth, cost_fn=None):
        """Returns the list of violating pairs and adds them to N_set.

        cost_fn(x, y): normalised cost used by the solver on this level for the active cells with
        indices x, y. If None, the (weaker) bounding box bound is used, which only adds more edges.
        """
        self._compute_extensions(alpha_prime, beta, target_depth)
        self._cost_fn = cost_fn
        new_edges = self._check_recursive(self.tree_X.cells[0], self.tree_Y.cells[0],
                                          alpha_prime, beta, target_depth)
        for x, y in new_edges:
            self.N_set.add((x, y))
        return new_edges

    def _check_recursive(self, cell_a, cell_b, alpha_prime, beta, target_depth):
        c_hat_val = self._c_hat(cell_a, cell_b)
        if c_hat_val - self.beta_hat[cell_b.id] >= self.alpha_prime_hat[cell_a.id]:
            return []

        a_terminal = (cell_a.depth == target_depth) or (not cell_a.children)
        b_terminal = (cell_b.depth == target_depth) or (not cell_b.children)

        if a_terminal and b_terminal:
            x = self._cell_to_idx_X[cell_a]
            y = self._cell_to_idx_Y[cell_b]
            if (x, y) in self.N_set:
                return []
            c = self._cost_fn(x, y) if self._cost_fn is not None else c_hat_val
            if c - beta[y] < alpha_prime[x]:
                return [(x, y)]
            return []

        found = []
        a_children = [cell_a] if a_terminal else cell_a.children
        b_children = [cell_b] if b_terminal else cell_b.children
        for ca in a_children:
            for cb in b_children:
                found.extend(self._check_recursive(ca, cb, alpha_prime, beta, target_depth))
        return found
