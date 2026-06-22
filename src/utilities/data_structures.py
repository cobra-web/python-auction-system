class SparseNeighborhood:
    """
    Manages the sparse subset of allowed assignment pairs \hat{N}.
    Essential for the fast iterations and O(1) lookups required by the 
    sparse/dense hybrid auction algorithm.
    """
    def __init__(self, n_X, n_Y, initial_edges=None):
        self.n_X = n_X
        self.n_Y = n_Y
        
        # Adjacency list: mapping x to a set of allowed y's for fast iteration
        # Using a set prevents duplicate edges and allows O(1) membership checks
        self.adj_X = {x: set() for x in range(n_X)}
        self.edge_count = 0
        
        if initial_edges is not None:
            self.add_edges(initial_edges)

    def add_edge(self, x, y):
        """Adds a single (x, y) edge to the active neighborhood."""
        if y not in self.adj_X[x]:
            self.adj_X[x].add(y)
            self.edge_count += 1

    def add_edges(self, edges):
        """Adds an iterable of (x, y) tuples to the active neighborhood."""
        for x, y in edges:
            self.add_edge(x, y)

    def get_allowed_y(self, x):
        """Returns the set of allowed y targets for a given x."""
        return self.adj_X[x]

    def has_edge(self, x, y):
        """O(1) check if a specific (x,y) assignment is currently allowed."""
        return y in self.adj_X[x]

    def get_all_edges(self):
        """Reconstructs the list of all active edges (useful for the Consistency Checker)."""
        edges = []
        for x, y_set in self.adj_X.items():
            for y in y_set:
                edges.append((x, y))
        return edges

    def sparsity_ratio(self):
        """
        Returns the fraction of the dense matrix that is currently active.
        Useful for logging to ensure the heuristic is actually keeping the problem sparse.
        """
        max_edges = self.n_X * self.n_Y
        if max_edges == 0:
            return 0.0
        return self.edge_count / max_edges

    def __repr__(self):
        return f"<SparseNeighborhood: {self.edge_count} active edges ({self.sparsity_ratio():.2%} dense)>"
