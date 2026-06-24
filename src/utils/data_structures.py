class SparseNeighborhood:
    def __init__(self, n_X, n_Y, initial_edges=None):
        self.n_X = n_X
        self.n_Y = n_Y
        
        self.adj_X = {x: set() for x in range(n_X)}
        self.edge_count = 0
        
        if initial_edges is not None:
            self.add_edges(initial_edges)

    def add_edge(self, x, y):
        if y not in self.adj_X[x]:
            self.adj_X[x].add(y)
            self.edge_count += 1

    def add_edges(self, edges):
        for x, y in edges:
            self.add_edge(x, y)

    def get_allowed_y(self, x):
        return self.adj_X[x]

    def has_edge(self, x, y):
        return y in self.adj_X[x]

    def get_all_edges(self):
        edges = []
        for x, y_set in self.adj_X.items():
            for y in y_set:
                edges.append((x, y))
        return edges

    def sparsity_ratio(self):
        max_edges = self.n_X * self.n_Y
        if max_edges == 0:
            return 0.0
        return self.edge_count / max_edges

    def __repr__(self):
        return f"<SparseNeighborhood: {self.edge_count} active edges ({self.sparsity_ratio():.2%} dense)>"
