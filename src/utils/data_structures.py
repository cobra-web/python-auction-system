import numpy as np

class SparseNeighborhood:
    def __init__(self, n_X, n_Y, initial_edges=None):
        self.n_X = n_X
        self.n_Y = n_Y
        
        self.adj_X = [np.array([], dtype=np.int32) for _ in range(n_X)]
        self.edge_count = 0
        
        if initial_edges is not None:
            self.add_edges(initial_edges)

    def add_edge(self, x, y):
        if y not in self.adj_X[x]:
            self.adj_X[x] = np.append(self.adj_X[x], y)
            self.edge_count += 1

    def add_edges(self, edges):
        edges = np.asarray(edges, dtype=np.int32)
        if len(edges) == 0:
            return

        unique_x = np.unique(edges[:, 0])
        for x in unique_x:
            new_y = edges[edges[:, 0] == x, 1]
            
            combined = np.concatenate((self.adj_X[x], new_y))
            unique_combined = np.unique(combined)
            
            self.edge_count += (len(unique_combined) - len(self.adj_X[x]))
            self.adj_X[x] = unique_combined

    def get_allowed_y(self, x):
        return self.adj_X[x]

    def has_edge(self, x, y):
        return y in self.adj_X[x]

    def get_all_edges(self):
        if self.edge_count == 0:
            return np.empty((0, 2), dtype=np.int32)
            
        x_indices = np.repeat(np.arange(self.n_X), [len(y_arr) for y_arr in self.adj_X])
        y_indices = np.concatenate(self.adj_X)
        
        return np.column_stack((x_indices, y_indices))

    def sparsity_ratio(self):
        max_edges = self.n_X * self.n_Y
        if max_edges == 0:
            return 0.0
        return self.edge_count / max_edges

    def __repr__(self):
        return f"<SparseNeighborhood: {self.edge_count} active edges ({self.sparsity_ratio():.2%} dense)>"
