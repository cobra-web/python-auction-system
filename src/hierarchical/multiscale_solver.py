def _build_coarsened_problem(self, depth):
        cells_X = self.tree_X.get_active_cells_at_depth(depth)
        cells_Y = self.tree_Y.get_active_cells_at_depth(depth)

        num_a = len(cells_X)
        num_b = len(cells_Y)

        mu_X_hat = np.zeros(num_a, dtype=float)
        mu_Y_hat = np.zeros(num_b, dtype=float)

        X_pts_hat = np.zeros((num_a, self.tree_X.dimensions))
        Y_pts_hat = np.zeros((num_b, self.tree_Y.dimensions))

        for i, cell_a in enumerate(cells_X):
            mu_X_hat[i] = np.sum(self.mu_X[cell_a.point_indices])
            # Use the true mathematical center of the data points
            X_pts_hat[i] = np.mean(self.X_pts[cell_a.point_indices], axis=0)

        for j, cell_b in enumerate(cells_Y):
            mu_Y_hat[j] = np.sum(self.mu_Y[cell_b.point_indices])
            # Use the true mathematical center of the data points
            Y_pts_hat[j] = np.mean(self.Y_pts[cell_b.point_indices], axis=0)

        return X_pts_hat, Y_pts_hat, mu_X_hat, mu_Y_hat, cells_X, cells_Y
