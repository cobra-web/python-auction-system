# --- Fixed Original Coordinate Reconstruction ---
        reconstructed_mu = np.zeros_like(self.C)
        leaf_cells_X = self.tree_X.generations[0]
        leaf_cells_Y = self.tree_Y.generations[0]

        for i, cell_a in enumerate(leaf_cells_X):
            for j, cell_b in enumerate(leaf_cells_Y):
                coarse_mass = current_mu[i, j]
                if coarse_mass <= 0:
                    continue

                idx_a = cell_a.point_indices
                idx_b = cell_b.point_indices

                # Sub-allocate the cell-level optimized mass into its constituent raw points
                # For basic evaluation, evenly split or greedily fill to keep integrity
                for original_x in idx_a:
                    for original_y in idx_b:
                        # Distribute mass safely without dropping single elements
                        reconstructed_mu[original_x, original_y] = coarse_mass // (len(idx_a) * len(idx_b))
                
                # Cleanup rounding remainders onto the first index pair so no mass vanishes
                remainder = coarse_mass - np.sum(reconstructed_mu[np.ix_(idx_a, idx_b)])
                if remainder > 0:
                    reconstructed_mu[idx_a[0], idx_b[0]] += remainder

        return reconstructed_mu
