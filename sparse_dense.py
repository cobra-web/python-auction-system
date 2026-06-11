import numpy as np
from collections import defaultdict

class HierarchicalHybridAuction:
    def __init__(self, X, Y, scales, epsilon=0.01):
        """
        X, Y: numpy arrays of shape (N, 2) containing 2D coordinates in [0, 1].
        scales: list of grid resolutions (e.g., [2, 4, 8, 16]) representing hierarchy.
        epsilon: bidding increment.
        """
        self.X = X
        self.Y = Y
        self.N = len(X)
        self.scales = scales
        self.epsilon = epsilon
        
        # Dual variables and assignment state
        self.beta = np.zeros(self.N)
        self.alpha_prime = np.zeros(self.N) # Tracks the second-best cost slack
        self.S = {} # Map Y -> X
        self.unassigned_x = set(range(self.N))
        
        # Precompute dense costs (in a real scenario, this is avoided, 
        # but kept here for exact distance lookups at the leaf level)
        self.cost_dense = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
        
        # Initialize sparse neighborhood N_hat with a basic guess (e.g., 5 nearest neighbors)
        self.N_hat = defaultdict(list)
        for i in range(self.N):
            nearest = np.argsort(self.cost_dense[i])[:5]
            self.N_hat[i] = list(nearest)
            
        # Precompute hierarchical grid cell memberships
        self.hierarchies_X = self._build_hierarchies(X, scales)
        self.hierarchies_Y = self._build_hierarchies(Y, scales)

    def _build_hierarchies(self, points, scales):
        """Assigns each point to a grid cell at multiple scales."""
        hierarchies = []
        for scale in scales:
            coords = np.floor(points * scale).astype(int)
            coords = np.clip(coords, 0, scale - 1)
            labels = coords[:, 0] * scale + coords[:, 1]
            
            # Map cell_id -> list of point indices
            cell_to_points = defaultdict(list)
            for pt_idx, cell_id in enumerate(labels):
                cell_to_points[cell_id].append(pt_idx)
            hierarchies.append(cell_to_points)
        return hierarchies

    def run(self):
        """Main loop of the Sparse/Dense Hybrid Auction Algorithm."""
        iteration = 0
        while self.unassigned_x:
            iteration += 1
            
            # --- 1. BIDDING PHASE ---
            bids = defaultdict(list)
            x_to_rebid = list(self.unassigned_x)
            
            for x in x_to_rebid:
                neighbors = self.N_hat[x]
                
                # Calculate effective costs for current valid neighbors
                eff_costs = [(self.cost_dense[x, y] - self.beta[y], y) for y in neighbors]
                eff_costs.sort(key=lambda item: item[0])
                
                best_eff_cost, y_star = eff_costs[0]
                
                # alpha'(x) is the second best constraint [cite: 109-110]
                if len(eff_costs) > 1:
                    self.alpha_prime[x] = eff_costs[1][0]
                else:
                    self.alpha_prime[x] = best_eff_cost + 1.0 # Arbitrary large margin if only 1 neighbor
                    
                bid_value = self.cost_dense[x, y_star] - self.alpha_prime[x] - self.epsilon
                bids[y_star].append((x, bid_value))
                
            # --- 2. ASSIGNMENT PHASE ---
            for y, P_y in bids.items():
                if P_y:
                    winning_x, lowest_bid = min(P_y, key=lambda item: item[1])
                    self.beta[y] = lowest_bid
                    
                    if y in self.S:
                        old_x = self.S[y]
                        self.unassigned_x.add(old_x)
                        
                    self.S[y] = winning_x
                    if winning_x in self.unassigned_x:
                        self.unassigned_x.remove(winning_x)

            # --- 3. CONSISTENCY CHECK PHASE (Hierarchical) ---
            # Periodically check if our sparse subset N_hat missed global optimum links [cite: 192-194]
            if iteration % 5 == 0 or not self.unassigned_x:
                new_links_found = self._hierarchical_consistency_check()
                
                # If new valid assignments were discovered, the affected 'x' are unassigned 
                # so they can re-evaluate their bids [cite: 198-199].
                if new_links_found:
                    for x in new_links_found:
                        self.unassigned_x.add(x)
                        # Remove x from current assignment to force a rebid
                        keys_to_remove = [y_key for y_key, x_val in self.S.items() if x_val == x]
                        for k in keys_to_remove:
                            del self.S[k]

        return self.S, self.beta

    def _hierarchical_consistency_check(self):
        """
        Recursively checks dual constraints on grid cells.
        Returns a set of 'x' indices that found new valid 'y' neighbors.
        """
        new_active_x = set()
        
        # Start at the coarsest scale (lowest resolution)
        coarsest_scale_idx = 0 
        
        cells_X = self.hierarchies_X[coarsest_scale_idx]
        cells_Y = self.hierarchies_Y[coarsest_scale_idx]
        
        for a_id, pts_a in cells_X.items():
            for b_id, pts_b in cells_Y.items():
                self._check_cell_pair(a_id, b_id, pts_a, pts_b, coarsest_scale_idx, new_active_x)
                
        return new_active_x

    def _check_cell_pair(self, a_id, b_id, pts_a, pts_b, scale_idx, new_active_x):
        # Compute hierarchical extensions for the cells 
        alpha_prime_hat = np.max(self.alpha_prime[pts_a])
        beta_hat = np.max(self.beta[pts_b])
        
        # c_hat is the minimum possible cost between any point in cell A and cell B
        # (In a fully optimized setup, this is a geometric lower bound, not a dense slice)
        c_hat = np.min(self.cost_dense[np.ix_(pts_a, pts_b)])
        
        # The core condition [cite: 194-196]
        if c_hat - beta_hat >= alpha_prime_hat:
            return # Condition holds, prune this entire block of possibilities.

        # If condition is violated and we are not at the finest scale, go deeper
        if scale_idx < len(self.scales) - 1:
            next_scale_idx = scale_idx + 1
            cells_X_next = self.hierarchies_X[next_scale_idx]
            cells_Y_next = self.hierarchies_Y[next_scale_idx]
            
            # Map parent cell points to child cell structures
            children_a = self._group_by_children(pts_a, cells_X_next)
            children_b = self._group_by_children(pts_b, cells_Y_next)
            
            for child_a_id, child_pts_a in children_a.items():
                for child_b_id, child_pts_b in children_b.items():
                    self._check_cell_pair(child_a_id, child_b_id, child_pts_a, child_pts_b, next_scale_idx, new_active_x)
        else:
            # We are at the leaf level (generation 0). Check point-to-point [cite: 197-198].
            for x in pts_a:
                for y in pts_b:
                    if y not in self.N_hat[x]:
                        if self.cost_dense[x, y] - self.beta[y] < self.alpha_prime[x]:
                            # A hidden constraint is violated! Add to active neighborhood.
                            self.N_hat[x].append(y)
                            new_active_x.add(x)

    def _group_by_children(self, parent_points, next_level_cells):
        """Helper to find which child cells the parent points belong to."""
        children = defaultdict(list)
        for pt in parent_points:
            for cell_id, cell_pts in next_level_cells.items():
                if pt in cell_pts:
                    children[cell_id].append(pt)
                    break
        return children

# --- Example Usage ---
if __name__ == "__main__":
    np.random.seed(42)
    N = 500
    
    # Generate random points in [0, 1] 2D space
    X = np.random.rand(N, 2)
    Y = np.random.rand(N, 2)
    
    # Grid sizes: 2x2, 4x4, 8x8 grids
    scales = [2, 4, 8]
    
    auction = HierarchicalHybridAuction(X, Y, scales=scales, epsilon=0.01)
    
    print("Starting Hierarchical Sparse/Dense Auction...")
    assignment, dual_beta = auction.run()
    
    print(f"Auction complete. Total assignments made: {len(assignment)}")
    
    # Calculate sparsity of the final neighborhood graph
    total_active_edges = sum(len(neighbors) for neighbors in auction.N_hat.values())
    print(f"Final active pairs: {total_active_edges} out of {N*N} maximum possible.")