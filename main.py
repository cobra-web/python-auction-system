import numpy as np
import time
from collections import defaultdict

# Standard Dense Auction Algorithm
def standard_auction_dense(X, Y, epsilon=0.01):
    """
    Standard Auction Algorithm operating on a dense cost matrix.
    Matches the baseline un-accelerated approach.
    """
    N = len(X)
    cost_matrix = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    
    S = {} 
    unassigned_x = set(range(N))
    beta = np.zeros(N)
    
    while unassigned_x:
        bids = defaultdict(list)
        
        # Bidding Phase
        for x in unassigned_x:
            eff_costs = cost_matrix[x, :] - beta
            
            partitioned_idx = np.argpartition(eff_costs, 1)
            best_y = partitioned_idx[0]
            second_best_y = partitioned_idx[1]
            
            best_eff_cost = eff_costs[best_y]
            alpha_prime = eff_costs[second_best_y]
            
            bid_value = cost_matrix[x, best_y] - alpha_prime - epsilon
            bids[best_y].append((x, bid_value))
            
        # Assignment Phase
        for y, P_y in bids.items():
            if P_y:
                winning_x, lowest_bid = min(P_y, key=lambda item: item[1])
                beta[y] = lowest_bid
                
                if y in S:
                    unassigned_x.add(S[y])
                    
                S[y] = winning_x
                unassigned_x.remove(winning_x)
                
    return S, cost_matrix

# Hierarchical Hybrid Auction Algorithm

class HierarchicalHybridAuction:
    def __init__(self, X, Y, scales, epsilon=0.01):
        self.X = X
        self.Y = Y
        self.N = len(X)
        self.scales = scales
        self.epsilon = epsilon
        
        self.beta = np.zeros(self.N)
        self.alpha_prime = np.zeros(self.N)
        self.S = {} 
        self.unassigned_x = set(range(self.N))
        
        self.cost_dense = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
        
        self.N_hat = defaultdict(list)
        for i in range(self.N):
            nearest = np.argsort(self.cost_dense[i])[:5]
            self.N_hat[i] = list(nearest)
            
        self.hierarchies_X = self._build_hierarchies(X, scales)
        self.hierarchies_Y = self._build_hierarchies(Y, scales)

    def _build_hierarchies(self, points, scales):
        hierarchies = []
        for scale in scales:
            coords = np.floor(points * scale).astype(int)
            coords = np.clip(coords, 0, scale - 1)
            labels = coords[:, 0] * scale + coords[:, 1]
            
            cell_to_points = defaultdict(list)
            for pt_idx, cell_id in enumerate(labels):
                cell_to_points[cell_id].append(pt_idx)
            hierarchies.append(cell_to_points)
        return hierarchies

    def run(self):
        iteration = 0
        while self.unassigned_x:
            iteration += 1
            bids = defaultdict(list)
            x_to_rebid = list(self.unassigned_x)
            
            # Bidding Phase
            for x in x_to_rebid:
                neighbors = self.N_hat[x]
                eff_costs = [(self.cost_dense[x, y] - self.beta[y], y) for y in neighbors]
                eff_costs.sort(key=lambda item: item[0])
                
                best_eff_cost, y_star = eff_costs[0]
                
                if len(eff_costs) > 1:
                    self.alpha_prime[x] = eff_costs[1][0]
                else:
                    self.alpha_prime[x] = best_eff_cost + 1.0 
                    
                bid_value = self.cost_dense[x, y_star] - self.alpha_prime[x] - self.epsilon
                bids[y_star].append((x, bid_value))
                
            # Assignment Phase
            for y, P_y in bids.items():
                if P_y:
                    winning_x, lowest_bid = min(P_y, key=lambda item: item[1])
                    self.beta[y] = lowest_bid
                    
                    if y in self.S:
                        self.unassigned_x.add(self.S[y])
                        
                    self.S[y] = winning_x
                    if winning_x in self.unassigned_x:
                        self.unassigned_x.remove(winning_x)

            # Consistency Check Phase
            if iteration % 5 == 0 or not self.unassigned_x:
                new_links_found = self._hierarchical_consistency_check()
                if new_links_found:
                    for x in new_links_found:
                        self.unassigned_x.add(x)
                        keys_to_remove = [y_key for y_key, x_val in self.S.items() if x_val == x]
                        for k in keys_to_remove:
                            del self.S[k]

        return self.S, self.cost_dense

    def _hierarchical_consistency_check(self):
        new_active_x = set()
        cells_X = self.hierarchies_X[0]
        cells_Y = self.hierarchies_Y[0]
        
        for a_id, pts_a in cells_X.items():
            for b_id, pts_b in cells_Y.items():
                self._check_cell_pair(a_id, b_id, pts_a, pts_b, 0, new_active_x)
        return new_active_x

    def _check_cell_pair(self, a_id, b_id, pts_a, pts_b, scale_idx, new_active_x):
        alpha_prime_hat = np.max(self.alpha_prime[pts_a])
        beta_hat = np.max(self.beta[pts_b])
        c_hat = np.min(self.cost_dense[np.ix_(pts_a, pts_b)])
        
        if c_hat - beta_hat >= alpha_prime_hat:
            return 

        if scale_idx < len(self.scales) - 1:
            next_scale_idx = scale_idx + 1
            cells_X_next = self.hierarchies_X[next_scale_idx]
            cells_Y_next = self.hierarchies_Y[next_scale_idx]
            
            children_a = self._group_by_children(pts_a, cells_X_next)
            children_b = self._group_by_children(pts_b, cells_Y_next)
            
            for child_a_id, child_pts_a in children_a.items():
                for child_b_id, child_pts_b in children_b.items():
                    self._check_cell_pair(child_a_id, child_b_id, child_pts_a, child_pts_b, next_scale_idx, new_active_x)
        else:
            for x in pts_a:
                for y in pts_b:
                    if y not in self.N_hat[x]:
                        if self.cost_dense[x, y] - self.beta[y] < self.alpha_prime[x]:
                            self.N_hat[x].append(y)
                            new_active_x.add(x)

    def _group_by_children(self, parent_points, next_level_cells):
        children = defaultdict(list)
        for pt in parent_points:
            for cell_id, cell_pts in next_level_cells.items():
                if pt in cell_pts:
                    children[cell_id].append(pt)
                    break
        return children

# Benchmark Runner
def calculate_total_cost(assignment, cost_matrix):
    return sum(cost_matrix[x, y] for y, x in assignment.items())

def run_benchmark():
    problem_sizes = [250, 500, 1000]
    np.random.seed(42)
    
    print(f"{'N':<6} | {'Dense Time (s)':<15} | {'Hybrid Time (s)':<15} | {'Dense Cost':<12} | {'Hybrid Cost':<12} | {'Active Pairs':<15}")
    print("-" * 85)
    
    for N in problem_sizes:
        X = np.random.rand(N, 2)
        Y = np.random.rand(N, 2)
        
        #Run Standard Dense
        t0 = time.time()
        assignment_dense, cost_matrix = standard_auction_dense(X, Y, epsilon=0.005)
        time_dense = time.time() - t0
        cost_dense = calculate_total_cost(assignment_dense, cost_matrix)
        
        #Run Hierarchical Hybrid
        scales = [2, 4, 8, 16]
        auction_hybrid = HierarchicalHybridAuction(X, Y, scales=scales, epsilon=0.005)
        
        t1 = time.time()
        assignment_hybrid, _ = auction_hybrid.run()
        time_hybrid = time.time() - t1
        cost_hybrid = calculate_total_cost(assignment_hybrid, cost_matrix)
        
        #Calculates active pairs used by hybrid
        active_pairs = sum(len(neighbors) for neighbors in auction_hybrid.N_hat.values())
        max_pairs = N * N
        sparsity_str = f"{active_pairs}/{max_pairs}"
        
        print(f"{N:<6} | {time_dense:<15.3f} | {time_hybrid:<15.3f} | {cost_dense:<12.5f} | {cost_hybrid:<12.5f} | {sparsity_str:<15}")

if __name__ == "__main__":
    run_benchmark()







