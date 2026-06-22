import numpy as np

class AuctionLAP:
    def __init__(self, cost_matrix, epsilon=None):
        self.C = np.array(cost_matrix, dtype=float)
        self.N = self.C.shape[0]
        
        # Initialization of state variables
        self.beta = np.zeros(self.N)        
        self.x_to_y = np.full(self.N, -1)   # S (X -> Y)
        self.y_to_x = np.full(self.N, -1)   # S (Y -> X)
        
        # epsilon < Delta c / |X|
        if epsilon is None:
            unique_costs = np.unique(self.C)
            if len(unique_costs) > 1:
                delta_c = np.min(np.diff(unique_costs))
                self.epsilon = delta_c / (self.N + 1.0)
            else:
                self.epsilon = 1e-4
        else:
            self.epsilon = epsilon

    def solve(self):
        iterations = 0
        while np.any(self.x_to_y == -1):
            bids = self._bidding_phase()
            self._assignment_phase(bids)
            iterations += 1
        
        total_cost = sum(self.C[x, self.x_to_y[x]] for x in range(self.N))
        return self.x_to_y, total_cost, iterations

    def _bidding_phase(self):
        bids = {y: [] for y in range(self.N)}
        unassigned_x = np.where(self.x_to_y == -1)[0]
        
        for x in unassigned_x:
            # c(x,y) - beta(y)
            eff_costs = self.C[x, :] - self.beta
            
            idx = np.argpartition(eff_costs, 1)
            y_star = idx[0]
            y_second = idx[1]
            
            if eff_costs[y_second] < eff_costs[y_star]:
                y_star, y_second = y_second, y_star
                
            # Eq 7
            alpha_prime = eff_costs[y_second]
            
            # Eq 8
            bid_value = self.C[x, y_star] - alpha_prime - self.epsilon
            bids[y_star].append((x, bid_value))
            
        return bids

    def _assignment_phase(self, bids):
        for y, y_bids in bids.items():
            if not y_bids:
                continue
                
            best_x, min_bid = min(y_bids, key=lambda item: item[1])
            
            # Eq 9
            self.beta[y] = min_bid
            
            # Update assignment S
            old_x = self.y_to_x[y]
            if old_x != -1:
                # Remove from the assignment S any pair (x,y) if one exists
                self.x_to_y[old_x] = -1
                
            # Add to S the new pair (x*, y)
            self.x_to_y[best_x] = y
            self.y_to_x[y] = best_x
