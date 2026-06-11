import numpy as np

def standard_auction(c_matrix, epsilon):
    num_x, num_y = c_matrix.shape
    if num_x != num_y:
        raise ValueError("Cost matrix must be square for standard assignment.")

    X = list(range(num_x))
    Y = list(range(num_y))

    S = {}              # Assignment {x: y}
    S_inv = {}          # Reverse lookup {y: x}
    beta = np.zeros(num_y) 

    while len(S) < num_x:
        bids = {y: [] for y in Y}
        
        # ----------------------------------------------------
        # BIDDING PHASE
        # ----------------------------------------------------
        unassigned_x = [x for x in X if x not in S]
        
        for x in unassigned_x:
            # Vectorized calculation: val = c(x, y) - beta[y] for all y
            vals = c_matrix[x, :] - beta
            
            # Efficiently find the best and second-best y
            # argpartition is faster than a full sort for finding top 2 elements
            idx = np.argpartition(vals, 1)
            y_star, y_second = idx[0], idx[1]
            
            # Ensure y_star is strictly the minimum
            if vals[y_star] > vals[y_second]:
                y_star, y_second = y_second, y_star
                
            best_val = vals[y_star]
            second_best_val = vals[y_second]
            
            # Compute the bid value
            b_xy_star = c_matrix[x, y_star] - second_best_val - epsilon
            bids[y_star].append((x, b_xy_star))
            
        for y in Y:
            if bids[y]:
                # Find the lowest bid submitted to y
                x_star, min_bid_val = min(bids[y], key=lambda item: item[1])
                
                # Decrease beta(y) to the lowest bid received
                beta[y] = min_bid_val
                
                if y in S_inv:
                    del S[S_inv[y]]
                
                S[x_star] = y
                S_inv[y] = x_star

    return S, beta


class HybridAuction:
    def __init__(self, X, Y, cost_func, epsilon):
        self.X = X
        self.Y = Y
        self.c = cost_func  # Function c(x, y) returning cost
        self.epsilon = epsilon
        
        self.S = {}
        self.S_inv = {}
        self.beta = {y: 0.0 for y in Y}
        self.alpha_prime = {x: float('inf') for x in X}
        
        self.N_hat = self.initialize_with_sparse_heuristic()
        
    def initialize_with_sparse_heuristic(self):
        """Initialize the non-maximal sparse subset. Returns a set of (x, y) tuples."""
        # e.g., K-Nearest Neighbors initialization
        return set() 

    # ----------------------------------------------------
    # HIERARCHICAL INTERFACES (To be implemented by user)
    # ----------------------------------------------------
    def hat_c(self, a, b): pass
    def hat_beta(self, b): pass
    def hat_alpha_prime(self, a): pass
    
    def ch(self, node): 
        """Return children of a hierarchical node."""
        return node.children
        
    def is_singleton(self, node):
        return len(self.ch(node)) == 0
        
    def element(self, node):
        """Extract the raw element (x or y) if the node is a singleton."""
        return node.value 
        
    def get_highest_generation_nodes(self):
        """Return (A_n, B_n) - the coarsest partitions of X and Y."""
        pass

    # ----------------------------------------------------
    # CORE ALGORITHM
    # ----------------------------------------------------
    def consistency_check(self, a, b, rebidding_list):
        """Helper function for the hierarchical consistency check."""
        
        # Check the inequality at generation n
        if self.hat_c(a, b) - self.hat_beta(b) >= self.hat_alpha_prime(a):
            return  # Condition holds
            
        # If the constraint is violated, drill down to a finer level
        if not self.is_singleton(a) and not self.is_singleton(b):
            for a_child in self.ch(a):
                for b_child in self.ch(b):
                    self.consistency_check(a_child, b_child, rebidding_list)
        else:
            # Reached generation 0 (singletons)
            x = self.element(a)
            y = self.element(b)
            
            # Candidate condition check at the finest resolution
            if self.c(x, y) - self.beta[y] < self.alpha_prime[x]:
                if (x, y) not in self.N_hat:
                    # Dynamically expand the sparse neighborhood
                    self.N_hat.add((x, y))
                    rebidding_list.add(x)

    def run_bidding_phase(self, active_X):
        """Standard bidding restricted strictly to N_hat."""
        bids = {}
        for x in active_X:
            if x in self.S:
                continue # Only unassigned elements bid
                
            best_val = float('inf')
            second_best_val = float('inf')
            y_star = None
            
            # Restrict search to sparse neighborhood N_hat
            N_x = [y for (x_val, y) in self.N_hat if x_val == x]
            
            for y in N_x:
                val = self.c(x, y) - self.beta[y]
                if val < best_val:
                    second_best_val = best_val
                    best_val = val
                    y_star = y
                elif val < second_best_val:
                    second_best_val = val
                    
            if y_star is not None:
                self.alpha_prime[x] = second_best_val
                b_xy_star = self.c(x, y_star) - second_best_val - self.epsilon
                
                if y_star not in bids:
                    bids[y_star] = []
                bids[y_star].append((x, b_xy_star))
                
        return bids

    def run_assignment_phase(self, bids):
        """Standard assignment resolving bids."""
        for y, y_bids in bids.items():
            if len(y_bids) > 0:
                x_star, min_bid_val = min(y_bids, key=lambda item: item[1])
                self.beta[y] = min_bid_val
                
                if y in self.S_inv:
                    assigned_x = self.S_inv[y]
                    del self.S[assigned_x]
                    
                self.S[x_star] = y
                self.S_inv[y] = x_star

    def solve(self):
        """Main Hybrid Loop."""
        while len(self.S) < len(self.X):
            
            # 1 & 2. Run Standard Bidding and Assignment (Constrained to N_hat)
            bids = self.run_bidding_phase(self.X)
            self.run_assignment_phase(bids)
            
            # 3. Consistency Check Phase
            # Note: Hierarchical extensions (hat_alpha_prime, hat_beta, hat_c) 
            # should be recomputed or updated here based on current state.
            
            rebidding_list = set()
            A_n, B_n = self.get_highest_generation_nodes()
            
            for a in A_n:
                for b in B_n:
                    self.consistency_check(a, b, rebidding_list)
                    
            # 4. Re-evaluation Phase
            if len(rebidding_list) > 0:
                # Elements with new sparse neighbors re-bid
                # If an element is re-bidding, we temporarily remove it from S to force a fresh bid
                for x in rebidding_list:
                    if x in self.S:
                        y_assigned = self.S[x]
                        del self.S[x]
                        del self.S_inv[y_assigned]
                        
                new_bids = self.run_bidding_phase(rebidding_list)
                self.run_assignment_phase(new_bids)

        return self.S, self.beta
