import numpy as np

class HierarchicalOTAuction:
    def __init__(self, mu_X, mu_Y, cost_matrix, eps_start=1.0, eps_target=1e-4, eps_factor=0.2):
        self.mu_X = mu_X
        self.mu_Y = mu_Y
        self.c = cost_matrix
        self.X_len, self.Y_len = cost_matrix.shape
        
        #e-scaling parameters
        self.eps = eps_start
        self.eps_target = eps_target
        self.eps_factor = eps_factor
        
        # OT Variables
        self.mu = np.zeros((self.X_len, self.Y_len)) # coupling matrix mu(x,y)
        self.beta_empty = np.zeros(self.Y_len)       # beta(empty, y) for unassigned mass
        self.beta_xy = np.zeros((self.X_len, self.Y_len)) # beta(x', y) for assigned mass
        
        self.alpha_prime = np.full(self.X_len, np.inf)
        
        # Sparse neighborhood initialization
        self.N_hat = set()
        
    def get_beta(self, y):
        assigned_mass = np.sum(self.mu[:, y])
        if np.isclose(assigned_mass, self.mu_Y[y]):
            # Sink is full, return the max beta of elements currently assigned to it
            assigned_x_indices = np.where(self.mu[:, y] > 0)[0]
            if len(assigned_x_indices) > 0:
                return np.max(self.beta_xy[assigned_x_indices, y])
        
        return self.beta_empty[y]

    def hat_c(self, a, b): pass
    def hat_beta(self, b): pass
    def hat_alpha_prime(self, a): pass
    def get_highest_generation_nodes(self): return [], []
    def is_singleton(self, node): return True
    def ch(self, node): return []
    def element(self, node): return node.value 


    
    def build_Pi_list(self, x, N_x):
        Pi_x = []
        for y in N_x:
            assigned_x_indices = np.where(self.mu[:, y] > 0)[0]
            for x_prime in assigned_x_indices:
                if x_prime != x:
                    val = self.c[x, y] - self.beta_xy[x_prime, y]
                    Pi_x.append((val, y, x_prime))
            
            assigned_mass = np.sum(self.mu[:, y])
            if assigned_mass < self.mu_Y[y]:
                val = self.c[x, y] - self.beta_empty[y]
                Pi_x.append((val, y, None)) # None represents the 'empty' mass atom
                
        # Sort ascending based on the value c(x,y) - beta(...)
        Pi_x.sort(key=lambda item: item[0])
        return Pi_x

    def run_bidding_phase(self, active_X):
        bids = {y: [] for y in range(self.Y_len)}
        
        for x in active_X:
            unassigned_mass = self.mu_X[x] - np.sum(self.mu[x, :])
            if unassigned_mass <= 0:
                continue
                
            N_x = [y for (x_val, y) in self.N_hat if x_val == x]
            if not N_x:
                continue
                
            # Create the ordered list Pi(x)
            Pi_x = self.build_Pi_list(x, N_x)
            if not Pi_x:
                continue
                
            # Determine integer m based on mass distribution to find alpha_prime
            # For simplicity in this array-based struct, we assume unit mass atoms 
            # and pick the second best distinct target as a proxy for equation (12)
            m = min(1, len(Pi_x) - 1) 
            self.alpha_prime[x] = Pi_x[m][0]
            
            # Calculate bid for the best target
            best_val, y_star, x_prime_target = Pi_x[0]
            b_xy_star = self.c[x, y_star] - self.alpha_prime[x] - self.eps
            
            # Submit bid (includes how much mass x wants to send, though 
            # true auction pushes atoms one by one)
            bids[y_star].append((x, b_xy_star, x_prime_target, unassigned_mass))
            
        return bids

    def run_assignment_phase(self, bids):
        """
        Resolves bids, updates coupling mu, and adjusts dual variables beta.
        """
        for y, y_bids in bids.items():
            if len(y_bids) > 0:
                # Find the lowest bid
                best_bid = min(y_bids, key=lambda item: item[1])
                x_star, min_bid_val, x_prime_target, mass_requested = best_bid
                
                # Update dual variable depending on what was outbid
                if x_prime_target is None:
                    self.beta_empty[y] = min_bid_val
                else:
                    self.beta_xy[x_prime_target, y] = min_bid_val
                    # Reallocate mass from x_prime to x_star
                    transfer_mass = min(self.mu[x_prime_target, y], mass_requested)
                    self.mu[x_prime_target, y] -= transfer_mass
                    self.mu[x_star, y] += transfer_mass

    def consistency_check(self, a, b, rebidding_list):
        # Hierarchical check: hat_c(a,b) - hat_beta(b) >= hat_alpha_prime(a)
        if self.hat_c(a, b) - self.hat_beta(b) >= self.hat_alpha_prime(a):
            return 
            
        if not self.is_singleton(a) or not self.is_singleton(b):
            a_children = self.ch(a) if not self.is_singleton(a) else [a]
            b_children = self.ch(b) if not self.is_singleton(b) else [b]
            for a_child in a_children:
                for b_child in b_children:
                    self.consistency_check(a_child, b_child, rebidding_list)
        else:
            x, y = self.element(a), self.element(b)
            # Compare finest resolution against actual beta(y)
            if self.c[x, y] - self.get_beta(y) < self.alpha_prime[x]:
                if (x, y) not in self.N_hat:
                    self.N_hat.add((x, y))
                    rebidding_list.add(x)

    def solve_fixed_epsilon(self):
        """Runs the hybrid auction loop for a fixed value of epsilon."""
        # Loop until all mass in X is assigned
        while not np.allclose(np.sum(self.mu, axis=1), self.mu_X):
            
            # Find elements with unassigned mass
            unassigned_X = np.where(np.sum(self.mu, axis=1) < self.mu_X)[0]
            bids = self.run_bidding_phase(unassigned_X)
            
            rebidding_list = set()
            A_n, B_n = self.get_highest_generation_nodes()
            
            for a in A_n:
                for b in B_n:
                    self.consistency_check(a, b, rebidding_list)
                    
            if len(rebidding_list) > 0:
                # Remove invalidated bids
                for x in rebidding_list:
                    for y in list(bids.keys()):
                        bids[y] = [bid for bid in bids[y] if bid[0] != x]
                
                new_bids = self.run_bidding_phase(list(rebidding_list))
                for y, y_bids in new_bids.items():
                    bids[y].extend(y_bids)

            self.run_assignment_phase(bids)

    def solve_with_scaling(self):
        """
        The epsilon-scaling wrapper.
        Repeatedly solves the problem for decreasing values of epsilon.
        """
        while self.eps > self.eps_target:
            self.solve_fixed_epsilon()
            
            # Decrease epsilon
            self.eps *= self.eps_factor
            
            # Reset alpha_prime for the new tighter epsilon bounds
            self.alpha_prime = np.full(self.X_len, np.inf)
            
        return self.mu, self.beta_empty, self.beta_xy
