import numpy as np

TOL = 1e-7

class AuctionOT:

    def __init__(self, cost_matrix, mu_X, mu_Y, epsilon=None,
                 allowed_edges=None, initial_beta=None, normalize=True):
        self.C_raw = np.array(cost_matrix, dtype=float)

        if normalize:
            m = np.max(np.abs(self.C_raw))
            self.max_c = m if m > 0 else 1.0
            self.C = self.C_raw / self.max_c
        else:
            self.max_c = 1.0
            self.C = self.C_raw

        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)
        self.N_X, self.N_Y = self.C.shape

        mass_X = float(self.mu_X.sum())
        mass_Y = float(self.mu_Y.sum())
        if abs(mass_X - mass_Y) > 1e-6 * max(1.0, mass_X):
            raise ValueError("Unbalanced OT problem. Sum of marginals must match.")

        self.epsilon = float(epsilon) if epsilon is not None else 1e-3
        
        # --- State Matrices ---
        self.mu = np.zeros((self.N_X, self.N_Y), dtype=float)
        
        # O(1) Incremental mass trackers (Eliminates np.sum in inner loops)
        self.assigned_Y = np.zeros(self.N_Y, dtype=float)
        self.unassigned_X = np.copy(self.mu_X)

        if initial_beta is not None:
            self.beta_diamond = np.array(initial_beta, dtype=float)
        else:
            self.beta_diamond = np.zeros(self.N_Y, dtype=float)
        
        self.beta_tilde = np.zeros((self.N_X, self.N_Y), dtype=float)

        if allowed_edges is not None:
            nbrs = [[] for _ in range(self.N_X)]
            for edge in allowed_edges:
                x, y = int(edge[0]), int(edge[1])
                nbrs[x].append(y)
            self.neighbors = [np.array(sorted(set(v)), dtype=int) for v in nbrs]
        else:
            self.neighbors = None

    def get_effective_beta(self):
        eff_beta = np.copy(self.beta_diamond)
        for y in range(self.N_Y):
            active_x = np.where(self.mu[:, y] > TOL)[0]
            if len(active_x) > 0:
                eff_beta[y] = max(eff_beta[y], np.max(self.beta_tilde[active_x, y]))
        return eff_beta

    def _admissible(self, x):
        if self.neighbors is None:
            return np.arange(self.N_Y)
        return self.neighbors[x]

    def _sorted_slots(self, x):
        """Pure vectorized candidate generation. No Python list appends."""
        ys = self._admissible(x)
        if ys.size == 0:
            return None

        # 1. Evaluate Free Capacity
        free_Y = self.mu_Y[ys] - self.assigned_Y[ys]
        free_mask = free_Y > TOL
        
        valid_ys_free = ys[free_mask]
        slacks_free = self.C[x, valid_ys_free] - self.beta_diamond[valid_ys_free]
        caps_free = free_Y[free_mask]
        xp_free = np.full(len(valid_ys_free), -1, dtype=int)

        # 2. Evaluate Occupied Capacity
        mu_sub = self.mu[:, ys]
        xp_indices, y_idx_local = np.nonzero(mu_sub > TOL)
        
        # Fast filter to exclude x displacing itself
        other_mask = xp_indices != x
        xp_indices = xp_indices[other_mask]
        y_idx_local = y_idx_local[other_mask]

        actual_ys = ys[y_idx_local]
        slacks_occ = self.C[x, actual_ys] - self.beta_tilde[xp_indices, actual_ys]
        caps_occ = mu_sub[xp_indices, y_idx_local]

        # Combine all candidates dynamically
        slacks = np.concatenate((slacks_free, slacks_occ))
        if slacks.size == 0:
            return None

        caps = np.concatenate((caps_free, caps_occ))
        ys_arr = np.concatenate((valid_ys_free, actual_ys))
        xp_arr = np.concatenate((xp_free, xp_indices))
        
        # Sort ascending by slack
        order = np.argsort(slacks, kind="stable")
        
        return ys_arr[order], slacks[order], caps[order], xp_arr[order]

    @staticmethod
    def _marginal_alpha_prime(slacks, caps, demand):
        cum = 0.0
        n = len(caps)
        for i in range(n):
            cum += caps[i]
            if cum >= demand - TOL:
                if cum > demand + TOL:
                    return slacks[i], i
                if i + 1 < n:
                    return slacks[i + 1], i
                return slacks[i], i
        return slacks[-1], n - 1

    def _place(self, x, y, owner_xp, amount, new_beta_tilde):
        """Moves mass and updates tracking arrays in O(1) time."""
        if amount <= TOL:
            return 0.0
            
        if owner_xp == -1:
            # Consuming free capacity
            free = self.mu_Y[y] - self.assigned_Y[y]
            take = min(amount, free)
            if take <= TOL: 
                return 0.0
            
            self.mu[x, y] += take
            self.assigned_Y[y] += take
            self.unassigned_X[x] -= take
            self.beta_tilde[x, y] = new_beta_tilde
            return take
        else:
            # Displacing mass owned by owner_xp
            avail = self.mu[owner_xp, y]
            take = min(amount, avail)
            if take <= TOL: 
                return 0.0
            
            self.mu[owner_xp, y] -= take
            self.mu[x, y] += take
            self.unassigned_X[owner_xp] += take
            self.unassigned_X[x] -= take
            self.beta_tilde[x, y] = new_beta_tilde
            return take

    def solve(self):
        max_iterations = 2_000_000
        iterations = 0
        eps = self.epsilon

        while iterations < max_iterations:
            # Snap unassigned mass dynamically rather than computing np.sum(mu)
            self.unassigned_X = np.where(self.unassigned_X < TOL, 0.0, self.unassigned_X)
            total_unassigned = float(np.sum(self.unassigned_X))

            if total_unassigned <= TOL:
                break

            moved_this_sweep = 0.0

            for x in range(self.N_X):
                r = self.unassigned_X[x]
                if r <= TOL:
                    continue

                slots = self._sorted_slots(x)
                if slots is None:
                    continue
                ys, slacks, caps, xp_arr = slots

                alpha_prime, m_idx = self._marginal_alpha_prime(slacks, caps, r)

                remaining = r
                for i in range(m_idx + 1):
                    if remaining <= TOL:
                        break
                    y = ys[i]
                    owner_xp = xp_arr[i]
                    
                    new_beta_tilde = self.C[x, y] - alpha_prime - eps
                    
                    placed = self._place(x, y, owner_xp, remaining, new_beta_tilde)
                    if placed > 0:
                        remaining -= placed
                        moved_this_sweep += placed

            iterations += 1
            if iterations >= max_iterations:
                break

            if moved_this_sweep <= TOL:
                print(f"[AuctionOT] Deadlock at iter {iterations}; unassigned = {total_unassigned:.3e}.")
                break

        cost = float(np.sum(self.mu * self.C))
        return self.mu, cost, iterations
