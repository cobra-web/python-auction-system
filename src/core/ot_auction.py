import numpy as np
from collections import defaultdict

TOL = 1e-7

class AuctionOT:
    def __init__(self, X_pts, Y_pts, mu_X, mu_Y, epsilon=None, allowed_edges=None, initial_beta=None, normalize=True, max_c=None):
    self.X_pts = np.array(X_pts, dtype=float)
    self.Y_pts = np.array(Y_pts, dtype=float)
    self.N_X = len(self.X_pts)
    self.N_Y = len(self.Y_pts)

    if max_c is not None:
        self.max_c = float(max_c)
    elif normalize:
        min_X, max_X = np.min(self.X_pts, axis=0), np.max(self.X_pts, axis=0)
        min_Y, max_Y = np.min(self.Y_pts, axis=0), np.max(self.Y_pts, axis=0)
        max_dist_sq = np.sum((np.maximum(max_X, max_Y) - np.minimum(min_X, min_Y))**2)
        self.max_c = max_dist_sq if max_dist_sq > 0 else 1.0
    else:
        self.max_c = 1.0

    # --- HELPER FUNCTIONS FOR HYBRID LOOKUPS ---
    
    def _get_mu(self, x, y):
        if self.is_sparse:
            idx = self.col_index[x].get(y)
            if idx is not None:
                return self.mu_arrs[x][idx]
            return 0.0
        return self.mu_dict[x].get(y, 0.0)

    def _set_mu(self, x, y, val):
        if self.is_sparse:
            idx = self.col_index[x].get(y)
            if idx is not None:
                self.mu_arrs[x][idx] = val
        else:
            self.mu_dict[x][y] = val

    def _get_beta_tilde(self, x, y):
        if self.is_sparse:
            idx = self.col_index[x].get(y)
            if idx is not None:
                return self.beta_tilde_arrs[x][idx]
            return 0.0
        return self.beta_tilde_dict[x].get(y, 0.0)

    def _set_beta_tilde(self, x, y, val):
        if self.is_sparse:
            idx = self.col_index[x].get(y)
            if idx is not None:
                self.beta_tilde_arrs[x][idx] = val
        else:
            self.beta_tilde_dict[x][y] = val

    def _get_active_xs_for_y(self, y):
        """Returns a list of x indices that have mass assigned to y"""
        # Unconditional O(1) set lookup instead of O(N_X) scan
        return [x for x in self.owners[y] if self._get_mu(x, y) > TOL]

    # ---------------------------------------------------------

    def _cost(self, x, ys):
        sq_dist = np.sum((self.X_pts[x] - self.Y_pts[ys])**2, axis=1)
        return sq_dist / self.max_c

    def _cost_raw(self, x, y):
        return np.sum((self.X_pts[x] - self.Y_pts[y])**2)

    def get_effective_beta(self):
        eff_beta = np.copy(self.beta_diamond)
        for y in range(self.N_Y):
            active_x = self._get_active_xs_for_y(y)
            if active_x:
                eff_beta[y] = max(eff_beta[y], max(self._get_beta_tilde(x, y) for x in active_x))
        return eff_beta

    def _admissible(self, x):
        if self.neighbors is None:
            return np.arange(self.N_Y)
        return self.neighbors[x]

    def _sorted_slots(self, x):
        ys = self._admissible(x)
        if ys.size == 0:
            return None

        free_Y = self.mu_Y[ys] - self.assigned_Y[ys]
        free_mask = free_Y > TOL
        
        valid_ys_free = ys[free_mask]
        costs_free = self._cost(x, valid_ys_free)
        slacks_free = costs_free - self.beta_diamond[valid_ys_free]
        caps_free = free_Y[free_mask]
        xp_free = np.full(len(valid_ys_free), -1, dtype=int)

        xp_indices = []
        y_idx_local = []
        mu_vals = []
        beta_tilde_vals = []
        
        for idx, y in enumerate(ys):
            active_xs = self._get_active_xs_for_y(y)
            for xp in active_xs:
                if xp != x:
                    xp_indices.append(xp)
                    y_idx_local.append(idx)
                    mu_vals.append(self._get_mu(xp, y))
                    beta_tilde_vals.append(self._get_beta_tilde(xp, y))

        if xp_indices:
            actual_ys = ys[y_idx_local]
            costs_occ = self._cost(x, actual_ys)
            slacks_occ = costs_occ - np.array(beta_tilde_vals)
            caps_occ = np.array(mu_vals)
            xp_indices = np.array(xp_indices)
        else:
            slacks_occ = np.array([])
            caps_occ = np.array([])
            xp_indices = np.array([], dtype=int)
            actual_ys = np.array([], dtype=int)

        slacks = np.concatenate((slacks_free, slacks_occ))
        if slacks.size == 0:
            return None

        caps = np.concatenate((caps_free, caps_occ))
        ys_arr = np.concatenate((valid_ys_free, actual_ys))
        xp_arr = np.concatenate((xp_free, xp_indices))
        
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
        if amount <= TOL:
            return 0.0
            
        if owner_xp == -1:
            free = self.mu_Y[y] - self.assigned_Y[y]
            take = min(amount, free)
            if take <= TOL: 
                return 0.0
            
            self._set_mu(x, y, self._get_mu(x, y) + take)
            
            # Unconditional ownership update
            self.owners[y].add(x) 

            self.assigned_Y[y] += take
            self.unassigned_X[x] -= take
            self._set_beta_tilde(x, y, new_beta_tilde)
            return take
        else:
            avail = self._get_mu(owner_xp, y)
            take = min(amount, avail)
            if take <= TOL: 
                return 0.0
            
            self._set_mu(owner_xp, y, self._get_mu(owner_xp, y) - take)
            self._set_mu(x, y, self._get_mu(x, y) + take)
            
            # Unconditional ownership transfer
            if self._get_mu(owner_xp, y) <= TOL:
                self.owners[y].discard(owner_xp)
            self.owners[y].add(x)

            self.unassigned_X[owner_xp] += take
            self.unassigned_X[x] -= take
            self._set_beta_tilde(x, y, new_beta_tilde)
            return take

    def solve(self):
        max_iterations = 2_000_000
        iterations = 0
        eps = self.epsilon

        while iterations < max_iterations:
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
                    
                    cost_val = self._cost(x, np.array([y]))[0]
                    new_beta_tilde = cost_val - alpha_prime - eps
                    
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

        # Reconstruct sparse total cost and output dictionary for compatibility
        out_mu = defaultdict(lambda: defaultdict(float))
        cost = 0.0
        for x in range(self.N_X):
            if self.is_sparse:
                for idx, y in enumerate(self.neighbors[x]):
                    mass = self.mu_arrs[x][idx]
                    if mass > TOL:
                        out_mu[x][y] = mass
                        cost += mass * self._cost_raw(x, y)
            else:
                for y, mass in self.mu_dict[x].items():
                    if mass > TOL:
                        out_mu[x][y] = mass
                        cost += mass * self._cost_raw(x, y)
                        
        return out_mu, cost, iterations
