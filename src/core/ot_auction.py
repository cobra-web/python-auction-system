import numpy as np
from collections import defaultdict

TOL = 1e-7

class AuctionOT:
    def __init__(self, X_pts, Y_pts, mu_X, mu_Y, epsilon=None,
                 allowed_edges=None, initial_beta=None, normalize=True):
        self.X_pts = np.array(X_pts, dtype=float)
        self.Y_pts = np.array(Y_pts, dtype=float)
        self.N_X = len(self.X_pts)
        self.N_Y = len(self.Y_pts)

        # Compute max_c based on bounding boxes to avoid dense NxN computation
        if normalize:
            min_X, max_X = np.min(self.X_pts, axis=0), np.max(self.X_pts, axis=0)
            min_Y, max_Y = np.min(self.Y_pts, axis=0), np.max(self.Y_pts, axis=0)
            max_dist_sq = np.sum((np.maximum(max_X, max_Y) - np.minimum(min_X, min_Y))**2)
            self.max_c = max_dist_sq if max_dist_sq > 0 else 1.0
        else:
            self.max_c = 1.0

        self.mu_X = np.array(mu_X, dtype=float)
        self.mu_Y = np.array(mu_Y, dtype=float)

        mass_X = float(self.mu_X.sum())
        mass_Y = float(self.mu_Y.sum())
        if abs(mass_X - mass_Y) > 1e-6 * max(1.0, mass_X):
            raise ValueError("Unbalanced OT problem. Sum of marginals must match.")

        self.epsilon = float(epsilon) if epsilon is not None else 1e-3
        
        # Sparse structures
        self.mu = defaultdict(lambda: defaultdict(float))
        self.assigned_Y = np.zeros(self.N_Y, dtype=float)
        self.unassigned_X = np.copy(self.mu_X)

        if initial_beta is not None:
            self.beta_diamond = np.array(initial_beta, dtype=float)
        else:
            self.beta_diamond = np.zeros(self.N_Y, dtype=float)
        
        self.beta_tilde = defaultdict(lambda: defaultdict(float))

        if allowed_edges is not None:
            nbrs = [[] for _ in range(self.N_X)]
            for edge in allowed_edges:
                x, y = int(edge[0]), int(edge[1])
                nbrs[x].append(y)
            self.neighbors = [np.array(sorted(set(v)), dtype=int) for v in nbrs]
        else:
            self.neighbors = None

    def _cost(self, x, ys):
        """Lazy calculation of normalized squared euclidean distance"""
        sq_dist = np.sum((self.X_pts[x] - self.Y_pts[ys])**2, axis=1)
        return sq_dist / self.max_c

    def _cost_raw(self, x, y):
        return np.sum((self.X_pts[x] - self.Y_pts[y])**2)

    def get_effective_beta(self):
        eff_beta = np.copy(self.beta_diamond)
        for y in range(self.N_Y):
            # Find any x that has mu[x][y] > TOL
            active_x = [x for x in self.mu.keys() if self.mu[x].get(y, 0.0) > TOL]
            if active_x:
                eff_beta[y] = max(eff_beta[y], max(self.beta_tilde[x].get(y, 0.0) for x in active_x))
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

        # Extract active assignments for y in ys to find occupied slots
        xp_indices = []
        y_idx_local = []
        mu_vals = []
        beta_tilde_vals = []
        
        for idx, y in enumerate(ys):
            for xp in self.mu.keys():
                if xp != x and self.mu[xp].get(y, 0.0) > TOL:
                    xp_indices.append(xp)
                    y_idx_local.append(idx)
                    mu_vals.append(self.mu[xp][y])
                    beta_tilde_vals.append(self.beta_tilde[xp][y])

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
            
            self.mu[x][y] += take
            self.assigned_Y[y] += take
            self.unassigned_X[x] -= take
            self.beta_tilde[x][y] = new_beta_tilde
            return take
        else:
            avail = self.mu[owner_xp].get(y, 0.0)
            take = min(amount, avail)
            if take <= TOL: 
                return 0.0
            
            self.mu[owner_xp][y] -= take
            self.mu[x][y] += take
            self.unassigned_X[owner_xp] += take
            self.unassigned_X[x] -= take
            self.beta_tilde[x][y] = new_beta_tilde
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

        # Reconstruct sparse total cost
        cost = sum(self.mu[x][y] * self._cost_raw(x, y) for x in self.mu for y in self.mu[x])
        return self.mu, cost, iterations
