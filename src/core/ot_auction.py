import numpy as np

TOL = 1e-7

class AuctionOT:
    """
    Forward auction implementing strict 2D dual variable splitting using dense matrices.
    beta_tilde(x, y) for occupied mass and beta(diamond, y) for free capacity.
    """

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
        # mu tracks exactly how much mass x owns in y
        self.mu = np.zeros((self.N_X, self.N_Y), dtype=float)

        # beta_diamond tracks the baseline price of unassigned (free) capacity in y
        if initial_beta is not None:
            self.beta_diamond = np.array(initial_beta, dtype=float)
        else:
            self.beta_diamond = np.zeros(self.N_Y, dtype=float)
        
        # beta_tilde tracks the price of occupied capacity for each (x, y) pair
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
        """Returns the effective 1D beta(y) for warm-starting the next epsilon phase."""
        eff_beta = np.copy(self.beta_diamond)
        for y in range(self.N_Y):
            # Find which sources actually own mass in y
            active_x = np.where(self.mu[:, y] > TOL)[0]
            if len(active_x) > 0:
                eff_beta[y] = max(eff_beta[y], np.max(self.beta_tilde[active_x, y]))
        return eff_beta

    def _admissible(self, x):
        if self.neighbors is None:
            return np.arange(self.N_Y)
        return self.neighbors[x]

    def _sorted_slots(self, x):
        """
        Build Pi(x): Evaluates free capacity via beta(diamond, y) AND 
        occupied capacity via beta_tilde(x', y) using vectorized lookups.
        """
        ys = self._admissible(x)
        if ys.size == 0:
            return None

        candidates = []
        
        # 1. Evaluate Free Capacity
        # Calculate current free space across all admissible y
        free_Y = self.mu_Y[ys] - np.sum(self.mu[:, ys], axis=0)
        free_mask = free_Y > TOL
        
        if np.any(free_mask):
            valid_ys = ys[free_mask]
            slacks = self.C[x, valid_ys] - self.beta_diamond[valid_ys]
            caps = free_Y[free_mask]
            for i in range(len(valid_ys)):
                # -1 indicates this candidate is free capacity
                candidates.append((slacks[i], caps[i], valid_ys[i], -1))

        # 2. Evaluate Occupied Capacity
        # Look at the mu matrix for admissible targets
        mu_sub = self.mu[:, ys]
        
        # Find coordinates (x', y_idx) where mass exists
        xp_indices, y_indices = np.nonzero(mu_sub > TOL)
        
        # Filter out x itself (a source cannot displace its own mass)
        other_mask = xp_indices != x
        xp_indices = xp_indices[other_mask]
        y_indices = y_indices[other_mask]

        if len(xp_indices) > 0:
            actual_ys = ys[y_indices]
            # Vectorized slack calculation using the beta_tilde matrix
            slacks = self.C[x, actual_ys] - self.beta_tilde[xp_indices, actual_ys]
            caps = mu_sub[xp_indices, y_indices]
            for i in range(len(xp_indices)):
                candidates.append((slacks[i], caps[i], actual_ys[i], xp_indices[i]))

        if not candidates:
            return None

        # Sort ascending by slack (lowest reduced cost wins)
        candidates.sort(key=lambda item: item[0])
        
        slacks = np.array([c[0] for c in candidates])
        caps = np.array([c[1] for c in candidates])
        ys_arr = np.array([c[2] for c in candidates], dtype=int)
        xp_arr = np.array([c[3] for c in candidates], dtype=int)
        
        return ys_arr, slacks, caps, xp_arr

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
        """Moves mass and strictly assigns the new beta_tilde price in the matrix."""
        if amount <= TOL:
            return 0.0
            
        if owner_xp == -1:
            # Consuming free capacity
            free = self.mu_Y[y] - np.sum(self.mu[:, y])
            take = min(amount, free)
            if take <= TOL: 
                return 0.0
            
            self.mu[x, y] += take
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
            self.beta_tilde[x, y] = new_beta_tilde
            return take

    def solve(self):
        max_iterations = 2_000_000
        iterations = 0
        eps = self.epsilon

        while iterations < max_iterations:
            assigned = np.sum(self.mu, axis=1)
            unassigned = self.mu_X - assigned
            unassigned = np.where(unassigned < TOL, 0.0, unassigned)
            total_unassigned = float(unassigned.sum())

            if total_unassigned <= TOL:
                break

            moved_this_sweep = 0.0

            for x in range(self.N_X):
                r = unassigned[x]
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
