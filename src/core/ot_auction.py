import numpy as np

TOL = 1e-7

class AuctionOT:
    """
    Forward auction implementing strict 2D dual variable splitting: 
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
        self.mu = np.zeros((self.N_X, self.N_Y), dtype=float)

        # --- 2D Dual Variable Tracking ---
        # 1. beta_diamond tracks the price of unassigned (free) capacity in y.
        if initial_beta is not None:
            self.beta_diamond = np.array(initial_beta, dtype=float)
        else:
            self.beta_diamond = np.zeros(self.N_Y, dtype=float)
        
        # 2. y_chunks tracks beta_tilde. 
        # For each y, we maintain a list of [beta_tilde, x_owner, mass_amount]
        self.y_chunks = [[] for _ in range(self.N_Y)]
        self.free_Y = np.copy(self.mu_Y)

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
            if len(self.y_chunks[y]) > 0:
                max_tilde = max(chunk[0] for chunk in self.y_chunks[y])
                eff_beta[y] = max(eff_beta[y], max_tilde)
        return eff_beta

    def _admissible(self, x):
        if self.neighbors is None:
            return np.arange(self.N_Y)
        return self.neighbors[x]

    def _sorted_slots(self, x):
        """
        Build Pi(x): Evaluates free capacity via beta(diamond, y) AND 
        occupied capacity via beta_tilde(x', y). 
        """
        ys = self._admissible(x)
        if ys.size == 0:
            return None

        candidates = []
        
        for y in ys:
            # Evaluate unassigned capacity using beta(diamond, y)
            free = self.free_Y[y]
            if free > TOL:
                slack = self.C[x, y] - self.beta_diamond[y]
                # chunk_idx = -1 denotes free space
                candidates.append((slack, free, y, -1))
            
            # Evaluate specific occupied chunks using beta_tilde(x', y)
            for i, chunk in enumerate(self.y_chunks[y]):
                b_tilde, x_owner, mass = chunk
                if x_owner != x and mass > TOL:
                    slack = self.C[x, y] - b_tilde
                    candidates.append((slack, mass, y, i))

        if not candidates:
            return None

        # Sort all candidates by reduced cost (slack) ascending. 
        # Because slack = c - beta, sorting ascending naturally places the chunks 
        # with the HIGHEST beta (most optimal to displace) at the front of the list.
        candidates.sort(key=lambda item: item[0])
        
        slacks = np.array([c[0] for c in candidates])
        caps = np.array([c[1] for c in candidates])
        ys_arr = np.array([c[2] for c in candidates], dtype=int)
        chunk_indices = np.array([c[3] for c in candidates], dtype=int)
        
        return ys_arr, slacks, caps, chunk_indices

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

    def _place(self, x, y, chunk_idx, amount, new_beta_tilde):
        """Displaces a specific fraction of mass and establishes a new beta_tilde."""
        # Guard against float dust placements entirely
        if amount <= TOL: 
            return 0.0

        if chunk_idx == -1:
            # Consuming free capacity
            take = min(amount, self.free_Y[y])
            if take <= TOL: 
                return 0.0
            
            self.free_Y[y] -= take
            self.mu[x, y] += take
            self.y_chunks[y].append([new_beta_tilde, x, take])
            return take
        else:
            # Displacing a specific owned chunk
            chunk = self.y_chunks[y][chunk_idx]
            take = min(amount, chunk[2])
            if take <= TOL: 
                return 0.0
            
            chunk[2] -= take               # Reduce mass of displaced chunk
            self.mu[chunk[1], y] -= take   # Update global mu for displaced owner
            
            self.mu[x, y] += take          # Update global mu for new owner
            self.y_chunks[y].append([new_beta_tilde, x, take]) # Register new beta_tilde
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
                ys, slacks, caps, chunk_indices = slots

                alpha_prime, m_idx = self._marginal_alpha_prime(slacks, caps, r)

                remaining = r
                touched_ys = set()  # Track which y's we modify to clean them safely later
                
                for i in range(m_idx + 1):
                    if remaining <= TOL:
                        break
                    y = ys[i]
                    chunk_idx = chunk_indices[i]
                    
                    new_beta_tilde = self.C[x, y] - alpha_prime - eps
                    
                    placed = self._place(x, y, chunk_idx, remaining, new_beta_tilde)
                    if placed > 0:
                        touched_ys.add(y)
                        remaining -= placed
                        moved_this_sweep += placed

                # IMMEDIATE CLEANUP: Prevent list bloat without breaking indices mid-loop
                for ty in touched_ys:
                    self.y_chunks[ty] = [c for c in self.y_chunks[ty] if c[2] > TOL]

            iterations += 1
            if iterations >= max_iterations:
                break

            if moved_this_sweep <= TOL:
                print(f"[AuctionOT] Deadlock at iter {iterations}; unassigned = {total_unassigned:.3e}.")
                break

        cost = float(np.sum(self.mu * self.C))
        return self.mu, cost, iterations
