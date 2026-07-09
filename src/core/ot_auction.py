import numpy as np

# The SparseNeighborhood import is kept for compatibility with the rest of your
# code base, but this solver no longer *depends* on its (unknown) interface: it
# builds its own adjacency from `allowed_edges` so behaviour is deterministic.
try:
    from src.utils.data_structures import SparseNeighborhood  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    SparseNeighborhood = None


# ===== GLOBAL TOLERANCE PARAMETER =====
# Any mass amount below this is treated as zero (no bidding, no displacement).
# Used for BOTH mass comparisons and capacity snapping so that repeated
# float += / -= cannot accumulate into a phantom "unassigned" residual that
# keeps the auction alive forever (this was one source of your infinite loop).
TOL = 1e-7


class AuctionOT:
    """Forward auction for balanced discrete optimal transport.

    Faithful to B. Schmitzer & C. Schnoerr, "A Hierarchical Approach to Optimal
    Transport" (SSVM 2013), Section 3, which itself follows Bertsekas & Castanon,
    "The auction algorithm for the transportation problem" (1989).

    Sign convention (Schmitzer's, Sec. 3): signs are flipped vs. the classical
    auction so that the *lowest* reduced cost wins. The dual variable beta(y)
    is a price on sinks; an unassigned source x picks the y minimising the
    reduced cost  s(x, y) = c(x, y) - beta(y).  This matches your original
    `reduced_cost = C[x, y] - beta[y]` convention.

    NOTE on the beta / beta-tilde split (paper Eq. 10):
        The generalised OT auction distinguishes a per-pair price beta~(x', y)
        for occupied atoms and beta(diamond, y) for still-free mass at y.
        This implementation collapses those into a single price beta(y) per sink
        -- exactly the relation the paper gives,
            beta(y) = max_{x': mu(x',y)>0} beta~(x', y)   if y is saturated,
            beta(y) = beta(diamond, y)                    otherwise.
        A single price is the standard Bertsekas/Castanon transportation auction
        and is correct for the *balanced dense* problem; the only cost is that a
        sink's price can transiently disturb a retained owner's epsilon-CS by at
        most epsilon, which is healed as epsilon is scaled down. If you ever need
        the full split (e.g. strongly unbalanced problems) it would live here.
    """

    def __init__(self, cost_matrix, mu_X, mu_Y, epsilon=None,
                 allowed_edges=None, initial_beta=None, normalize=True):
        self.C_raw = np.array(cost_matrix, dtype=float)

        # --- Normalisation is now single-sourced ---------------------------------
        # BUG FIX: previously BOTH this class and EpsScalingManager divided C by
        # its max, so the cost matrix was normalised twice and the epsilon
        # schedule no longer matched the scale the solver actually ran on.
        # EpsScalingManager now normalises once and calls this with normalize=False.
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

        # --- Feasibility guard ---------------------------------------------------
        # The auction assigns ALL source mass. If total supply != total demand it
        # can NEVER drive unassigned mass to zero -> guaranteed infinite loop.
        # This is the single most common cause of "hits max_iterations with
        # leftover mass". Schmitzer (Sec. 2) requires sum(mu_X) == sum(mu_Y).
        mass_X = float(self.mu_X.sum())
        mass_Y = float(self.mu_Y.sum())
        if abs(mass_X - mass_Y) > 1e-6 * max(1.0, mass_X):
            raise ValueError(
                f"Unbalanced OT problem: sum(mu_X)={mass_X:.6f} != "
                f"sum(mu_Y)={mass_Y:.6f}. Balance the marginals before solving; "
                f"otherwise the auction cannot place all source mass and will "
                f"loop until max_iterations."
            )

        if epsilon is None:
            epsilon = 1e-3
        self.epsilon = float(epsilon)

        self.mu = np.zeros((self.N_X, self.N_Y), dtype=float)

        if initial_beta is not None:
            self.beta = np.array(initial_beta, dtype=float)
        else:
            self.beta = np.zeros(self.N_Y, dtype=float)

        # --- Adjacency (admissible targets N(x)) ---------------------------------
        # Dense problem -> every y is admissible. Sparse -> build per-source lists.
        if allowed_edges is not None:
            nbrs = [[] for _ in range(self.N_X)]
            for edge in allowed_edges:
                x, y = int(edge[0]), int(edge[1])
                nbrs[x].append(y)
            self.neighbors = [np.array(sorted(set(v)), dtype=int) for v in nbrs]
        else:
            self.neighbors = None

    # ------------------------------------------------------------------ helpers --
    def _admissible(self, x):
        """N(x): admissible sinks for source x (dense = all)."""
        if self.neighbors is None:
            return np.arange(self.N_Y)
        return self.neighbors[x]

    def _sorted_slots(self, x):
        """Build the ordered candidate list Pi(x) of paper Eq. (10)-(11).

        Each admissible sink y contributes one entry:
            reduced cost  s = c(x, y) - beta(y)          (Eq. 2 / Eq. 10)
            capacity that x could grab here:
                cap(y) = mu_Y[y] - mu[x, y]
                       = free space + mass currently owned by *others*
                         (x may never displace itself -- paper's rule and yours).
        Entries with no grabbable capacity are dropped. The remaining entries are
        returned sorted ascending by reduced cost, i.e. exactly the ordering
        required by Eq. (11):  s(y_1) <= s(y_2) <= ... .

        Returns (ys, slacks, caps) or None if x is fully blocked.
        """
        ys = self._admissible(x)
        if ys.size == 0:
            return None

        slack = self.C[x, ys] - self.beta[ys]
        cap = self.mu_Y[ys] - self.mu[x, ys]

        # Snap sub-tolerance capacities to exactly zero (float-underflow guard).
        cap = np.where(cap < TOL, 0.0, cap)
        mask = cap > 0.0
        if not np.any(mask):
            return None

        ys, slack, cap = ys[mask], slack[mask], cap[mask]
        order = np.argsort(slack, kind="stable")
        return ys[order], slack[order], cap[order]

    @staticmethod
    def _marginal_alpha_prime(slacks, caps, demand):
        """Compute alpha'(x) for an OT bid -- the integer-m rule, Eq. (12).

        In the LAP a source needs exactly one atom, so alpha (Eq. 6) and
        alpha' (Eq. 7) are simply the first two entries of Pi(x). In OT a source
        with unassigned mass `demand` may consume several of the cheapest sinks
        at once. Schmitzer: "one will determine an integer m > 1 such that
        alpha'(x) = c(x, y_m) - beta(x'_m, y_m)."  alpha'(x) is therefore the
        reduced cost of the *marginal* atom just past the point where the
        source's demand is exhausted -- NOT merely the second-cheapest sink.

        Using the true marginal (rather than a per-atom "second best") is what
        stops the price war / micro-bidding stall: the bid increment reflects the
        real gap to the next alternative for the whole chunk of mass.

        Returns (alpha_prime, m_idx) where m_idx is the index of the last sink
        that receives mass in this bid (fill sinks 0..m_idx inclusive).
        """
        cum = 0.0
        n = len(caps)
        for i in range(n):
            cum += caps[i]
            if cum >= demand - TOL:
                if cum > demand + TOL:
                    # Demand runs out *inside* sink i: leftover capacity remains
                    # at sink i, so the next-best alternative atom sits at s(y_i).
                    return slacks[i], i
                if i + 1 < n:
                    # Demand met exactly at the boundary of sink i: the marginal
                    # alternative is the next sink. (LAP case: i=0, m=2.)
                    return slacks[i + 1], i
                # Boundary, but no further sink exists.
                return slacks[i], i
        # Total admissible capacity < demand: take everything we can; the marginal
        # is the worst sink we touch. Any leftover demand stays unassigned and
        # will re-bid next sweep (or trip the deadlock guard if truly infeasible).
        return slacks[-1], n - 1

    def _place(self, x, y, amount):
        """Move up to `amount` of x's unassigned mass into sink y.

        Fills free capacity first, then displaces mass owned by *other* sources
        (conservation of mass: displacement is a transfer, the displaced source
        simply reappears as unassigned on the next sweep because we recompute
        unassigned = mu_X - mu.sum(axis=1) from scratch). Column load can never
        exceed mu_Y[y]: `take_free` is capped by free space and displacement does
        not change the column total. Returns the amount actually placed.
        """
        if amount <= TOL:
            return 0.0

        load = float(np.sum(self.mu[:, y]))
        free = self.mu_Y[y] - load
        if free < 0.0:
            free = 0.0

        take_free = min(amount, free)
        self.mu[x, y] += take_free
        placed = take_free
        remaining = amount - take_free

        if remaining > TOL:
            for xp in range(self.N_X):
                if xp == x:  # a source may never displace itself
                    continue
                owned = self.mu[xp, y]
                if owned <= TOL:
                    continue
                d = min(owned, remaining)
                self.mu[xp, y] -= d
                self.mu[x, y] += d
                placed += d
                remaining -= d
                if remaining <= TOL:
                    break

        return placed

    # -------------------------------------------------------------------- solve --
    def solve(self):
        """Gauss-Seidel forward auction with a fixed epsilon.

        Returns (mu, cost, iterations). EpsScalingManager calls this repeatedly
        with a decreasing epsilon, warm-starting beta each time.
        """
        max_iterations = 2_000_000
        iterations = 0
        eps = self.epsilon

        while iterations < max_iterations:
            assigned = np.sum(self.mu, axis=1)
            unassigned = self.mu_X - assigned
            unassigned = np.where(unassigned < TOL, 0.0, unassigned)  # snap
            total_unassigned = float(unassigned.sum())

            # Convergence: all source mass placed (marginals satisfied to TOL).
            if total_unassigned <= TOL:
                break

            moved_this_sweep = 0.0

            for x in range(self.N_X):
                r = unassigned[x]
                if r <= TOL:
                    continue

                slots = self._sorted_slots(x)
                if slots is None:
                    # x owns all reachable capacity yet still wants more: blocked.
                    continue
                ys, slacks, caps = slots

                # --- Bidding phase (Eqs. 6-8, 10-12) ---------------------------
                alpha_prime, m_idx = self._marginal_alpha_prime(slacks, caps, r)

                remaining = r
                for i in range(m_idx + 1):
                    if remaining <= TOL:
                        break
                    y = ys[i]

                    # Dual update, Eqs. (8)-(9):
                    #   beta(y) := c(x, y) - alpha'(x) - epsilon
                    # This lowers beta by (alpha'(x) - s(x,y)) + epsilon >= epsilon,
                    # so beta is MONOTONE DECREASING (>= epsilon per accepted bid),
                    # which is what guarantees termination. Afterwards x's reduced
                    # cost at y equals alpha'(x) + epsilon, i.e. x is epsilon-happy
                    # (epsilon-complementary slackness, Schmitzer Sec. 3).
                    new_beta = self.C[x, y] - alpha_prime - eps
                    if new_beta < self.beta[y]:
                        self.beta[y] = new_beta

                    # --- Assignment phase: grab the mass -----------------------
                    placed = self._place(x, y, remaining)
                    remaining -= placed
                    moved_this_sweep += placed

                iterations += 1
                if iterations >= max_iterations:
                    break

            if moved_this_sweep <= TOL:
                print(
                    f"[AuctionOT] No mass moved in a full sweep at iter "
                    f"{iterations}; remaining unassigned = {total_unassigned:.3e}. "
                    f"The (sparse) neighbourhood is likely infeasible."
                )
                break

        cost = float(np.sum(self.mu * self.C))
        return self.mu, cost, iterations
