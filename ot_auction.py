"""
Auction Algorithm for Optimal Transport (OT)
============================================
Based on: Schmitzer & Schnörr, "A Hierarchical Approach to Optimal Transport", SSVM 2013
          and Bertsekas & Castanon, "The Auction Algorithm for the Transportation Problem" (1989)

Paper reference: Section 3 – "The Auction Algorithm for Optimal Transport"
                 Section 2 – Equations (4a), (4b), (5)

The OT Problem:
  Given mass distributions μ_X on X and μ_Y on Y (with equal total mass),
  find a coupling μ(x,y) ≥ 0 that:
    - respects the marginals: Σ_y μ(x,y) = μ_X(x),  Σ_x μ(x,y) = μ_Y(y)
    - minimises total transport cost: Σ_{x,y} c(x,y) · μ(x,y)

Key difference from LAP:
  - Each source x ∈ X has a mass μ_X(x) that can be split across multiple targets.
  - Each target y ∈ Y has a capacity μ_Y(y).
  - Dual variables β̃(x,y) are per-pair, not per-target (Section 3, OT extension).
  - x can bid for multiple targets simultaneously.

Implementation strategy:
  We use integer mass distributions and reduce OT to LAP by "expanding" each
  node into μ_X(x) / μ_Y(y) copies (made implicit via careful bookkeeping).
  This follows the approach of Bertsekas & Castanon (1989), referenced in [4].
"""

import numpy as np
from collections import defaultdict


def auction_ot(
    cost: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    eps: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Solve the Optimal Transport problem via a generalised Auction Algorithm.

    State of the algorithm (Section 3, OT extension):
      - coupling μ(x,y): how much mass flows from x to y
      - dual variables β̃(x,y): per-pair prices for already-coupled (x,y)
      - dual variable β̃(◇,y): price for "free" capacity at y (not yet bid on)
      - α(x) is held implicitly via the ordered list Π(x)  (Eq. 10, 11)

    Optimality condition (Eq. 5):
      μ(x,y) > 0  ⟹  α(x) + β(y) = c(x,y)
      where β(y) = max_{x: μ(x,y)>0} β̃(x,y)  if y is full,
                 = β̃(◇,y)                      otherwise

    Args:
        cost:   2D array (N_x, N_y), cost[x,y] = c(x,y)
        mu_x:   1D integer array of length N_x, source masses
        mu_y:   1D integer array of length N_y, target masses
        eps:    bid increment (see lap_auction for details)

    Returns:
        coupling:    2D array (N_x, N_y), the optimal transport plan
        total_cost:  Σ c(x,y) · coupling[x,y]
    """
    N_x, N_y = cost.shape
    mu_x = np.array(mu_x, dtype=int)
    mu_y = np.array(mu_y, dtype=int)

    assert mu_x.sum() == mu_y.sum(), "Total mass must be equal: Σ μ_X = Σ μ_Y"

    if eps is None:
        finite_vals = cost[np.isfinite(cost)]
        diffs = np.diff(np.unique(finite_vals))
        delta_c = diffs.min() if len(diffs) > 0 else 1.0
        eps = delta_c / (mu_x.sum() + 1)

    # ── State initialisation ─────────────────────────────────────────────────

    # μ(x,y): current coupling (transport plan)
    coupling = np.zeros((N_x, N_y), dtype=int)

    # Remaining supply and demand
    supply = mu_x.copy()   # how much mass x still needs to send
    demand = mu_y.copy()   # how much capacity y still has

    # β̃(x,y): dual price for the pair (x,y), only defined when coupling[x,y] > 0
    # Stored as a dict: (x,y) → price
    beta_pair: dict[tuple[int,int], float] = {}

    # β̃(◇,y): price for free capacity at y
    beta_free = np.zeros(N_y, dtype=float)

    # Track which x nodes still have unsent mass (exclude zero-mass sources)
    active_sources = set(int(x) for x in range(N_x) if supply[x] > 0)

    iteration = 0
    while active_sources:
        iteration += 1
        # bids[y] = list of (bid_value, x, amount_of_mass)
        bids: dict[int, list] = defaultdict(list)

        # ── BIDDING PHASE ────────────────────────────────────────────────────
        # Each active x builds the ordered list Π(x) (Eq. 10, 11):
        # Π(x) contains reduced costs c(x,y) - β̃(·,y) for all reachable y,
        # combining already-coupled targets and free-capacity targets.
        for x in active_sources:
            pi = []  # list of (reduced_cost, y, x_prime_or_diamond)

            for y in range(N_y):
                if np.isinf(cost[x, y]):
                    continue  # forbidden assignment

                if coupling[x, y] > 0:
                    # y is already partially coupled to x: use β̃(x,y)
                    rc = cost[x, y] - beta_pair[(x, y)]
                    pi.append((rc, y, 'coupled'))
                elif demand[y] > 0:
                    # y has free capacity: use β̃(◇,y)
                    rc = cost[x, y] - beta_free[y]
                    pi.append((rc, y, 'free'))
                # If demand[y]==0 and coupling[x,y]==0: y is full for others, skip

            if not pi:
                raise ValueError(f"Source x={x} has no feasible targets. Problem infeasible.")

            # Sort Π(x) in ascending order of reduced cost (Eq. 11)
            pi.sort(key=lambda t: t[0])

            # α(x) = first entry of Π(x)  (best reduced cost)
            alpha_x = pi[0][0]

            # Determine how much mass x can send in this round (Eq. 12):
            # Find the cut-off position m: all positions up to m get mass,
            # m is the first position with reduced cost > α(x).
            # The "second-best" α'(x) is the reduced cost at position m.
            m = 1
            while m < len(pi) and np.isclose(pi[m][0], alpha_x):
                m += 1  # ties: x can send to all equally good targets

            alpha_prime_x = pi[m][0] if m < len(pi) else alpha_x + eps

            # Send mass to the top-m targets, split as evenly as possible
            mass_to_send = supply[x]
            targets = pi[:m]
            mass_per_target = mass_to_send // len(targets)
            remainder = mass_to_send % len(targets)

            for i, (rc, y, kind) in enumerate(targets):
                amount = mass_per_target + (1 if i < remainder else 0)
                if amount == 0:
                    continue
                # Bid value (Eq. 8 generalised): b = c(x,y) - α'(x) - ε
                bid_val = cost[x, y] - alpha_prime_x - eps
                bids[y].append((bid_val, x, amount))

        # ── ASSIGNMENT PHASE ─────────────────────────────────────────────────
        # For each y that received bids, accept the lowest bid.
        # Update β̃ and coupling accordingly.
        for y, y_bids in bids.items():
            # Sort bids ascending: lowest bid wins (paper flips auction sign)
            y_bids.sort(key=lambda t: t[0])
            best_bid, x_star, amount = y_bids[0]

            # How much can y actually absorb?
            amount = min(amount, demand[y])
            if amount == 0:
                continue

            # Update coupling
            coupling[x_star, y] += amount
            supply[x_star] -= amount
            demand[y] -= amount

            # Update dual variable β̃
            if coupling[x_star, y] > 0:
                beta_pair[(x_star, y)] = best_bid
            if demand[y] == 0:
                # y is now full: β(y) = max β̃(x,y) over all x with coupling>0
                pass  # beta_pair already stores individual prices
            else:
                beta_free[y] = best_bid

            # Update active sources
            if supply[x_star] == 0:
                active_sources.discard(x_star)

    total_cost = float(np.sum(cost * coupling))
    return coupling, total_cost


# ── Convenience: β(y) as used in optimality condition (Eq. 5) ────────────────

def get_beta(coupling: np.ndarray, beta_pair: dict, beta_free: np.ndarray) -> np.ndarray:
    """
    Reconstruct β(y) from the dual state (Section 3, OT extension):
      β(y) = max_{x: μ(x,y)>0} β̃(x,y)   if y is fully assigned
           = β̃(◇,y)                       otherwise
    """
    N_y = beta_free.shape[0]
    beta = beta_free.copy()
    for (x, y), val in beta_pair.items():
        if coupling[x, y] > 0:
            beta[y] = max(beta[y], val)
    return beta


# ── Verification helper ───────────────────────────────────────────────────────

def verify_ot(
    cost: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    coupling: np.ndarray,
) -> dict:
    """
    Verify feasibility and near-optimality of a coupling.

    Feasibility checks (Eq. 4a):
      1. coupling ≥ 0
      2. Σ_y coupling[x,y] = μ_X(x)  for all x
      3. Σ_x coupling[x,y] = μ_Y(y)  for all y

    Optimality: compare against scipy's exact solver.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching

    feasible = (
        np.all(coupling >= 0)
        and np.allclose(coupling.sum(axis=1), mu_x)
        and np.allclose(coupling.sum(axis=0), mu_y)
    )
    our_cost = float(np.sum(cost * coupling))

    # Ground truth via network simplex (for small problems)
    try:
        from scipy.optimize import linprog
        # Build LP: this gets expensive for large N, only for verification
        N_x, N_y = cost.shape
        c_flat = cost.flatten()
        # Equality constraints: marginals
        A_eq = np.zeros((N_x + N_y, N_x * N_y))
        b_eq = np.concatenate([mu_x, mu_y]).astype(float)
        for x in range(N_x):
            A_eq[x, x*N_y:(x+1)*N_y] = 1
        for y in range(N_y):
            A_eq[N_x+y, y::N_y] = 1
        res = linprog(c_flat, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)]*len(c_flat), method='highs')
        opt_cost = res.fun if res.success else None
    except Exception:
        opt_cost = None

    return {
        "feasible": feasible,
        "our_cost": our_cost,
        "optimal_cost": opt_cost,
        "gap": abs(our_cost - opt_cost) if opt_cost is not None else None,
    }
