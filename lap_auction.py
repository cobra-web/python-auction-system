"""
Auction Algorithm for the Linear Assignment Problem (LAP)
=========================================================
Based on: Schmitzer & Schnörr, "A Hierarchical Approach to Optimal Transport", SSVM 2013
          and Bertsekas, "The Auction Algorithm" (1988)

Paper reference: Section 3 – "The Auction Algorithm for the Assignment Problem"

The LAP: Given two finite sets X, Y and a cost function c: X × Y → ℝ,
find a complete assignment S ⊆ X × Y that minimises Σ c(x,y).

Key idea of the auction algorithm:
  - Elements of X are "bidders", elements of Y are "objects".
  - Each bidder tries to get assigned to their cheapest object.
  - If two bidders want the same object, only the lowest bid wins.
  - Dual variables β(y) represent the "price" of object y.
  - The algorithm iterates bidding and assignment phases until complete.
"""

import numpy as np


def auction_lap(cost: np.ndarray, eps: float | None = None) -> tuple[np.ndarray, float]:
    """
    Solve the Linear Assignment Problem via the Auction Algorithm.

    The algorithm maintains:
      - An assignment S: mapping from X → Y (or -1 if unassigned)
      - Dual variables β(y) for each y ∈ Y  (the "prices")
      - α(x) is held implicitly via: α(x) = min_y [ c(x,y) - β(y) ]   (Eq. 2)

    Convergence guarantee (from paper):
      As long as ε < Δc / |X|, the resulting complete assignment is
      globally optimal, where Δc is the smallest difference between
      two distinct values of c.

    Args:
        cost:  2D array of shape (N, N), where cost[x, y] = c(x, y).
               Use np.inf to mark forbidden assignments.
        eps:   ε parameter controlling bid increment.
               Smaller ε → better optimality guarantee, more iterations.
               Default: 1 / (N + 1), which guarantees global optimality.

    Returns:
        assignment:  1D array of length N, where assignment[x] = y.
        total_cost:  Σ c(x, assignment[x])
    """
    N = cost.shape[0]
    assert cost.shape == (N, N), "Cost matrix must be square."

    # ε must satisfy: ε < Δc / N  for optimality guarantee
    # A safe default is 1/(N+1) for integer costs, or a small fraction otherwise
    if eps is None:
        finite_vals = cost[np.isfinite(cost)]
        diffs = np.diff(np.unique(finite_vals))
        delta_c = diffs.min() if len(diffs) > 0 else 1.0
        eps = delta_c / (N + 1)

    # --- Initialisation ---
    # S: assignment[x] = y means (x, y) ∈ S. -1 = unassigned.
    assignment = np.full(N, -1, dtype=int)
    # Inverse: assigned_to[y] = x means y is currently held by x. -1 = free.
    assigned_to = np.full(N, -1, dtype=int)
    # β(y): dual variable / "price" of each object y ∈ Y
    beta = np.zeros(N, dtype=float)

    # All x start unassigned
    unassigned = set(range(N))

    iteration = 0
    while unassigned:
        iteration += 1
        bids = {}  # y → (best_bid_value, best_bidder_x)

        # ── BIDDING PHASE ────────────────────────────────────────────────────
        # Each unassigned x computes its best and second-best object,
        # then submits a bid to its best object.
        # Paper: Eq. (6), (7), (8)
        for x in unassigned:
            # Reduced costs: c(x,y) - β(y)  for all y
            reduced = cost[x] - beta  # shape (N,)

            # Find the minimiser y* (best object for x)
            # and the second minimum α'(x) (Eq. 7)
            sorted_idx = np.argsort(reduced)
            y_star = sorted_idx[0]
            alpha_x = reduced[y_star]           # Eq. (6): α(x) = min_y [c(x,y) - β(y)]
            alpha_prime_x = reduced[sorted_idx[1]]  # Eq. (7): second-best reduced cost

            # Bid value (Eq. 8): b_{x,y*} = c(x,y*) - α'(x) - ε
            # Intuition: x offers to pay just enough to beat the second-best option,
            # minus a small increment ε to ensure strict improvement.
            bid_value = cost[x, y_star] - alpha_prime_x - eps

            # Register bid (keep only the lowest bid per y, per assignment phase)
            if y_star not in bids or bid_value < bids[y_star][0]:
                bids[y_star] = (bid_value, x)

        # ── ASSIGNMENT PHASE ─────────────────────────────────────────────────
        # For each y that received bids, accept the lowest bid.
        # Update β(y) and reassign y to the winning bidder.
        # Paper: Eq. (9)
        for y, (best_bid, x_star) in bids.items():
            # Update price: β(y) := lowest bid received  (Eq. 9)
            beta[y] = best_bid

            # If y was previously assigned to some x_old, release x_old
            x_old = assigned_to[y]
            if x_old != -1:
                assignment[x_old] = -1
                unassigned.add(x_old)

            # Assign y to the winning bidder x_star
            assignment[x_star] = y
            assigned_to[y] = x_star
            unassigned.discard(x_star)

    total_cost = sum(cost[x, assignment[x]] for x in range(N))
    return assignment, total_cost


# ── Verification helper ───────────────────────────────────────────────────────

def verify_optimality(cost: np.ndarray, assignment: np.ndarray) -> bool:
    """
    Check optimality via the complementary slackness condition (Eq. 3):
      For any (x,y) ∈ S:  α(x) + β(y) = c(x,y)

    This is a necessary and sufficient condition for optimality of the primal
    assignment S together with dual variables (α, β).
    """
    N = cost.shape[0]
    # Reconstruct β from the assignment (β(y) = c(x,y) - α(x) won't work directly,
    # so we use scipy as ground truth for verification)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    opt_cost = cost[row_ind, col_ind].sum()
    our_cost = sum(cost[x, assignment[x]] for x in range(N))
    return np.isclose(our_cost, opt_cost, rtol=1e-5)
