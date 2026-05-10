"""
Demo & Tests for the Auction Algorithm (LAP + OT)
==================================================
Run this file to verify both implementations against known solutions
and get a feel for how the algorithms behave.

Usage:
    python main.py
"""

import numpy as np
from lap_auction import auction_lap, verify_optimality
from ot_auction import auction_ot, verify_ot


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1: Linear Assignment Problem (LAP)
# ═══════════════════════════════════════════════════════════════════════════════

def demo_lap_simple():
    """
    Tiny hand-checkable example.
    Cost matrix:
        y0  y1  y2
    x0 [  4,  1,  3 ]
    x1 [  2,  0,  5 ]
    x2 [  3,  2,  2 ]

    Optimal assignment: x0→y1, x1→y0, x2→y2  (cost = 1+2+2 = 5)
    """
    print("=" * 60)
    print("LAP Demo: Small hand-checkable example")
    print("=" * 60)

    cost = np.array([
        [4, 1, 3],
        [2, 0, 5],
        [3, 2, 2],
    ], dtype=float)

    assignment, total = auction_lap(cost)

    print(f"Cost matrix:\n{cost}\n")
    print(f"Assignment: {assignment}  (x_i → y_j)")
    print(f"Total cost: {total}")
    print(f"Optimal?    {verify_optimality(cost, assignment)}")
    print()


def demo_lap_random(N: int = 20, seed: int = 42):
    """
    Random square cost matrix, verified against scipy's Hungarian method.
    """
    print("=" * 60)
    print(f"LAP Demo: Random {N}×{N} cost matrix")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    cost = rng.integers(1, 100, size=(N, N)).astype(float)

    assignment, total = auction_lap(cost)
    optimal = verify_optimality(cost, assignment)

    print(f"Assignment: {assignment}")
    print(f"Total cost: {total:.1f}")
    print(f"Optimal?    {optimal}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2: Optimal Transport (OT)
# ═══════════════════════════════════════════════════════════════════════════════

def demo_ot_simple():
    """
    Small OT example with non-uniform masses.

    Two sources X = {x0, x1} with masses μ_X = [3, 2]
    Three targets Y = {y0, y1, y2} with masses μ_Y = [2, 1, 2]
    Total mass: 5

    Cost matrix:
          y0  y1  y2
    x0  [  1,  3,  5 ]
    x1  [  4,  2,  1 ]
    """
    print("=" * 60)
    print("OT Demo: Small example with non-uniform masses")
    print("=" * 60)

    cost = np.array([
        [1, 3, 5],
        [4, 2, 1],
    ], dtype=float)

    mu_x = np.array([3, 2])
    mu_y = np.array([2, 1, 2])

    coupling, total = auction_ot(cost, mu_x, mu_y)
    result = verify_ot(cost, mu_x, mu_y, coupling)

    print(f"Cost matrix:\n{cost}")
    print(f"μ_X = {mu_x},  μ_Y = {mu_y}")
    print(f"\nOptimal coupling μ(x,y):\n{coupling}")
    print(f"\nTotal transport cost: {total:.4f}")
    print(f"Feasible?            {result['feasible']}")
    if result['optimal_cost'] is not None:
        print(f"LP optimal cost:     {result['optimal_cost']:.4f}")
        print(f"Gap:                 {result['gap']:.2e}")
    print()


def demo_ot_1d_distributions():
    """
    Intuitive 1D example: moving mass from one histogram to another.

    Imagine 4 positions on a line: {0, 1, 2, 3}
    Source distribution:  [3, 1, 0, 2]  (heavy left)
    Target distribution:  [0, 2, 2, 2]  (heavy right)
    Cost: squared Euclidean distance  c(x,y) = (x-y)²

    We expect mass to flow from left to right.
    """
    print("=" * 60)
    print("OT Demo: 1D histogram transport (intuitive)")
    print("=" * 60)

    positions = np.array([0, 1, 2, 3])
    mu_x = np.array([3, 1, 0, 2])
    mu_y = np.array([0, 2, 2, 2])

    # Cost: squared Euclidean distance
    cost = (positions[:, None] - positions[None, :]) ** 2
    cost = cost.astype(float)

    print(f"Positions: {positions}")
    print(f"μ_X (source): {mu_x}")
    print(f"μ_Y (target): {mu_y}")
    print(f"Cost (squared distance):\n{cost}\n")

    coupling, total = auction_ot(cost, mu_x, mu_y)
    result = verify_ot(cost, mu_x, mu_y, coupling)

    print(f"Optimal coupling:\n{coupling}")
    print(f"\nTotal cost: {total:.4f}")
    print(f"Feasible?   {result['feasible']}")
    if result['optimal_cost'] is not None:
        print(f"LP cost:    {result['optimal_cost']:.4f}")
        print(f"Gap:        {result['gap']:.2e}")

    # Human-readable interpretation
    print("\nTransport plan (x → y: amount):")
    N_x, N_y = coupling.shape
    for x in range(N_x):
        for y in range(N_y):
            if coupling[x, y] > 0:
                print(f"  position {x} → position {y}: {coupling[x, y]} units")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3: Connection between LAP and OT
# ═══════════════════════════════════════════════════════════════════════════════

def demo_lap_as_ot():
    """
    Show that LAP is a special case of OT with uniform unit masses.
    μ_X = μ_Y = [1, 1, ..., 1]

    The OT coupling should be a permutation matrix (0/1 entries),
    matching the LAP solution.
    """
    print("=" * 60)
    print("Connection: LAP as special case of OT (μ_X = μ_Y = 1)")
    print("=" * 60)

    rng = np.random.default_rng(0)
    N = 5
    cost = rng.integers(1, 20, size=(N, N)).astype(float)
    mu = np.ones(N, dtype=int)

    # Solve as LAP
    lap_assign, lap_cost = auction_lap(cost)

    # Solve as OT
    coupling, ot_cost = auction_ot(cost, mu, mu)

    print(f"Cost matrix:\n{cost}\n")
    print(f"LAP assignment: {lap_assign}  (cost={lap_cost:.1f})")
    print(f"OT coupling:\n{coupling}")
    print(f"OT cost: {ot_cost:.1f}")
    print(f"Costs match: {np.isclose(lap_cost, ot_cost)}")
    print(f"Coupling is permutation matrix: {np.all(coupling <= 1) and np.all(coupling >= 0)}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  Auction Algorithm: LAP + OT  |  Schmitzer & Schnörr 2013")
    print("█" * 60 + "\n")

    demo_lap_simple()
    demo_lap_random(N=20)
    demo_ot_simple()
    demo_ot_1d_distributions()
    demo_lap_as_ot()

    print("All demos completed.")
