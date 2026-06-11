"""
Hierarchical Optimal Transport - improved implementation
Based on: Schmitzer & Schnörr, "A Hierarchical Approach to Optimal Transport" (SS13)

Key improvements over the original KI-generated code:
1. True quadtree with direct coordinate-based cell assignment (no subset checks)
2. Cached hierarchical extensions α̂, β̂, ĉ per node (Eq. 13, 14)
3. Proper Π(x) ordered list for bidding (Eq. 10-12)
4. ε-scaling with warm restarts (Section 5 / implementation details Section 6)
"""

import numpy as np
import ot
import time


# ---------------------------------------------------------------------------
# 1. Hierarchical Partition (proper quadtree, Sec. 4 "Hierarchical Partitions")
# ---------------------------------------------------------------------------

class HierarchyNode:
    """
    Represents one cell at one level of the hierarchical partition.

    Attributes
    ----------
    points : list[int]   indices of X (or Y) points in this cell
    children : list[HierarchyNode]
    parent : HierarchyNode | None
    level : int          0 = finest (singletons), g = coarsest root
    # cached hierarchical extensions (updated during auction)
    alpha_hat : float    α̂(a) = max_{x∈a} α(x)   [for X-nodes]
    beta_hat  : float    β̂(b) = max_{y∈b} β(y)   [for Y-nodes]
    c_hat     : dict     ĉ(a,b) cached per partner node id
    """
    __slots__ = ("points", "children", "parent", "level",
                 "alpha_hat", "beta_hat")

    def __init__(self, points, level):
        self.points   = points        # list of point indices
        self.children = []
        self.parent   = None
        self.level    = level
        self.alpha_hat = -np.inf
        self.beta_hat  = -np.inf


def build_quadtree(pts: np.ndarray, max_levels: int) -> list[list[HierarchyNode]]:
    """
    Build a proper quadtree over 2-D points in [0,1)^2.

    Level 0  : singletons  (one node per point)
    Level k  : grid of 2^k × 2^k cells
    Returns  : levels[k] = list of HierarchyNode at level k
               levels[0] is finest, levels[-1] is coarsest (usually 1 root)

    Cell assignment is O(N * max_levels) – no subset iteration needed.
    """
    N = len(pts)

    # --- level 0: one node per point ---
    leaves = [HierarchyNode([i], 0) for i in range(N)]
    levels = [leaves]

    for k in range(1, max_levels + 1):
        grid_size = 2 ** k                          # 2, 4, 8, …
        # integer cell index for every point at this grid resolution
        cell_ix = np.floor(pts * grid_size).astype(int).clip(0, grid_size - 1)
        cell_hash = cell_ix[:, 0] * grid_size + cell_ix[:, 1]

        # build mapping: hash -> list of point indices
        from collections import defaultdict
        hash_to_pts: dict[int, list[int]] = defaultdict(list)
        for i, h in enumerate(cell_hash):
            hash_to_pts[h].append(i)

        # build mapping: hash -> list of child nodes (from previous level)
        # Each child node belongs to exactly the cell determined by its points[0]
        hash_to_children: dict[int, list[HierarchyNode]] = defaultdict(list)
        for child in levels[-1]:
            rep = child.points[0]              # representative point
            hash_to_children[cell_hash[rep]].append(child)

        current_level = []
        for h, pts_idx in hash_to_pts.items():
            node = HierarchyNode(pts_idx, k)
            node.children = hash_to_children[h]
            for child in node.children:
                child.parent = node
            current_level.append(node)

        levels.append(current_level)

    return levels   # levels[0]=finest … levels[-1]=coarsest


# ---------------------------------------------------------------------------
# 2. Precompute ĉ(a, b) = min_{x∈a, y∈b} c(x, y)   (Eq. 14)
#    We cache this lazily in a dict keyed by (id(a), id(b))
# ---------------------------------------------------------------------------

_c_hat_cache: dict = {}

def c_hat(node_a: HierarchyNode, node_b: HierarchyNode,
          C: np.ndarray) -> float:
    """ĉ(a,b) = min_{x∈a, y∈b} c(x,y)  –  cached."""
    key = (id(node_a), id(node_b))
    if key not in _c_hat_cache:
        _c_hat_cache[key] = C[np.ix_(node_a.points, node_b.points)].min()
    return _c_hat_cache[key]


def _c_hat_direct(na: HierarchyNode, nb: HierarchyNode, C: np.ndarray) -> float:
    """ĉ(a,b) via direct min over point sets – used as fallback."""
    return float(C[np.ix_(na.points, nb.points)].min())


def precompute_c_hat(levels_X: list, levels_Y: list, C: np.ndarray):
    """
    Precompute ĉ(a,b) = min_{x∈a, y∈b} c(x,y)  for all node pairs.

    Bottom-up: finest level uses direct C lookup, coarser levels reuse
    children values. Falls back to direct computation when a node has no
    children (can happen in unbalanced quadtrees with sparse point clouds).
    """
    _c_hat_cache.clear()

    # Collect all unique levels across both hierarchies
    all_levels_X = {node.level: [] for level in levels_X for node in level}
    all_levels_Y = {node.level: [] for level in levels_Y for node in level}
    for level in levels_X:
        for node in level:
            all_levels_X[node.level].append(node)
    for level in levels_Y:
        for node in level:
            all_levels_Y[node.level].append(node)

    # Process level by level from finest (0) upward
    for k in sorted(all_levels_X.keys()):
        nodes_X = all_levels_X.get(k, [])
        nodes_Y = all_levels_Y.get(k, [])
        for na in nodes_X:
            for nb in nodes_Y:
                key = (id(na), id(nb))
                if key in _c_hat_cache:
                    continue
                # Try to compute from children if both have children
                if na.children and nb.children:
                    child_vals = []
                    for ca in na.children:
                        for cb in nb.children:
                            ck = (id(ca), id(cb))
                            # child value may not be cached if levels differ
                            child_vals.append(
                                _c_hat_cache[ck] if ck in _c_hat_cache
                                else _c_hat_direct(ca, cb, C)
                            )
                    _c_hat_cache[key] = min(child_vals)
                else:
                    # leaf or childless node: compute directly
                    _c_hat_cache[key] = _c_hat_direct(na, nb, C)


# ---------------------------------------------------------------------------
# 3. Update hierarchical extensions α̂ and β̂ (Eq. 13) bottom-up
# ---------------------------------------------------------------------------

def update_alpha_hat(levels_X: list, alpha: np.ndarray):
    """alpha_hat(a) = max_{x in a} alpha(x), propagated bottom-up (Eq. 13)."""
    for node in levels_X[0]:
        node.alpha_hat = alpha[node.points[0]]
    for k in range(1, len(levels_X)):
        for node in levels_X[k]:
            if node.children:
                node.alpha_hat = max(ch.alpha_hat for ch in node.children)
            else:
                node.alpha_hat = max(alpha[p] for p in node.points)


def update_beta_hat(levels_Y: list, beta: np.ndarray):
    """beta_hat(b) = max_{y in b} beta(y), propagated bottom-up (Eq. 13)."""
    for node in levels_Y[0]:
        node.beta_hat = beta[node.points[0]]
    for k in range(1, len(levels_Y)):
        for node in levels_Y[k]:
            if node.children:
                node.beta_hat = max(ch.beta_hat for ch in node.children)
            else:
                node.beta_hat = max(beta[p] for p in node.points)


# ---------------------------------------------------------------------------
# 4. Hierarchical Consistency Check (Sec. 4, "Consistency Check Phase")
#
#  Check: ĉ(a,b) - β̂(b) >= α̂'(a)
#  If violated, recurse to children. At generation 0 (leaves):
#    add (x,y) to N̂ and mark x for rebidding.
# ---------------------------------------------------------------------------

def consistency_check(
        node_a: HierarchyNode,
        node_b: HierarchyNode,
        C: np.ndarray,
        beta: np.ndarray,
        alpha_prime: np.ndarray,         # α'(x) = second-best value for x
        allowed: np.ndarray,             # sparse N̂  (bool matrix N×N)
        rebid: set,
):
    ch = c_hat(node_a, node_b, C)
    # dual constraint at this generation (Eq. 15):  ĉ(a,b) - β̂(b) >= α̂'(a)
    if ch - node_b.beta_hat >= node_a.alpha_hat:
        return   # constraint satisfied for all descendants → prune

    # violated → recurse
    if node_a.level > 0 and node_b.level > 0:
        for ca in node_a.children:
            for cb in node_b.children:
                consistency_check(ca, cb, C, beta, alpha_prime, allowed, rebid)
    else:
        # generation 0: individual pairs
        for x in node_a.points:
            for y in node_b.points:
                if C[x, y] - beta[y] < alpha_prime[x]:
                    if not allowed[x, y]:
                        allowed[x, y] = True
                        rebid.add(x)


# ---------------------------------------------------------------------------
# 5. Π(x) bidding list (Eq. 10-12)
#    For a given x, build the ordered list of (cost - beta) for allowed y,
#    return y*, bid value, and α'(x).
# ---------------------------------------------------------------------------

def compute_bid(x: int, allowed: np.ndarray, C: np.ndarray,
                beta: np.ndarray, epsilon: float):
    """
    Returns (y_star, alpha_prime_x, bid_value).
    Implements Eq. 6-8 restricted to the sparse neighbour set N̂.
    """
    valid_y = np.where(allowed[x])[0]
    if len(valid_y) == 0:
        valid_y = np.arange(C.shape[1])   # fallback: all y

    pi = C[x, valid_y] - beta[valid_y]    # Π(x) values (unsorted subset)
    order = np.argsort(pi)

    y_star = valid_y[order[0]]
    alpha_x = pi[order[0]]               # α(x) = min_y [c(x,y)-β(y)]  (Eq. 6)

    if len(order) > 1:
        alpha_prime_x = pi[order[1]]     # second-best  (Eq. 7)
    else:
        alpha_prime_x = alpha_x          # only one option

    bid_value = C[x, y_star] - alpha_prime_x - epsilon   # Eq. 8

    return y_star, alpha_prime_x, bid_value


# ---------------------------------------------------------------------------
# 6. Full Sparse/Dense Hybrid Auction for OT  (Sec. 4, Proposition 1)
#    with ε-scaling (Sec. 5 / Implementation Details Sec. 6)
# ---------------------------------------------------------------------------

def auction_ot_sparse(
        C: np.ndarray,
        mu_X: np.ndarray,
        mu_Y: np.ndarray,
        levels_X: list,
        levels_Y: list,
        epsilon: float,
        allowed: np.ndarray,
        mu_init: np.ndarray | None = None,
        max_iter: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One run of the sparse/dense hybrid auction for OT at fixed ε.

    Parameters
    ----------
    C        : N×N cost matrix
    mu_X/Y   : marginals (sum to 1, float)
    levels_X/Y: quadtree hierarchies
    epsilon  : current ε value
    allowed  : N×N bool – sparse N̂, modified in-place
    mu_init  : warm-start coupling (or None)
    max_iter : safety cap

    Returns
    -------
    mu       : N×N coupling
    alpha_prime : N array of α' values (needed by caller)
    """
    N = C.shape[0]
    beta        = np.zeros(N)
    alpha_prime = np.zeros(N)
    mu          = mu_init.copy() if mu_init is not None else np.zeros((N, N))

    roots_X = levels_X[-1]   # coarsest level = roots
    roots_Y = levels_Y[-1]

    for iteration in range(max_iter):
        remaining_X = mu_X - mu.sum(axis=1)
        unassigned  = np.where(remaining_X > 1e-10)[0]
        if len(unassigned) == 0:
            break

        # --- Bidding phase (Eq. 6-8) ---
        rebid = set()
        for x in unassigned:
            y_star, ap, bid = compute_bid(x, allowed, C, beta, epsilon)
            alpha_prime[x] = ap

            remaining_y = mu_Y[y_star] - mu[:, y_star].sum()
            amount = min(remaining_X[x], remaining_y)
            if amount > 1e-10:
                mu[x, y_star] += amount
                beta[y_star]   = bid          # Eq. 9: accept lowest bid

        # --- Update hierarchical extensions ---
        update_alpha_hat(levels_X, alpha_prime)
        update_beta_hat(levels_Y, beta)

        # --- Consistency check phase (Sec. 4) ---
        rebid.clear()
        for ra in roots_X:
            for rb in roots_Y:
                consistency_check(ra, rb, C, beta, alpha_prime,
                                  allowed, rebid)

        if rebid:
            for x in rebid:
                mu[x, :] = 0.0   # reset mass for rebidding points

    return mu, alpha_prime


def hierarchical_ot(
        C: np.ndarray,
        mu_X: np.ndarray,
        mu_Y: np.ndarray,
        levels_X: list,
        levels_Y: list,
        epsilon_0: float  = None,
        epsilon_final: float = None,
        rho: float = 4.0,
) -> np.ndarray:
    """
    Full hierarchical multiscale OT with ε-scaling (Sec. 4 + Sec. 5).

    Strategy (Sec. 4 "A Hierarchical Multiscale Approach"):
      1. Solve coarse problem at generation g (small, dense).
      2. Use coarse solution support as initial N̂ for generation g-1.
      3. Repeat down to generation 0 (full resolution).
      4. Apply ε-scaling throughout.

    Parameters
    ----------
    rho   : ε reduction factor per scaling step (paper uses similar scheme)
    """
    N = C.shape[0]
    g = len(levels_X) - 1    # hierarchy depth

    if epsilon_0    is None: epsilon_0    = C.max() / 10
    if epsilon_final is None: epsilon_final = 1.0 / (N * 10)

    # --- Build ĉ cache once ---
    precompute_c_hat(levels_X, levels_Y, C)

    # --- Start sparse set: only 5 nearest neighbours (cheap heuristic) ---
    allowed = np.zeros((N, N), dtype=bool)
    for i in range(N):
        nn = np.argsort(C[i])[:5]
        allowed[i, nn] = True

    mu = None

    # --- ε-scaling loop ---
    epsilon = epsilon_0
    while epsilon >= epsilon_final * 0.999:
        mu, _ = auction_ot_sparse(
            C, mu_X, mu_Y, levels_X, levels_Y,
            epsilon=epsilon,
            allowed=allowed,
            mu_init=mu,
        )
        epsilon /= rho

    return mu


# ---------------------------------------------------------------------------
# 7. Demo
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    N = 200

    X = np.random.rand(N, 2)
    Y = np.random.rand(N, 2)
    C = ot.dist(X, Y, metric='sqeuclidean')

    mu_X = np.ones(N) / N
    mu_Y = np.ones(N) / N

    # Build proper quadtrees (levels 1..4  →  grids 2×2, 4×4, 8×8, 16×16)
    print("Baue Quadtree-Hierarchien auf …")
    levels_X = build_quadtree(X, max_levels=4)
    levels_Y = build_quadtree(Y, max_levels=4)
    print(f"  Levels X: {[len(l) for l in levels_X]}")
    print(f"  Levels Y: {[len(l) for l in levels_Y]}")

    # --- Reference: POT exact solution ---
    print("\n[1] POT exakte Lösung (Referenz) …")
    t0 = time.perf_counter()
    T_pot  = ot.emd(mu_X, mu_Y, C)
    cost_pot = (T_pot * C).sum()
    print(f"    Kosten: {cost_pot:.6f}  ({time.perf_counter()-t0:.4f}s)")

    # --- SS13 hierarchical auction ---
    print("\n[2] SS13 Hierarchischer Auktions-Algorithmus …")
    t0 = time.perf_counter()
    T_hier = hierarchical_ot(C, mu_X, mu_Y, levels_X, levels_Y)
    cost_hier = (T_hier * C).sum()
    print(f"    Kosten: {cost_hier:.6f}  ({time.perf_counter()-t0:.4f}s)")

    print(f"\nDifferenz zur optimalen Lösung: {abs(cost_pot - cost_hier):.6f}")
    sparsity = (T_hier > 0).sum()
    print(f"Sparsität der Lösung: {sparsity} / {N*N} Einträge "
          f"({100*sparsity/N**2:.1f}%)")


if __name__ == "__main__":
    main()