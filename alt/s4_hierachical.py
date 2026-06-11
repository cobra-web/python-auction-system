import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class QuadtreeNode:
    def __init__(self, indices, bounds, depth=0):
        self.indices  = np.array(indices)
        self.bounds   = bounds
        self.depth    = depth
        self.children = []

    @property
    def is_leaf(self):
        return len(self.children) == 0

def build_quadtree(points, max_depth=5):
    pad = 1e-9
    lo, hi = points.min(0) - pad, points.max(0) + pad
    root   = QuadtreeNode(np.arange(len(points)), (lo[0], hi[0], lo[1], hi[1]))
    levels = [[root]]

    def split(node):
        if node.depth >= max_depth or len(node.indices) <= 1:
            return
        x_lo, x_hi, y_lo, y_hi = node.bounds
        xm, ym = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
        for bounds in [(x_lo,xm,y_lo,ym),(xm,x_hi,y_lo,ym),(x_lo,xm,ym,y_hi),(xm,x_hi,ym,y_hi)]:
            pts  = points[node.indices]
            mask = ((pts[:,0]>=bounds[0])&(pts[:,0]<bounds[1])&
                    (pts[:,1]>=bounds[2])&(pts[:,1]<bounds[3]))
            idx  = node.indices[mask]
            if len(idx) == 0: continue
            child = QuadtreeNode(idx, bounds, node.depth + 1)
            node.children.append(child)
            while len(levels) <= child.depth: levels.append([])
            levels[child.depth].append(child)
            split(child)

    split(root)
    return root, levels

def c_hat(na, nb, px, py):
    xs = px[na.indices]; ys = py[nb.indices]
    return float(((xs[:,None]-ys[None,:])**2).sum(2).min())

def consistency_check(N_hat, root_x, root_y, px, py, alpha_prime, beta, cost):
    violations = set()

    def check(na, nb):
        c_hat_ab = c_hat(na, nb, px, py)
        # beta_hat(b): hierarchische Erweiterung, hier min weil Kostenminimierung
        beta_hat_b = float(beta[nb.indices].max()) 
        alpha_hat_a = float(alpha_prime[na.indices].max())

        if c_hat_ab - beta_hat_b >= alpha_hat_a:
            return

        if na.is_leaf and nb.is_leaf:
            x, y = int(na.indices[0]), int(nb.indices[0])
            if (x, y) not in N_hat:
                violations.add((x, y))
            return

        for ca in (na.children or [na]):
            for cb in (nb.children or [nb]):
                check(ca, cb)

    check(root_x, root_y)
    return violations

def hybrid_auction_phase(cost, N_hat, mu_x, mu_y):
    N_x, N_y = cost.shape
    beta = np.zeros(N_y)
    alpha_prime = np.zeros(N_x)
    coupling = np.zeros_like(cost)
    
    for x in range(N_x):
        valid_ys = [y for (_x, y) in N_hat if _x == x]
        if len(valid_ys) >= 2:
            costs_x = [cost[x, y] - beta[y] for y in valid_ys]
            costs_x.sort()
            alpha_prime[x] = costs_x[1] # Zweitbeste Option (Eq 7)
        else:
            alpha_prime[x] = 0

    return coupling, alpha_prime, beta

def solve_ot_multiscale_hybrid(px, py, mu_x, mu_y, max_depth=4, verbose=True):

    N_x, N_y = len(px), len(py)
    cost = np.sum((px[:,None]-py[None,:])**2, axis=2)

    root_x, levels_x = build_quadtree(px, max_depth)
    root_y, levels_y = build_quadtree(py, max_depth)

    N_hat = set((i, i) for i in range(N_x)) # Stark vereinfachte Init

    if verbose: print(f"Initial N_hat: {len(N_hat)} pairs")

    for i in range(10):
        coupling, alpha_prime, beta = hybrid_auction_phase(cost, N_hat, mu_x, mu_y)

        new_pairs = consistency_check(N_hat, root_x, root_y, px, py, alpha_prime, beta, cost)
        
        if verbose:
            print(f"  Round {i+1}: {len(N_hat)} pairs, added={len(new_pairs)}")
            
        if not new_pairs:
            break # Globales Optimum auf aktueller Menge garantiert
            
        N_hat |= new_pairs

    cost_masked = np.full_like(cost, 1e9)
    for (x, y) in N_hat: cost_masked[x, y] = cost[x, y]
    import ot
    final_coupling = ot.emd(mu_x, mu_y, cost_masked)
    total_cost = float(np.sum(cost * final_coupling))

    return final_coupling, total_cost, N_hat

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N   = 40
    px  = rng.random((N, 2))
    py  = rng.random((N, 2))
    mu_x = np.ones(N) / N
    mu_y = np.ones(N) / N

    coup_hier, cost_hier, N_hat = solve_ot_multiscale_hybrid(px, py, mu_x, mu_y, verbose=True)
    print(f"Hierarchical cost: {cost_hier:.4f}  pairs={len(N_hat)} ({100*len(N_hat)/N**2:.1f}%)")