import numpy as np
from src.core.ot_auction import AuctionOT
from src.utils.eps_scaling import EpsScalingManager
from src.hierarchical.partitions import HierarchicalPartition
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver
import ot

def test_neighborhood_sufficiency():
    print("\n--- DECISIVE NEIGHBORHOOD SUFFICIENCY TEST ---")
    N = 128  # <-- Run N=128 for rapid verification
    np.random.seed(42)
    
    X_pts = np.random.rand(N, 2)
    Y_pts = np.random.rand(N, 2)
    mu_X = np.random.randint(1, 6, size=N).astype(float)
    mu_Y = np.random.randint(1, 6, size=N).astype(float)
    
    diff = np.sum(mu_X) - np.sum(mu_Y)
    if diff > 0: mu_Y[0] += diff
    elif diff < 0: mu_X[0] += abs(diff)
        
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = np.sum((X_pts[i] - Y_pts[j])**2)
            
    exact_cost = ot.emd2(mu_X, mu_Y, C)
    
    gmin = np.minimum(X_pts.min(axis=0), Y_pts.min(axis=0))
    gmax = np.maximum(X_pts.max(axis=0), Y_pts.max(axis=0))
    GLOBAL_MAX_C = float(np.sum((gmax - gmin) ** 2)) or 1.0

    tight_eps = 1e-4

    # 1. Cold-start Dense OT Reference at Tight Epsilon
    dense_mgr = EpsScalingManager(
        AuctionOT, X_pts=X_pts, Y_pts=Y_pts, mu_X=mu_X, mu_Y=mu_Y,
        normalize=False, max_c=GLOBAL_MAX_C, target_eps=tight_eps, min_eps=1e-6
    )
    dense_mu_dict, dense_cost, _, _ = dense_mgr.solve()
    dense_mu = np.zeros((N, N))
    for x in dense_mu_dict:
        for y, m in dense_mu_dict[x].items():
            dense_mu[x, y] = m
    dense_gap = ((np.sum(dense_mu * C) - exact_cost) / exact_cost) * 100

    # 2. Extract Neighborhood from Hierarchical Solver
    tree_X = HierarchicalPartition(X_pts, max_points_per_cell=1, max_allowed_depth=15)
    tree_Y = HierarchicalPartition(Y_pts, max_points_per_cell=1, max_allowed_depth=15)
    
    hier_solver = HierarchicalMultiscaleSolver(
        tree_X, tree_Y, mu_X, mu_Y, 
        max_c=GLOBAL_MAX_C, target_eps=tight_eps, min_eps=1e-6
    )
    _ = hier_solver.solve()
    final_edges = hier_solver.last_N_guess

    # 3. Solve Restricted Single-Level OT on extracted neighborhood
    sparse_mgr = EpsScalingManager(
        AuctionOT, X_pts=X_pts, Y_pts=Y_pts, mu_X=mu_X, mu_Y=mu_Y,
        allowed_edges=final_edges, normalize=False, max_c=GLOBAL_MAX_C, 
        target_eps=tight_eps, min_eps=1e-6
    )
    sparse_mu_dict, _, _, _ = sparse_mgr.solve()
    sparse_mu = np.zeros((N, N))
    for x in sparse_mu_dict:
        for y, m in sparse_mu_dict[x].items():
            sparse_mu[x, y] = m
    sparse_gap = ((np.sum(sparse_mu * C) - exact_cost) / exact_cost) * 100

    print(f"\n[Dense OT Gap @ eps={tight_eps}]:      {dense_gap:.4f}%")
    print(f"[Restricted Sparse Gap @ eps={tight_eps}]: {sparse_gap:.4f}%")
    print(f"Gap Difference:                       {abs(dense_gap - sparse_gap):.4f}%")

if __name__ == "__main__":
    test_neighborhood_sufficiency()
