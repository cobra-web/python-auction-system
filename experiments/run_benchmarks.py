import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# Adjust these imports to match your project's structure
from src.utils.eps_scaling import EpsScalingManager
from src.core.ot_auction import AuctionOT
from src.hierarchical.partitions import HierarchicalPartition
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver

def build_matched_trees(X_pts, Y_pts, max_points_per_cell=1, max_allowed_depth=15):
    """
    Builds spatial trees for X and Y, forcing them to the same maximum depth
    to ensure the hierarchical solver levels align perfectly.
    """
    probe_X = HierarchicalPartition(X_pts, max_points_per_cell=max_points_per_cell, max_allowed_depth=max_allowed_depth)
    probe_Y = HierarchicalPartition(Y_pts, max_points_per_cell=max_points_per_cell, max_allowed_depth=max_allowed_depth)

    target_depth = max(probe_X.max_depth, probe_Y.max_depth)
    
    tree_X = HierarchicalPartition(X_pts, max_points_per_cell=max_points_per_cell, max_allowed_depth=target_depth)
    tree_Y = HierarchicalPartition(Y_pts, max_points_per_cell=max_points_per_cell, max_allowed_depth=target_depth)
    return tree_X, tree_Y

def run_benchmarks():
    print("=" * 115)
    print("BACHELOR THESIS: OPTIMAL TRANSPORT SOLVER BENCHMARK")
    print("=" * 115)
    print(f"{'N':<6} | {'LAP Time':<10} | {'LAP Cost':<10} | {'Dense Time':<12} | {'Hier Time':<12} | {'Dense Cost':<12} | {'Hier Cost':<12} | {'Pairs (D/H)':<13} | {'Match?':<6}")
    print("-" * 115)

    scales = [16, 32, 64, 128]
    np.random.seed(42) # Consistent seed for reproducible benchmark runs

    for N in scales:
        # ---------------------------------------------------------
        # 1. Generate Problem Data
        # ---------------------------------------------------------
        X_pts = np.random.rand(N, 2)
        Y_pts = np.random.rand(N, 2)
        
        mu_X = np.random.randint(1, 6, size=N)
        mu_Y = np.random.randint(1, 6, size=N)
        
        # Balance total mass precisely
        diff = np.sum(mu_X) - np.sum(mu_Y)
        if diff > 0:
            mu_Y[0] += diff
        elif diff < 0:
            mu_X[0] += abs(diff)

        # Baseline exact distance matrix for cost verification
        C_exact = cdist(X_pts, Y_pts)

        # ---------------------------------------------------------
        # 2. LAP Reference (SciPy)
        # ---------------------------------------------------------
        # Expand points based on mass to formulate a classic assignment problem
        expanded_X = np.repeat(X_pts, mu_X, axis=0)
        expanded_Y = np.repeat(Y_pts, mu_Y, axis=0)
        C_expanded = cdist(expanded_X, expanded_Y)
        
        t0 = time.time()
        row_ind, col_ind = linear_sum_assignment(C_expanded)
        lap_time = time.time() - t0
        lap_cost = C_expanded[row_ind, col_ind].sum()

        # ---------------------------------------------------------
        # 3. Dense Solver (Standard Epsilon-Scaling Auction)
        # ---------------------------------------------------------
        t0 = time.time()
        dense_manager = EpsScalingManager(AuctionOT, X_pts=X_pts, Y_pts=Y_pts, mu_X=mu_X, mu_Y=mu_Y)
        dense_mu_dict, _, _, _ = dense_manager.solve()
        dense_runtime = time.time() - t0

        dense_mu = np.zeros((N, N))
        dense_cost = 0.0
        dense_active_pairs = 0
        
        for x, targets in dense_mu_dict.items():
            for y, mass in targets.items():
                if mass > 1e-5:
                    dense_mu[x, y] = mass
                    dense_cost += mass * C_exact[x, y]
                    dense_active_pairs += 1

        # ---------------------------------------------------------
        # 4. Hierarchical Solver (Multiscale Shielding)
        # ---------------------------------------------------------
        t0 = time.time()
        # Force 1-point capacity to ensure identical fine-level resolution
        tree_X, tree_Y = build_matched_trees(X_pts, Y_pts, max_points_per_cell=1)
        multiscale_solver = HierarchicalMultiscaleSolver(tree_X, tree_Y, mu_X, mu_Y)
        sparse_hier_mu = multiscale_solver.solve()
        hierarchical_runtime = time.time() - t0

        hier_mu = np.zeros((N, N))
        hier_cost = 0.0
        hier_active_pairs = 0
        
        for x, y, mass in sparse_hier_mu:
            if mass > 1e-5:
                hier_mu[x, y] += mass
                hier_cost += mass * C_exact[x, y]
                hier_active_pairs += 1

        # ---------------------------------------------------------
        # 5. Structure Check & Table Output
        # ---------------------------------------------------------
        matrices_identical = np.allclose(dense_mu, hier_mu, atol=1e-3)

        pairs_str = f"{dense_active_pairs}/{hier_active_pairs}"
        match_str = "Yes" if matrices_identical else "No"
        
        print(f"{N:<6} | "
              f"{lap_time:<10.5f} | "
              f"{lap_cost:<10.4f} | "
              f"{dense_runtime:<12.5f} | "
              f"{hierarchical_runtime:<12.5f} | "
              f"{dense_cost:<12.4f} | "
              f"{hier_cost:<12.4f} | "
              f"{pairs_str:<13} | "
              f"{match_str:<6}")

if __name__ == "__main__":
    run_benchmarks()
