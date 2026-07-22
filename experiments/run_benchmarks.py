import time
import os
import sys
import ot
import numpy as np
from src.utils.cost_functions import squared_euclidean
from src.utils.eps_scaling import EpsScalingManager
from src.core.ot_auction import AuctionOT
from src.core.lap_auction import AuctionLAP

from src.hierarchical.partitions import HierarchicalPartition
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver

class SilencePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def build_matched_trees(X_pts, Y_pts, max_points_per_cell=1, max_allowed_depth=15):
    probe_X = HierarchicalPartition(X_pts, max_points_per_cell=max_points_per_cell, max_allowed_depth=max_allowed_depth)
    probe_Y = HierarchicalPartition(Y_pts, max_points_per_cell=max_points_per_cell, max_allowed_depth=max_allowed_depth)

    target_depth = max(probe_X.max_depth, probe_Y.max_depth)
    
    tree_X = HierarchicalPartition(X_pts, max_points_per_cell=max_points_per_cell, max_allowed_depth=target_depth)
    tree_Y = HierarchicalPartition(Y_pts, max_points_per_cell=max_points_per_cell, max_allowed_depth=target_depth)
    return tree_X, tree_Y

def run_comprehensive_benchmarks():
    print("\n=============================================================================================================")
    print("BACHELOR THESIS: OPTIMAL TRANSPORT SOLVER BENCHMARK (WITH PROFESSOR DIAGNOSTICS)")
    print("=============================================================================================================\n")
    
    # Table Header (Replaced Primal Match with Gap %)
    header = f"| {'N':<4} | {'Method':<15} | {'Time (s)':<10} | {'Rep. Cost':<10} | {'Man. Cost':<10} | {'Mass (Soll)':<15} | {'Pairs (Max)':<12} | {'Gap (%)':<12} |"
    separator = "-" * len(header)
    
    print(separator)
    print(header)
    print(separator)
    
    scales = [16, 32, 64, 128, 1024]
    
    for N in scales:
        np.random.seed(50)
        X_pts = np.random.rand(N, 2)
        Y_pts = np.random.rand(N, 2)
        
        # ot.emd2 requires float types for masses
        mu_X = np.random.randint(1, 6, size=N).astype(float)
        mu_Y = np.random.randint(1, 6, size=N).astype(float)
        
        diff = np.sum(mu_X) - np.sum(mu_Y)
        if diff > 0:
            mu_Y[0] += diff
        elif diff < 0:
            mu_X[0] += abs(diff)
            
        total_problem_mass = np.sum(mu_X)
        
        C = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                C[i, j] = np.sum((X_pts[i] - Y_pts[j])**2)
                
        # --- NEW: EXACT COST CALCULATION ---
        try:
            exact_cost = ot.emd2(mu_X, mu_Y, C)
        except Exception as e:
            exact_cost = float('inf')  # Fallback if POT fails
            print(f"Warning: Exact cost failed to compute for N={N}")

        try:
            t0 = time.perf_counter()
            with SilencePrints():
                lap_solver = AuctionLAP(C)
                lap_assignment, lap_cost, lap_iters = lap_solver.solve()
            t_lap = time.perf_counter() - t0
            
            lap_mu = np.zeros_like(C)
            for x, y in enumerate(lap_assignment):
                if y != -1: lap_mu[x, y] = 1.0
                
            manual_lap_cost = np.sum(lap_mu * C)
            print(f"| {N:<4} | {'LAP Reference':<15} | {t_lap:<10.5f} | {'-':<10} | {manual_lap_cost:<10.4f} | {'-':<15} | {'-':<12} | {'-':<12} |")
        except Exception as e:
            print(f"| {N:<4} | {'LAP Reference':<15} | {'FAIL':<10} | {'-':<10} | {'-':<10} | {'-':<15} | {'-':<12} | {'-':<12} |")
            
        dense_mu = None
        try:
            t1 = time.perf_counter()
            with SilencePrints():
                dense_manager = EpsScalingManager(AuctionOT, X_pts=X_pts, Y_pts=Y_pts, mu_X=mu_X, mu_Y=mu_Y)
                dense_mu_dict, dense_reported_cost, _, _ = dense_manager.solve()
            t_dense = time.perf_counter() - t1
            
            dense_mu = np.zeros((N, N))
            for x in dense_mu_dict:
                for y, m in dense_mu_dict[x].items():
                    dense_mu[x, y] = m

            manual_dense_cost = np.sum(dense_mu * C)
            actual_dense_mass = np.sum(dense_mu)
            active_dense_pairs = np.sum(dense_mu > 1e-5)
            
            # --- NEW: DENSE GAP AND MARGINAL ASSERTIONS ---
            dense_gap = ((manual_dense_cost - exact_cost) / exact_cost) * 100
            assert np.allclose(dense_mu.sum(axis=1), mu_X, atol=1e-5), f"Dense: Source marginal violated at N={N}!"
            assert np.allclose(dense_mu.sum(axis=0), mu_Y, atol=1e-5), f"Dense: Target marginal violated at N={N}!"
            
            print(f"| {N:<4} | {'DENSE OT':<15} | {t_dense:<10.5f} | {dense_reported_cost:<10.4f} | {manual_dense_cost:<10.4f} | {f'{actual_dense_mass:.2f} ({total_problem_mass:.0f})':<15} | {f'{active_dense_pairs} ({2*N-1})':<12} | {f'{dense_gap:>8.3f}%':<12} |")
        except Exception as e:
            print(f"| {N:<4} | {'DENSE OT':<15} | {'FAIL':<10} | {'-':<10} | {'-':<10} | {'-':<15} | {'-':<12} | {'-':<12} |")
            import traceback; traceback.print_exc()

        try:
            tree_X, tree_Y = build_matched_trees(X_pts, Y_pts)

            t2 = time.perf_counter()
            with SilencePrints():
                multiscale_solver = HierarchicalMultiscaleSolver(tree_X, tree_Y, mu_X, mu_Y)
                sparse_hier_mu = multiscale_solver.solve()
            t_hier = time.perf_counter() - t2
            
            hier_mu = np.zeros((N, N))
            for x, y, mass in sparse_hier_mu:
                hier_mu[x, y] += mass

            manual_hier_cost = np.sum(hier_mu * C)
            actual_hier_mass = np.sum(hier_mu)
            active_hier_pairs = np.sum(hier_mu > 1e-5)
            
            # --- NEW: HIERARCHICAL GAP AND MARGINAL ASSERTIONS ---
            hier_gap = ((manual_hier_cost - exact_cost) / exact_cost) * 100
            assert np.allclose(hier_mu.sum(axis=1), mu_X, atol=1e-5), f"Hierarchical: Source marginal violated at N={N}!"
            assert np.allclose(hier_mu.sum(axis=0), mu_Y, atol=1e-5), f"Hierarchical: Target marginal violated at N={N}!"
            
            print(f"| {N:<4} | {'HIERARCH. OT':<15} | {t_hier:<10.5f} | {'-':<10} | {manual_hier_cost:<10.4f} | {f'{actual_hier_mass:.2f} ({total_problem_mass:.0f})':<15} | {f'{active_hier_pairs} ({2*N-1})':<12} | {f'{hier_gap:>8.3f}%':<12} |")
            
        except Exception as e:
            print(f"| {N:<4} | {'HIERARCH. OT':<15} | {'FAIL':<10} | {'-':<10} | {'-':<10} | {'-':<15} | {'-':<12} | {'-':<12} |")
            import traceback; traceback.print_exc()
            
        print(separator)

if __name__ == "__main__":
    run_comprehensive_benchmarks()
