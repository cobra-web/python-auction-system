import time
import os
import sys
import ot
import numpy as np
import matplotlib.pyplot as plt

from src.utils.eps_scaling import EpsScalingManager
from src.core.ot_auction import AuctionOT
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
    print("BACHELOR THESIS: RIGOROUS BENCHMARK (UNIFIED UNITS & DECISIVE SCALING)")
    print("=============================================================================================================\n")
    
    scales = [16, 32, 64, 128, 256, 512, 1024]
    
    results = {
        'N': [],
        'dense_time_mean': [], 'dense_time_std': [],
        'hier_time_mean': [], 'hier_time_std': [],
        'dense_gap_mean': [], 'hier_gap_mean': []
    }

    print(f"| {'N':<4} | {'Method':<15} | {'Time Mean (s)':<15} | {'Time Std (s)':<15} | {'Gap Mean (%)':<12} |")
    print("-" * 73)

    for N in scales:
        # 5 seeds for small N, 3 seeds for large N to optimize runtime
        num_seeds = 5 if N <= 128 else 3
        seeds = [42, 50, 100, 2024, 999][:num_seeds]

        dense_times, hier_times = [], []
        dense_gaps, hier_gaps = [], []
        
        tight_target = 0.5 / (N + 1)
        tight_min = 1e-5

        for seed in seeds:
            np.random.seed(seed)
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
                    
            try:
                exact_cost = ot.emd2(mu_X, mu_Y, C)
            except Exception:
                continue

            tree_X, tree_Y = build_matched_trees(X_pts, Y_pts)

            gmin = np.minimum(X_pts.min(axis=0), Y_pts.min(axis=0))
            gmax = np.maximum(X_pts.max(axis=0), Y_pts.max(axis=0))
            GLOBAL_MAX_C = float(np.sum((gmax - gmin) ** 2)) or 1.0

            # ---------------------------------------------------------
            # DENSE OT
            # ---------------------------------------------------------
            t1 = time.perf_counter()
            with SilencePrints():
                dense_manager = EpsScalingManager(
                    AuctionOT, X_pts=X_pts, Y_pts=Y_pts, mu_X=mu_X, mu_Y=mu_Y,
                    normalize=False, max_c=GLOBAL_MAX_C, target_eps=tight_target, min_eps=tight_min
                )
                dense_mu_dict, _, _, _ = dense_manager.solve()
            dense_times.append(time.perf_counter() - t1)
            
            dense_mu = np.zeros((N, N))
            for x in dense_mu_dict:
                for y, m in dense_mu_dict[x].items():
                    dense_mu[x, y] = m
            dense_gaps.append(((np.sum(dense_mu * C) - exact_cost) / exact_cost) * 100)

            # ---------------------------------------------------------
            # HIERARCHICAL OT
            # ---------------------------------------------------------
            t2 = time.perf_counter()
            with SilencePrints():
                multiscale_solver = HierarchicalMultiscaleSolver(
                    tree_X, tree_Y, mu_X, mu_Y, 
                    max_c=GLOBAL_MAX_C, target_eps=tight_target, min_eps=tight_min
                )
                sparse_hier_mu = multiscale_solver.solve()
            hier_times.append(time.perf_counter() - t2)
            
            hier_cost = sum(mass * C[x, y] for x, y, mass in sparse_hier_mu)
            hier_gaps.append(((hier_cost - exact_cost) / exact_cost) * 100)

        # Aggregate Statistics
        results['N'].append(N)
        results['dense_time_mean'].append(np.mean(dense_times))
        results['dense_time_std'].append(np.std(dense_times))
        results['hier_time_mean'].append(np.mean(hier_times))
        results['hier_time_std'].append(np.std(hier_times))
        results['dense_gap_mean'].append(np.mean(dense_gaps))
        results['hier_gap_mean'].append(np.mean(hier_gaps))

        print(f"| {N:<4} | {'DENSE OT':<15} | {np.mean(dense_times):<15.4f} | {np.std(dense_times):<15.4f} | {np.mean(dense_gaps):>8.3f}%   |")
        print(f"| {N:<4} | {'HIERARCH. OT':<15} | {np.mean(hier_times):<15.4f} | {np.std(hier_times):<15.4f} | {np.mean(hier_gaps):>8.3f}%   |")
        print("-" * 73)
        
        # Save intermediate progress so plots can still be drawn if interrupted
        plot_results(results)

    return results

def plot_results(results):
    if len(results['N']) == 0:
        return
        
    N_arr = np.array(results['N'])
    
    # 1. Log-Log Scaling Plot
    plt.figure(figsize=(8, 6))
    plt.loglog(N_arr, results['dense_time_mean'], marker='o', label='Dense Auction', linewidth=2)
    plt.loglog(N_arr, results['hier_time_mean'], marker='s', label='Hierarchical Auction', linewidth=2)
    
    plt.fill_between(N_arr, 
                     np.array(results['dense_time_mean']) - np.array(results['dense_time_std']),
                     np.array(results['dense_time_mean']) + np.array(results['dense_time_std']), 
                     alpha=0.2)
    plt.fill_between(N_arr, 
                     np.array(results['hier_time_mean']) - np.array(results['hier_time_std']),
                     np.array(results['hier_time_mean']) + np.array(results['hier_time_std']), 
                     alpha=0.2)
    
    plt.title('Computation Time vs Problem Size (Log-Log)', fontsize=14)
    plt.xlabel('Number of Points (N)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('thesis_scaling_plot.pdf')
    plt.close()
    
    # 2. Gap Convergence Plot
    plt.figure(figsize=(8, 6))
    plt.semilogx(N_arr, results['dense_gap_mean'], marker='o', label='Dense Gap %', linewidth=2)
    plt.semilogx(N_arr, results['hier_gap_mean'], marker='s', label='Hierarchical Gap %', linewidth=2)
    
    plt.title('Optimality Gap vs Exact POT Reference', fontsize=14)
    plt.xlabel('Number of Points (N)', fontsize=12)
    plt.ylabel('Relative Gap (%)', fontsize=12)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('thesis_gap_plot.pdf')
    plt.close()

if __name__ == "__main__":
    final_results = run_comprehensive_benchmarks()
