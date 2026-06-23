import time
import numpy as np
from src.utils.cost_functions import squared_euclidean
from src.hierarchical.partitions import HierarchicalPartition
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver
from src.utils.eps_scaling import EpsScalingManager
from src.core.ot_auction import AuctionOT
from src.utils.sinkhorn import log_sinkhorn

def run_comprehensive_benchmarks(N_sizes=[16, 32, 64, 128]):
    print("BACHELOR THESIS: OPTIMAL TRANSPORT SOLVER BENCHMARK")
    print(f"{'Scale (N)':<10} | {'Solver Type':<15} | {'Computed Cost':<14} | {'Runtime (s)':<12}")
    print("-" * 66)
    
    for N in N_sizes:
        np.random.seed(42) # Strict seed consistency across competitors
        
        # 1. Generate 2D spatial point distributions
        X_pts = np.random.rand(N, 2)
        Y_pts = np.random.rand(N, 2)
        C = squared_euclidean(X_pts, Y_pts)
        
        # 2. Generate balanced integer mass distributions
        mu_X = np.random.randint(1, 5, size=N)
        mu_Y = np.random.randint(1, 5, size=N)
        diff = np.sum(mu_X) - np.sum(mu_Y)
        if diff > 0:
            mu_Y[0] += diff
        elif diff < 0:
            mu_X[0] += abs(diff)
            
        # ----------------------------------------------------
        # RUN COMPETITOR 1: Log-Stabilized Sinkhorn
        # ----------------------------------------------------
        try:
            t0 = time.perf_counter()
            _, sink_cost, _ = log_sinkhorn(C, mu_X, mu_Y, gamma=0.01, max_iters=2000)
            t_sink = time.perf_counter() - t0
            print(f"{N:<10} | {'Log-Sinkhorn':<15} | {sink_cost:<14.4f} | {t_sink:<12.5f}")
        except Exception as e:
            print(f"{N:<10} | {'Log-Sinkhorn':<15} | {'FAILED':<14} | {'-':<12}")

        # ----------------------------------------------------
        # RUN COMPETITOR 2: Flat Dense Auction
        # ----------------------------------------------------
        try:
            t1 = time.perf_counter()
            dense_manager = EpsScalingManager(AuctionOT, C, mu_X=mu_X, mu_Y=mu_Y)
            _, dense_cost, _, _ = dense_manager.solve()
            t_dense = time.perf_counter() - t1
            print(f"{N:<10} | {'Dense Auction':<15} | {dense_cost:<14.4f} | {t_dense:<12.5f}")
        except Exception as e:
            print(f"{N:<10} | {'Dense Auction':<15} | {'FAILED':<14} | {'-':<12}")

        # ----------------------------------------------------
        # RUN COMPETITOR 3: Hierarchical Multiscale Auction
        # ----------------------------------------------------
        try:
            # Build trees and pad generation depths to match layouts
            tree_X = HierarchicalPartition(X_pts)
            tree_Y = HierarchicalPartition(Y_pts)
            max_g = max(tree_X.g, tree_Y.g)
            while tree_X.g < max_g:
                tree_X.generations.append(tree_X.generations[-1])
                tree_X.g += 1
            while tree_Y.g < max_g:
                tree_Y.generations.append(tree_Y.generations[-1])
                tree_Y.g += 1
                
            t2 = time.perf_counter()
            multiscale_solver = HierarchicalMultiscaleSolver(tree_X, tree_Y, C, mu_X, mu_Y)
            hier_mu = multiscale_solver.solve()
            hier_cost = np.sum(hier_mu * C)
            t_hier = time.perf_counter() - t2
            print(f"{N:<10} | {'Hierarchical':<15} | {hier_cost:<14.4f} | {t_hier:<12.5f}")
        except Exception as e:
            print(f"{N:<10} | {'Hierarchical':<15} | {'FAILED':<14} | {'-':<12}")
            
        print("-" * 66)

if __name__ == "__main__":
    run_comprehensive_benchmarks(N_sizes=[16, 32, 64])
