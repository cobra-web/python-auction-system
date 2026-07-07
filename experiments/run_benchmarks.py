import time
import os
import numpy as np
from src.utils.cost_functions import squared_euclidean
from src.utils.eps_scaling import EpsScalingManager
from src.core.ot_auction import AuctionOT
from src.core.lap_auction import AuctionLAP

from src.hierarchical.partitions import HierarchicalPartition
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver

class SilencePrints:
    def __enter__(self):
        import sys
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys
        sys.stdout.close()
        sys.stdout = self._original_stdout


def build_matched_trees(X_pts, Y_pts, max_points_per_cell=1, max_allowed_depth=10):
    # First pass: find natural depth of each point set independently.
    probe_X = HierarchicalPartition(X_pts, max_points_per_cell=max_points_per_cell,
                                     max_allowed_depth=max_allowed_depth)
    probe_Y = HierarchicalPartition(Y_pts, max_points_per_cell=max_points_per_cell,
                                     max_allowed_depth=max_allowed_depth)

    target_g = max(probe_X.g, probe_Y.g)

    # Second pass: rebuild both to the same target_g so generations[k] means
    # the same tree depth on both sides. This replaces the old post-hoc
    # patch that duplicated the coarsest generation and desynced
    # generations[gen_coarse - 1] from the real children of generations[gen_coarse].
    tree_X = HierarchicalPartition(X_pts, target_g=target_g,
                                    max_points_per_cell=max_points_per_cell,
                                    max_allowed_depth=max_allowed_depth)
    tree_Y = HierarchicalPartition(Y_pts, target_g=target_g,
                                    max_points_per_cell=max_points_per_cell,
                                    max_allowed_depth=max_allowed_depth)

    assert tree_X.g == tree_Y.g == target_g
    return tree_X, tree_Y


def run_comprehensive_benchmarks(N_sizes=[16, 32, 64, 128]):
    print("\nBACHELOR THESIS: OPTIMAL TRANSPORT SOLVER BENCHMARK")
    print(f"{'Scale (N)':<10} | {'Solver Type':<15} | {'Active Pairs':<12} | {'Computed Cost':<14} | {'Runtime (s)':<12}")
    print("-" * 80)

    for N in N_sizes:
        np.random.seed(50)

        X_pts = np.random.rand(N, 2)
        Y_pts = np.random.rand(N, 2)
        C = squared_euclidean(X_pts, Y_pts)

        mu_X = np.random.randint(1, 5, size=N)
        mu_Y = np.random.randint(1, 5, size=N)
        diff = np.sum(mu_X) - np.sum(mu_Y)
        if diff > 0:
            mu_Y[0] += diff
        elif diff < 0:
            mu_X[0] += abs(diff)

        dense_pairs = N * N

        # LAP Auction (Strict 1-to-1)
        try:
            t0 = time.perf_counter()
            with SilencePrints():
                lap_solver = AuctionLAP(C)
                _, lap_cost, _ = lap_solver.solve()
            t_lap = time.perf_counter() - t0
            print(f"{N:<10} | {'LAP Auction':<15} | {dense_pairs:<12} | {lap_cost:<14.4f} | {t_lap:<12.5f}")
        except Exception as e:
            print(f"{N:<10} | {'LAP Auction':<15} | {'FAILED':<12} | {'FAILED':<14} | {'-':<12}")
            import traceback; traceback.print_exc()

        # Dense Auction (General OT)
        try:
            t1 = time.perf_counter()
            with SilencePrints():
                dense_manager = EpsScalingManager(AuctionOT, C, mu_X=mu_X, mu_Y=mu_Y)
                _, dense_cost, _, _ = dense_manager.solve()
            t_dense = time.perf_counter() - t1
            print(f"{N:<10} | {'Dense OT':<15} | {dense_pairs:<12} | {dense_cost:<14.4f} | {t_dense:<12.5f}")
        except Exception as e:
            print(f"{N:<10} | {'Dense OT':<15} | {'FAILED':<12} | {'FAILED':<14} | {'-':<12}")
            import traceback; traceback.print_exc()

        # Hierarchical Multiscale Auction
        try:
            tree_X, tree_Y = build_matched_trees(X_pts, Y_pts)

            t2 = time.perf_counter()
            with SilencePrints():
                multiscale_solver = HierarchicalMultiscaleSolver(tree_X, tree_Y, C, mu_X, mu_Y)
                hier_mu = multiscale_solver.solve()
            hier_cost = np.sum(hier_mu * C)
            t_hier = time.perf_counter() - t2

            hier_pairs = len(multiscale_solver.last_N_guess)

            print(f"{N:<10} | {'Hierarchical OT':<15} | {hier_pairs:<12} | {hier_cost:<14.4f} | {t_hier:<12.5f}")
        except Exception as e:
            print(f"{N:<10} | {'Hierarchical OT':<15} | {'FAILED':<12} | {'FAILED':<14} | {'-':<12}")
            import traceback; traceback.print_exc()

        print("-" * 80)

if __name__ == "__main__":
    run_comprehensive_benchmarks(N_sizes=[16, 32, 64, 128])
