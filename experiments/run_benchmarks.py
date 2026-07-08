import time
import os
import sys
import traceback
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


def build_matched_trees(X_pts, Y_pts, max_points_per_cell=8, max_allowed_depth=10):
    # First pass: find natural depth of each point set independently
    probe_X = HierarchicalPartition(
        X_pts, 
        max_points_per_cell=max_points_per_cell,
        max_allowed_depth=max_allowed_depth
    )
    probe_Y = HierarchicalPartition(
        Y_pts, 
        max_points_per_cell=max_points_per_cell,
        max_allowed_depth=max_allowed_depth
    )

    target_g = max(probe_X.g, probe_Y.g)

    # Second pass: rebuild both to the same target_g
    tree_X = HierarchicalPartition(
        X_pts, 
        target_g=target_g,
        max_points_per_cell=max_points_per_cell,
        max_allowed_depth=max_allowed_depth
    )
    tree_Y = HierarchicalPartition(
        Y_pts, 
        target_g=target_g,
        max_points_per_cell=max_points_per_cell,
        max_allowed_depth=max_allowed_depth
    )

    assert tree_X.g == tree_Y.g == target_g
    return tree_X, tree_Y


def generate_problem_instance(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random point locations in [0, 1]^2
    X_pts = np.random.rand(N, 2)
    Y_pts = np.random.rand(N, 2)
    
    # Compute squared Euclidean distance cost
    C = squared_euclidean(X_pts, Y_pts)
    
    # Generate random masses in range [1, 5)
    mu_X_raw = np.random.randint(1, 5, size=N).astype(float)
    mu_Y_raw = np.random.randint(1, 5, size=N).astype(float)
    
    # ===== CRITICAL: Balance the masses to have exact equal sums =====
    # Method: Scale both to have the same total mass (use their average)
    total_X = np.sum(mu_X_raw)
    total_Y = np.sum(mu_Y_raw)
    total_avg = (total_X + total_Y) / 2.0
    
    # Scale both mass vectors to the average total
    # This ensures: sum(mu_X) == sum(mu_Y) == total_avg
    mu_X = mu_X_raw * (total_avg / total_X)
    mu_Y = mu_Y_raw * (total_avg / total_Y)
    
    # Verify balance (should be exact up to floating point error)
    assert np.isclose(np.sum(mu_X), np.sum(mu_Y), rtol=1e-10), \
        f"Mass imbalance: sum(mu_X)={np.sum(mu_X)}, sum(mu_Y)={np.sum(mu_Y)}"
    
    return X_pts, Y_pts, C, mu_X, mu_Y


def run_comprehensive_benchmarks(N_sizes=[16, 32, 64, 128], base_seed=42, verbose=True):
    print("\n" + "="*110)
    print("BACHELOR THESIS: OPTIMAL TRANSPORT SOLVER BENCHMARK")
    print("="*110)
    print(f"{'Scale':<6} | {'Solver':<18} | {'Active Pairs':<15} | {'Computed Cost':<18} | {'Runtime (s)':<12}")
    print("-"*110)
    
    results = {}  # Store results for comparison
    
    for N in N_sizes:
        # Generate problem instance once, use for all three solvers
        # Use unique seed for each N but reproducible across runs
        instance_seed = base_seed + N
        X_pts, Y_pts, C, mu_X, mu_Y = generate_problem_instance(N, seed=instance_seed)
        
        # Verify balance
        if verbose:
            print(f"  [Problem N={N}] sum(mu_X) = {np.sum(mu_X):.6f}, sum(mu_Y) = {np.sum(mu_Y):.6f}")
        
        dense_pairs = N * N
        results[N] = {}
        
        # ===== LAP AUCTION (1-to-1 assignment) =====
        try:
            t0 = time.perf_counter()
            with SilencePrints():
                lap_solver = AuctionLAP(C)
                _, lap_cost, _ = lap_solver.solve()
            t_lap = time.perf_counter() - t0
            
            results[N]['LAP'] = {'cost': lap_cost, 'time': t_lap, 'pairs': dense_pairs}
            print(f"{N:<6} | {'LAP Auction':<18} | {dense_pairs:<15} | {lap_cost:<18.8f} | {t_lap:<12.5f}")
        except Exception as e:
            print(f"{N:<6} | {'LAP Auction':<18} | {'FAILED':<15} | {'FAILED':<18} | {'-':<12}")
            if verbose:
                print(f"  ERROR: {type(e).__name__}: {str(e)}")
                traceback.print_exc()

        # ===== DENSE OT AUCTION =====
        try:
            if verbose:
                print(f"  [Dense OT] Starting with cost shape {C.shape}, mu_X {mu_X.shape}, mu_Y {mu_Y.shape}")
            
            t1 = time.perf_counter()
            with SilencePrints():
                # Pass mu_X and mu_Y as keyword arguments
                dense_manager = EpsScalingManager(
                    AuctionOT, 
                    C, 
                    mu_X=mu_X, 
                    mu_Y=mu_Y
                )
                dense_mu, dense_cost, dense_iters, _ = dense_manager.solve()
            t_dense = time.perf_counter() - t1
            
            # Verify coupling is valid
            assigned_X = np.sum(dense_mu, axis=1)
            assigned_Y = np.sum(dense_mu, axis=0)
            
            supply_deficit = np.max(np.abs(assigned_X - mu_X))
            demand_deficit = np.max(np.abs(assigned_Y - mu_Y))
            
            if verbose:
                print(f"  [Dense OT] Supply deficit: {supply_deficit:.2e}, Demand deficit: {demand_deficit:.2e}")
            
            assert np.allclose(assigned_X, mu_X, atol=1e-6), \
                f"Dense OT: supply not satisfied (deficit={supply_deficit:.2e})"
            assert np.allclose(assigned_Y, mu_Y, atol=1e-6), \
                f"Dense OT: demand not satisfied (deficit={demand_deficit:.2e})"
            
            results[N]['Dense'] = {'cost': dense_cost, 'time': t_dense, 'pairs': dense_pairs, 'iters': dense_iters}
            print(f"{N:<6} | {'Dense OT':<18} | {dense_pairs:<15} | {dense_cost:<18.8f} | {t_dense:<12.5f}")
        except Exception as e:
            print(f"{N:<6} | {'Dense OT':<18} | {'FAILED':<15} | {'FAILED':<18} | {'-':<12}")
            print(f"  ERROR: {type(e).__name__}: {str(e)}")
            if verbose:
                traceback.print_exc()

        # ===== HIERARCHICAL MULTISCALE OT =====
        try:
            if verbose:
                print(f"  [Hierarchical] Building matched trees...")
            
            tree_X, tree_Y = build_matched_trees(X_pts, Y_pts)

            t2 = time.perf_counter()
            with SilencePrints():
                multiscale_solver = HierarchicalMultiscaleSolver(tree_X, tree_Y, C, mu_X, mu_Y)
                hier_mu = multiscale_solver.solve()
            t_hier = time.perf_counter() - t2

            # Compute cost using ORIGINAL (unnormalized) cost matrix
            hier_cost = np.sum(hier_mu * C)
            hier_pairs = len(multiscale_solver.last_N_guess) if hasattr(multiscale_solver, 'last_N_guess') else 0
            
            # Verify coupling is valid
            assigned_X = np.sum(hier_mu, axis=1)
            assigned_Y = np.sum(hier_mu, axis=0)
            
            supply_deficit = np.max(np.abs(assigned_X - mu_X))
            demand_deficit = np.max(np.abs(assigned_Y - mu_Y))
            
            if verbose:
                print(f"  [Hierarchical] Supply deficit: {supply_deficit:.2e}, Demand deficit: {demand_deficit:.2e}")
                print(f"  [Hierarchical] Active pairs: {hier_pairs}/{dense_pairs} ({100.0*hier_pairs/dense_pairs:.1f}% density)")
            
            assert np.allclose(assigned_X, mu_X, atol=1e-6), \
                f"Hierarchical: supply not satisfied (deficit={supply_deficit:.2e})"
            assert np.allclose(assigned_Y, mu_Y, atol=1e-6), \
                f"Hierarchical: demand not satisfied (deficit={demand_deficit:.2e})"

            results[N]['Hierarchical'] = {'cost': hier_cost, 'time': t_hier, 'pairs': hier_pairs}
            print(f"{N:<6} | {'Hierarchical OT':<18} | {hier_pairs:<15} | {hier_cost:<18.8f} | {t_hier:<12.5f}")
        except Exception as e:
            print(f"{N:<6} | {'Hierarchical OT':<18} | {'FAILED':<15} | {'FAILED':<18} | {'-':<12}")
            print(f"  ERROR: {type(e).__name__}: {str(e)}")
            if verbose:
                traceback.print_exc()

        print("-"*110)

    # ===== VALIDATION AND COMPARISON =====
    print("\n" + "="*110)
    print("COST VALIDATION (Dense OT should match Hierarchical OT)")
    print("="*110)
    
    match_count = 0
    mismatch_count = 0
    
    for N in N_sizes:
        if 'Dense' in results[N] and 'Hierarchical' in results[N]:
            dense_cost = results[N]['Dense']['cost']
            hier_cost = results[N]['Hierarchical']['cost']
            cost_diff = abs(dense_cost - hier_cost)
            rel_diff = cost_diff / dense_cost if dense_cost != 0 else 0
            
            # Threshold for matching: 1e-5 (0.001%) relative difference
            match_status = "✓ MATCH" if rel_diff < 1e-5 else "✗ MISMATCH"
            if rel_diff < 1e-5:
                match_count += 1
            else:
                mismatch_count += 1
            
            print(f"N={N:<4} | Dense: {dense_cost:<18.8f} | Hier: {hier_cost:<18.8f} | "
                  f"Diff: {cost_diff:<12.2e} | Rel: {rel_diff:<12.2e} | {match_status}")
        else:
            print(f"N={N:<4} | INCOMPLETE DATA (one or both solvers failed)")
            mismatch_count += 1
    
    print("="*110)
    print(f"SUMMARY: {match_count} matches, {mismatch_count} mismatches")
    print("="*110 + "\n")
    
    return results


if __name__ == "__main__":
    # Run benchmarks with explicit error reporting
    # Change verbose=False for cleaner output without debug info
    results = run_comprehensive_benchmarks(N_sizes=[16, 32, 64, 128], base_seed=42, verbose=True)
