import time
import numpy as np
from src.utils.cost_functions import squared_euclidean
from src.hierarchical.partitions import HierarchicalPartition
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver
from src.utils.eps_scaling import EpsScalingManager
from src.core.ot_auction import AuctionOT

def run_hierarchical_benchmark(N=32):
    print(f"  RUNNING HIERARCHICAL SOLVER BENCHMARK (N={N})")
    np.random.seed(101)
    
    X_pts = np.random.rand(N, 2)
    Y_pts = np.random.rand(N, 2)
    
    #dense cost matrix
    C = squared_euclidean(X_pts, Y_pts)
    
    #balanced integer mass distributions
    mu_X = np.random.randint(1, 5, size=N)
    mu_Y = np.random.randint(1, 5, size=N)
    diff = np.sum(mu_X) - np.sum(mu_Y)
    if diff > 0:
        mu_Y[0] += diff
    elif diff < 0:
        mu_X[0] += abs(diff)
        
    print("Building spatial hierarchical partitions...")
    tree_X = HierarchicalPartition(X_pts)
    tree_Y = HierarchicalPartition(Y_pts)

    #if no match, rebuild the shallower one
    if tree_X.g != tree_Y.g:
        target_gen = max(tree_X.g, tree_Y.g)
        if tree_X.g < target_gen:
            tree_X = HierarchicalPartition(X_pts, target_g=target_gen)
        else:
            tree_Y = HierarchicalPartition(Y_pts, target_g=target_gen)
    print(f"Tree depth established and equalised: {tree_X.g} generations.")
    
    # RUN DENSE BASELINE
    print("\n[Executing Dense Baseline Solver...]")
    t0 = time.perf_counter()
    dense_manager = EpsScalingManager(AuctionOT, C, mu_X=mu_X, mu_Y=mu_Y)
    dense_mu, dense_cost, dense_iters, _ = dense_manager.solve()
    t_dense = time.perf_counter() - t0
    print(f"Dense Baseline Result -> Cost: {dense_cost:.4f} | Time: {t_dense:.4f}s")
    
    # RUN HIERARCHICAL SOLVER
    print("\n[Executing Hierarchical Multiscale Solver...]")
    t1 = time.perf_counter()
    multiscale_solver = HierarchicalMultiscaleSolver(tree_X, tree_Y, C, mu_X, mu_Y)
    hier_mu = multiscale_solver.solve()
    t_hier = time.perf_counter() - t1
    
    # Verify correctness
    final_hier_cost = np.sum(hier_mu * C)
    print("FINAL VERIFICATION:")
    print(f"Dense Total Cost:        {dense_cost:.4f}")
    print(f"Hierarchical Total Cost:  {final_hier_cost:.4f}")
    print(f"Runtimes -> Dense: {t_dense:.4f}s | Hierarchical: {t_hier:.4f}s")
    
    assert np.isclose(dense_cost, final_hier_cost, atol=0.1), "Error: Cost mismatch detected!"
    print("Success: The hierarchical multi-scale solver achieved global optimality!")

if __name__ == "__main__":
    run_hierarchical_benchmark(N=32)
