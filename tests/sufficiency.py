import numpy as np
import collections
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from src.core.ot_auction import AuctionOT
from src.utils.eps_scaling import EpsScalingManager
from src.hierarchical.partitions import build_matched_trees
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver
import ot

def test_neighborhood_sufficiency():
    print("\n--- DECISIVE NEIGHBORHOOD SUFFICIENCY TEST ---")
    N = 128  # Rapid verification
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

    # 1. Cold-start Dense OT Reference
    print("\n--- Running Cold-Start Dense Baseline ---")
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

    # 2. Build Matched Trees & Solve Hierarchical
    print("\n--- Extracting Neighborhood via Hierarchical Solver ---")
    tree_X, tree_Y = build_matched_trees(X_pts, Y_pts, max_points_per_cell=1)
    
    hier_solver = HierarchicalMultiscaleSolver(
        tree_X, tree_Y, mu_X, mu_Y, 
        max_c=GLOBAL_MAX_C, target_eps=tight_eps, min_eps=1e-6
    )
    _ = hier_solver.solve()
    
    # --- FIX 1: Translate Cell Indices to Point Indices ---
    final_edges_cells = hier_solver.last_N_guess
    final_X = tree_X.get_active_cells_at_depth(hier_solver.max_depth)
    final_Y = tree_Y.get_active_cells_at_depth(hier_solver.max_depth)
    
    assert len(final_X) == N and len(final_Y) == N, "Leaves are not one point each!"

    final_edges = [(final_X[i].point_indices[0], final_Y[j].point_indices[0])
                   for (i, j) in final_edges_cells]

    # --- FIX 2: Bipartite Graph Feasibility Guard via Max Flow ---
    total_mass = np.sum(mu_X)
    # Graph structure: Source (0) -> X_nodes (1..N) -> Y_nodes (N+1..2N) -> Sink (2N+1)
    source, sink = 0, 2 * N + 1
    row_ind, col_ind, data = [], [], []

    for i in range(N):
        row_ind.append(source)
        col_ind.append(i + 1)
        data.append(mu_X[i])

    for x, y in final_edges:
        row_ind.append(x + 1)
        col_ind.append(N + 1 + y)
        data.append(total_mass)

    for j in range(N):
        row_ind.append(N + 1 + j)
        col_ind.append(sink)
        data.append(mu_Y[j])

    capacity_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(2 * N + 2, 2 * N + 2))
    flow_result = maximum_flow(capacity_matrix, source, sink)

    print(f"Graph Capacity Check: Max Flow = {flow_result.flow_value:.2f} / Total Mass = {total_mass:.2f}")
    assert np.isclose(flow_result.flow_value, total_mass), "INFEASIBLE NEIGHBORHOOD: Max flow < total mass!"

    # 3. Solve Single-Level OT on Extracted Point Neighborhood
    print("\n--- Running Restricted Sparse Solve on Extracted Graph ---")
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

    print("\n" + "="*50)
    print(f"[Dense OT Gap @ eps={tight_eps}]:      {dense_gap:.4f}%")
    print(f"[Restricted Sparse Gap @ eps={tight_eps}]: {sparse_gap:.4f}%")
    print(f"Gap Difference:                       {abs(dense_gap - sparse_gap):.4f}%")
    print("="*50)

if __name__ == "__main__":
    test_neighborhood_sufficiency()
