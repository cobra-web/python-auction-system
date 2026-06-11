import numpy as np
import ot
import time

def get_grid_labels(points, grid_size):
    coords = np.floor(points * grid_size).astype(int)
    coords = np.clip(coords, 0, grid_size - 1)
    labels = coords[:, 0] * grid_size + coords[:, 1]
    return labels

def hierarchical_ot_multiscale(px, py, mu_x, mu_y, scales):
    N_x, N_y = len(px), len(py)
    cost_dense = ot.dist(px, py, metric='sqeuclidean')
    active_mask = np.zeros((N_x, N_y), dtype=bool)
    
    for scale in scales:
        labels_x = get_grid_labels(px, scale)
        labels_y = get_grid_labels(py, scale)
        
        unique_x, inv_x = np.unique(labels_x, return_inverse=True)
        unique_y, inv_y = np.unique(labels_y, return_inverse=True)
        
        coarse_mu_x = np.bincount(inv_x, weights=mu_x)
        coarse_mu_y = np.bincount(inv_y, weights=mu_y)
        
        centroids_x = np.array([px[labels_x == ux].mean(axis=0) for ux in unique_x])
        centroids_y = np.array([py[labels_y == uy].mean(axis=0) for uy in unique_y])
        
        coarse_cost = ot.dist(centroids_x, centroids_y, metric='sqeuclidean')
        
        coarse_coup = ot.emd(coarse_mu_x, coarse_mu_y, coarse_cost)
        coarse_support = np.argwhere(coarse_coup > 1e-9)
        
        for i, j in coarse_support:
            ux = unique_x[i]
            uy = unique_y[j]
            
            mask_x = (labels_x == ux)

            mask_y = (labels_y == uy)
            
            active_mask[np.ix_(mask_x, mask_y)] = True

        print(f"Skala {scale}x{scale}: {active_mask.sum()} / {N_x * N_y} Kandidaten aktiviert.")
    
    print("\nStarte Iterationen auf feinster Ebene...")
    for iteration in range(10): 
        cost_masked = np.where(active_mask, cost_dense, 1e9)
        
        coupling, log = ot.emd(mu_x, mu_y, cost_masked, log=True)
        total_cost = np.sum(coupling * cost_dense)
        
        alpha = log['u']
        beta = log['v']
        
        slack = alpha[:, None] + beta[None, :] - cost_dense
        
        violation_mask = (slack > 1e-5) & (~active_mask)
        num_violations = violation_mask.sum()
        
        print(f"  Runde {iteration+1}: {active_mask.sum()} aktiv | Kosten = {total_cost:.5f} | Neue Verletzungen = {num_violations}")
        
        if num_violations == 0:
            break 
            
        active_mask |= violation_mask
        
    return coupling, total_cost, active_mask.sum()

if __name__ == "__main__":
    np.random.seed(42)
    N = 2000 # Reduziert auf 2000 für schnellere Durchläufe beim Testen

    px, py = np.random.rand(N, 2), np.random.rand(N, 2)
    mu_x, mu_y = np.ones(N)/N, np.ones(N)/N
    
    cost_dense = ot.dist(px, py, metric='sqeuclidean')
    
    print("--- Starte Dense OT ---")
    t0_dense = time.perf_counter()
    coup_dense = ot.emd(mu_x, mu_y, cost_dense)
    cost_dense_val = np.sum(coup_dense * cost_dense)
    time_dense = (time.perf_counter() - t0_dense) * 1000
    
    print("\n--- Starte Hierarchical OT ---")
    t0_sparse = time.perf_counter()
    coup_sparse, cost_sparse_val, final_pairs = hierarchical_ot_multiscale(px, py, mu_x, mu_y, scales=[2, 4, 8, 16])
    time_sparse = (time.perf_counter() - t0_sparse) * 1000
    
    print(f"\nErgebnis Dense OT:  cost = {cost_dense_val:.5f} | time = {time_dense:.1f}ms | pairs = {N*N}")
    print(f"Ergebnis Sparse OT: cost = {cost_sparse_val:.5f} | time = {time_sparse:.1f}ms | pairs = {final_pairs}")