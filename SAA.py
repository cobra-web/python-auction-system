import numpy as np

def ot_auction_round(mu_X, mu_Y, cost_matrix, neighbors, beta, epsilon):
    num_X, num_Y = cost_matrix.shape
    
    coupling = np.zeros((num_X, num_Y), dtype=np.int32)
    
    rem_mu_X = mu_X.copy()
    rem_mu_Y = mu_Y.copy()
    
    bids_received = {y: [] for y in range(num_Y)}
    
    max_inner_iters = 2000
    inner_iter = 0
    
    while np.any(rem_mu_X > 0) and inner_iter < max_inner_iters:
        for y in range(num_Y):
            bids_received[y].clear()
            
        for x in range(num_X):
            if rem_mu_X[x] > 0:
                x_neighbors = neighbors[x]
                if len(x_neighbors) == 0:
                    continue
                
                pi_list = []
                for y in x_neighbors:
                    val = cost_matrix[x, y] - beta[y]
                    pi_list.append((y, val))
                
                pi_list.sort(key=lambda item: item[1])
                
                mass_to_fill = rem_mu_X[x]
                pi_idx = 0
                
                while mass_to_fill > 0 and pi_idx < len(pi_list):
                    y_best, val_best = pi_list[pi_idx]
                    
                    if pi_idx + 1 < len(pi_list):
                        val_next = pi_list[pi_idx + 1][1]
                    else:
                        val_next = val_best + epsilon  # Fallback if no neighbor remains
                        
                    bid_value = cost_matrix[x, y_best] - val_next - epsilon
                    
                    offered_mass = min(mass_to_fill, rem_mu_Y[y_best])
                    if offered_mass <= 0:
                        offered_mass = rem_mu_X[x] # Default block offer if unassigned
                        
                    bids_received[y_best].append((x, bid_value, offered_mass))
                    
                    mass_to_fill -= offered_mass
                    pi_idx += 1

        for y in range(num_Y):
            if bids_received[y]:
                best_bid = min(bids_received[y], key=lambda item: item[1])
                best_x, lowest_bid, offered_mass = best_bid
                
                beta[y] = lowest_bid
                
                actual_mass = min(rem_mu_X[best_x], rem_mu_Y[y], offered_mass)
                if actual_mass > 0:
                    coupling[best_x, y] += actual_mass
                    rem_mu_X[best_x] -= actual_mass
                    rem_mu_Y[y] -= actual_mass
                    
        inner_iter += 1
        
    return coupling, beta


def optimal_transport_auction_with_scaling(mu_X, mu_Y, cost_matrix, neighborhood_mask, theta=4.0):
    num_X, num_Y = cost_matrix.shape
    
    neighbors = {x: np.where(neighborhood_mask[x, :])[0] for x in range(num_X)}
    
    beta = np.zeros(num_Y, dtype=np.float64)
    
    max_c = np.max(cost_matrix[neighborhood_mask])
    min_c = np.min(cost_matrix[neighborhood_mask])
    C = max_c - min_c
    
    epsilon = C / 4.0
    if epsilon == 0: 
        epsilon = 1.0
        
    delta_c = 1.0  # Assumes integer costs or fine grid resolution
    target_epsilon = (delta_c / num_X) - 1e-5
    
    scale_phase = 1
    current_coupling = None
    
    while epsilon > target_epsilon:
        print(f"--- Scaling Phase {scale_phase} | Current Epsilon: {epsilon:.4f} ---")
        
        current_coupling, beta = ot_auction_round(
            mu_X, mu_Y, cost_matrix, neighbors, beta, epsilon
        )
        
        epsilon /= theta
        scale_phase += 1
        
        if epsilon < 1e-6:
            break
            
    return current_coupling, beta


if __name__ == "__main__":
    # Total units of mass to balance across sets
    total_mass = 10
    
    # Supply distributions (mu_X) and Demand distributions (mu_Y)
    mu_X = np.array([5, 5], dtype=np.int32)
    mu_Y = np.array([3, 7], dtype=np.int32)
    
    # 2x2 Dense Cost Matrix mapping
    cost_matrix = np.array([
        [2.0, 8.0],
        [6.0, 1.0]
    ], dtype=np.float64)
    
    # Define a custom neighborhood set mask matrix N
    # Let's assume all paths are valid links except x0 to y1 (set to False/Infinity)
    neighborhood_mask = np.array([
        [True,  True],
        [True,  True]
    ], dtype=bool)
    
    print("Executing General OT Scaled Auction Solver...")
    optimal_coupling, final_prices = optimal_transport_auction_with_scaling(
        mu_X, mu_Y, cost_matrix, neighborhood_mask, theta=4.0
    )
    
    print("\n=== FINAL DISCOVERED OPTIMAL COUPLING ===")
    print(optimal_coupling)
    print("\n=== FINAL DUAL PRICE VARIABLES (BETA) ===")
    print(final_prices)