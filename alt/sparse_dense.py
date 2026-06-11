import numpy as np
import ot

def build_hierarchical_grid_costs(cost_matrix, num_generations):

    hierarchies = [cost_matrix]
    current_grid = cost_matrix
    
    for _ in range(num_generations - 1):

        new_shape = (current_grid.shape[0] // 2, current_grid.shape[1] // 2)
        new_grid = np.zeros(new_shape)
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                new_grid[i, j] = np.min(current_grid[2*i:2*i+2, 2*j:2*j+2])
        hierarchies.append(new_grid)
        current_grid = new_grid
        
    return hierarchies[::-1] 

def hierarchical_auction_algorithm(cost_matrix, initial_beta=None, epsilon=1e-3, generations=3):

    num_x, num_y = cost_matrix.shape
    
    N_hat = {x: [x] for x in range(num_x)} 
    
    S = np.full(num_x, -1)
    
    if initial_beta is not None:
        beta = np.copy(initial_beta)
    else:
        beta = np.zeros(num_y)
    
    c_hat_hierarchies = build_hierarchical_grid_costs(cost_matrix, generations)
    
    while -1 in S:
        unassigned_x = np.where(S == -1)[0]
        bids = {y: [] for y in range(num_y)}
        alpha_prime_list = np.zeros(num_x)
        
        for x in unassigned_x:
            neighbors = N_hat[x]
            costs_minus_beta = [cost_matrix[x, y] - beta[y] for y in neighbors]
            
            sorted_idx = np.argsort(costs_minus_beta)
            y_star = neighbors[sorted_idx[0]]
            
            if len(sorted_idx) > 1:
                alpha_prime = costs_minus_beta[sorted_idx[1]]
            else:
                alpha_prime = costs_minus_beta[0]
                
            alpha_prime_list[x] = alpha_prime
            bid_value = cost_matrix[x, y_star] - alpha_prime - epsilon
            bids[y_star].append((x, bid_value))
            
        for x in unassigned_x:
            for y in range(num_y):
                if y not in N_hat[x]:
                    # Wenn die Bedingung verletzt ist, erweitern wir N_hat
                    if cost_matrix[x, y] - beta[y] < alpha_prime_list[x]:
                        N_hat[x].append(y)
        
        for y, received_bids in bids.items():
            if not received_bids:
                continue
            lowest_bid_x, lowest_bid_value = min(received_bids, key=lambda item: item[1])
            beta[y] = lowest_bid_value
            
            old_x = np.where(S == y)[0]
            if len(old_x) > 0:
                S[old_x[0]] = -1
            S[lowest_bid_x] = y
            
    return S, beta

if __name__ == "__main__":
    num_points = 100
    xs = np.random.rand(num_points, 2)
    xt = np.random.rand(num_points, 2)

    cost_matrix_fine = ot.dist(xs, xt, metric='sqeuclidean')

    hierarchies = build_hierarchical_grid_costs(cost_matrix_fine, num_generations=2)
    cost_matrix_coarse = hierarchies[0] # Das ist die 50x50 Matrix

    print("Löse grobes Problem mit POT ot.emd...")
    a_coarse = np.ones(50) / 50
    b_coarse = np.ones(50) / 50
    T_coarse, log_coarse = ot.emd(a_coarse, b_coarse, cost_matrix_coarse, log=True)
    
    beta_coarse = log_coarse['v']

    print("Skaliere duale Variablen für die feine Stufe hoch...")
    beta_fine_initial = np.zeros(num_points)
    for i in range(num_points):
        beta_fine_initial[i] = beta_coarse[i // 2]

    print("Starte hybriden Algorithmus auf feiner Matrix (100x100) mit POT-Initialwerten...")
    assignment, final_beta = hierarchical_auction_algorithm(
        cost_matrix=cost_matrix_fine, 
        initial_beta=beta_fine_initial, 
        epsilon=1e-3,
        generations=2
    )
    
    print("Hybride Zuweisung erfolgreich abgeschlossen.")