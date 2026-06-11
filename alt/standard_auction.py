import numpy as np
import ot  

def standard_auction_algorithm(cost_matrix, initial_beta=None, epsilon=1e-3):
    num_x, num_y = cost_matrix.shape
    
    S = np.full(num_x, -1) 
    
    if initial_beta is not None:
        beta = np.copy(initial_beta)
    else:
        beta = np.zeros(num_y)
        
    while -1 in S:
        unassigned_x = np.where(S == -1)[0]
        bids = {y: [] for y in range(num_y)}
        
        for x in unassigned_x:
            costs_minus_beta = cost_matrix[x, :] - beta
            
            sorted_indices = np.argsort(costs_minus_beta)
            y_star = sorted_indices[0]
            
            if len(sorted_indices) > 1:
                alpha_prime = costs_minus_beta[sorted_indices[1]]
            else:
                alpha_prime = costs_minus_beta[0]
                
            bid_value = cost_matrix[x, y_star] - alpha_prime - epsilon
            bids[y_star].append((x, bid_value))
            
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
    num_points = 50
    xs = np.random.rand(num_points, 2)
    xt = np.random.rand(num_points, 2)

    print("Berechne Kostenmatrix mit ot.dist...")
    cost_matrix = ot.dist(xs, xt, metric='sqeuclidean')

    a, b = np.ones((num_points,)) / num_points, np.ones((num_points,)) / num_points

    print("Löse exaktes Transportproblem mit ot.emd, um Initialwerte zu erhalten...")
    T, log = ot.emd(a, b, cost_matrix, log=True)

    initial_beta_pot = log['v'] 

    print("Starte Auktionsalgorithmus mit POT-Initialwerten...")
    assignment, final_beta = standard_auction_algorithm(
        cost_matrix=cost_matrix, 
        initial_beta=initial_beta_pot, 
        epsilon=1e-3
    )
    
    print("Zuweisung abgeschlossen.")