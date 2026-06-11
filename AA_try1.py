import numpy as np
import ot
import time

def auction_algorithm_lap(cost_matrix, epsilon=0.01, initial_prices=None):
    N = cost_matrix.shape[0]
    
    if initial_prices is not None:
        prices = initial_prices.copy() # beta(y) im Paper
    else:
        prices = np.zeros(N)
        
    assignment = np.full(N, -1) # Welches y gehört zu welchem x? (-1 = unassigned)
    owner = np.full(N, -1)      # Welches x besitzt welches y? (-1 = unassigned)
    
    unassigned_bidders = list(range(N))
    
    iterations = 0
    while unassigned_bidders:
        iterations += 1
        
        bids = {} # dictionary für die Gebote: y -> (x, bid_value)
        
        for x in unassigned_bidders:
            net_costs = cost_matrix[x, :] - prices
            
            best_y, second_best_y = np.partition(net_costs, 1)[:2]
            best_y_idx = np.where(net_costs == best_y)[0][0]
            
            bid_value = prices[best_y_idx] + (second_best_y - best_y) + epsilon
            
            if best_y_idx not in bids or bid_value > bids[best_y_idx][1]:
                bids[best_y_idx] = (x, bid_value)
                
        for y, (x, bid_value) in bids.items():
            prices[y] = bid_value
            
            old_owner = owner[y]
            if old_owner != -1:
                assignment[old_owner] = -1
                unassigned_bidders.append(old_owner)
            
            assignment[x] = y
            owner[y] = x
            unassigned_bidders.remove(x)
            
    total_cost = np.sum(cost_matrix[np.arange(N), assignment])
    
    return assignment, prices, total_cost, iterations

if __name__ == "__main__":
    np.random.seed(42)
    N = 100 # Kleine Zahl für puren Python-Code
    px, py = np.random.rand(N, 2), np.random.rand(N, 2)
    
    cost = ot.dist(px, py, metric='sqeuclidean')
    
    print("Starte reinen Auktions-Algorithmus (Kaltstart)...")
    t0 = time.perf_counter()
    assign, final_prices, cost_val, iters = auction_algorithm_lap(cost, epsilon=0.01)
    t1 = time.perf_counter()
    
    print(f"Kosten: {cost_val:.4f} | Iterationen: {iters} | Zeit: {(t1-t0)*1000:.1f} ms")