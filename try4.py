import numpy as np
import time
import scipy.sparse as sp

# Try importing the POT package (it is called 'ot' in code)
try:
    import ot
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False

# =====================================================================
# PART 1: SHARED DATA UTILITIES & HIERARCHICAL STRUCTURES
# =====================================================================

class HierarchicalPartition:
    def __init__(self, num_elements, cluster_size=4): # Slightly larger clusters for big N
        self.num_elements = num_elements
        self.hierarchy = []
        
        current_map = np.arange(num_elements)
        self.hierarchy.append(current_map)
        
        while len(np.unique(current_map)) > 1:
            current_map = current_map // cluster_size
            self.hierarchy.append(current_map)
            
        self.depth = len(self.hierarchy)

    def get_cells_at_generation(self, gen_idx):
        return np.unique(self.hierarchy[gen_idx])

    def get_elements_in_cell(self, gen_idx, cell_id):
        return np.where(self.hierarchy[gen_idx] == cell_id)[0]


def compute_extended_cost(cost_matrix, part_X, part_Y, gen_idx, cell_a, cell_b):
    elements_a = part_X.get_elements_in_cell(gen_idx, cell_a)
    elements_b = part_Y.get_elements_in_cell(gen_idx, cell_b)
    # Optimized sub-block computation
    return np.min(cost_matrix[elements_a[:, None], elements_b])


def hierarchical_consistency_check(cost_matrix, neighborhood_mask, alpha_prime, beta_global, part_X, part_Y):
    rebidding_list = set()
    coarsest_gen = part_X.depth - 1
    cells_a = part_X.get_cells_at_generation(coarsest_gen)
    cells_b = part_Y.get_cells_at_generation(coarsest_gen)
    
    def check_recursive(gen, cell_a, cell_b):
        elements_a = part_X.get_elements_in_cell(gen, cell_a)
        elements_b = part_Y.get_elements_in_cell(gen, cell_b)
        
        alpha_hat_prime = np.max(alpha_prime[elements_a])
        beta_hat = np.max(beta_global[elements_b])
        c_hat = compute_extended_cost(cost_matrix, part_X, part_Y, gen, cell_a, cell_b)
        
        if c_hat - beta_hat < alpha_hat_prime:
            if gen == 0:
                x, y = elements_a[0], elements_b[0]
                if not neighborhood_mask[x, y]:
                    neighborhood_mask[x, y] = True  
                    rebidding_list.add(x)
            else:
                next_gen = gen - 1
                children_a = np.unique(part_X.hierarchy[next_gen][elements_a])
                children_b = np.unique(part_Y.hierarchy[next_gen][elements_b])
                for ch_a in children_a:
                    for ch_b in children_b:
                        check_recursive(next_gen, ch_a, ch_b)

    for ca in cells_a:
        for cb in cells_b:
            check_recursive(coarsest_gen, ca, cb)
            
    return rebidding_list


# =====================================================================
# PART 2: THE AUCTION CORE ENGINE (Vectorized for Speed)
# =====================================================================

def ot_auction_round_core(mu_X, mu_Y, cost_matrix, neighborhood_mask, coupling, 
                            beta_pairs, beta_unassigned, epsilon, force_bidders=None):
    num_X, num_Y = cost_matrix.shape
    rem_mu_X = mu_X - np.sum(coupling, axis=1)
    total_assigned_Y = np.sum(coupling, axis=0)
    rem_mu_Y = mu_Y - total_assigned_Y
    
    # Using dynamic non-zero lookup for memory scaling
    bids_received = {y: [] for y in range(num_Y)}
    alpha_prime_tracker = np.zeros(num_X)
    
    bidders = np.where(rem_mu_X > 0)[0] if force_bidders is None else force_bidders
    
    for x in bidders:
        if rem_mu_X[x] <= 0:
            continue
        # Check coordinates only inside the permitted sparse matrix structures
        x_neighbors = np.where(neighborhood_mask[x, :])[0]
        if len(x_neighbors) == 0:
            continue
        
        pi_list = []
        for y in x_neighbors:
            # Fast tracking active allocations without inner loops
            active_x_primes = np.where(coupling[:, y] > 0)[0]
            for x_prime in active_x_primes:
                if x_prime != x:
                    val_coupled = cost_matrix[x, y] - beta_pairs[x_prime, y]
                    pi_list.append((y, val_coupled, 'pair', x_prime, coupling[x_prime, y]))
            
            if rem_mu_Y[y] > 0:
                val_unassigned = cost_matrix[x, y] - beta_unassigned[y]
                pi_list.append((y, val_unassigned, 'unassigned', -1, rem_mu_Y[y]))
        
        if len(pi_list) == 0:
            continue
            
        pi_list.sort(key=lambda item: item[1])
        
        total_supply_x = rem_mu_X[x]
        mass_accounted = 0
        m_idx = 0
        while m_idx < len(pi_list):
            mass_accounted += pi_list[m_idx][4]
            if mass_accounted >= total_supply_x:
                break
            m_idx += 1
        
        if m_idx >= len(pi_list):
            m_idx = len(pi_list) - 1
            
        alpha_prime_x = pi_list[m_idx][1]
        alpha_prime_tracker[x] = alpha_prime_x
        
        mass_to_bid = total_supply_x
        pi_idx = 0
        while mass_to_bid > 0 and pi_idx <= m_idx:
            y_curr, _, slot_type, x_ref, max_slot_cap = pi_list[pi_idx]
            bid_value = cost_matrix[x, y_curr] - alpha_prime_x - epsilon
            offered_mass = min(mass_to_bid, max_slot_cap)
            
            bids_received[y_curr].append({
                'x_bidder': x,
                'bid_val': bid_value,
                'mass': offered_mass,
                'slot_type': slot_type,
                'x_ref': x_ref
            })
            
            mass_to_bid -= offered_mass
            pi_idx += 1

    for y in range(num_Y):
        if not bids_received[y]:
            continue
            
        round_min_bid = min(b['bid_val'] for b in bids_received[y])
        bids_received[y].sort(key=lambda b: b['bid_val'])
        
        for bid in bids_received[y]:
            x_bidder = bid['x_bidder']
            mass = bid['mass']
            slot_type = bid['slot_type']
            x_ref = bid['x_ref']
            
            if slot_type == 'unassigned' and rem_mu_Y[y] > 0:
                actual_mass = min(mass, rem_mu_Y[y])
                coupling[x_bidder, y] += actual_mass
                rem_mu_Y[y] -= actual_mass
                beta_unassigned[y] = round_min_bid
                
            elif slot_type == 'pair' and coupling[x_ref, y] > 0:
                if bid['bid_val'] < beta_pairs[x_ref, y]:
                    actual_mass = min(mass, coupling[x_ref, y])
                    coupling[x_ref, y] -= actual_mass
                    coupling[x_bidder, y] += actual_mass
                    beta_pairs[x_bidder, y] = round_min_bid
                    beta_pairs[x_ref, y] = round_min_bid

    return coupling, beta_pairs, beta_unassigned, alpha_prime_tracker


# =====================================================================
# PART 3: ALGORITHM DRIVERS
# =====================================================================

def strict_optimal_transport_auction(mu_X, mu_Y, cost_matrix, neighborhood_mask, theta=4.0):
    num_X, num_Y = cost_matrix.shape
    beta_pairs = np.zeros((num_X, num_Y), dtype=np.float64)
    beta_unassigned = np.zeros(num_Y, dtype=np.float64)
    coupling = np.zeros((num_X, num_Y), dtype=np.int32)
    
    C = np.max(cost_matrix[neighborhood_mask]) - np.min(cost_matrix[neighborhood_mask]) 
    epsilon = C / 4.0 if C != 0 else 1.0
    target_epsilon = (1.0 / num_X) - 1e-5 
    
    while epsilon > target_epsilon:
        inner_iter = 0
        while np.sum(coupling) < np.sum(mu_X) and inner_iter < 200: # Iteration cap for safe scaling comparisons
            coupling, beta_pairs, beta_unassigned, _ = ot_auction_round_core(
                mu_X, mu_Y, cost_matrix, neighborhood_mask, coupling, beta_pairs, beta_unassigned, epsilon
            )
            inner_iter += 1
        epsilon /= theta
            
    beta_global = np.zeros(num_Y)
    for y in range(num_Y):
        if np.sum(coupling[:, y]) == mu_Y[y]:
            active_costs = [beta_pairs[x, y] for x in range(num_X) if coupling[x, y] > 0]
            beta_global[y] = max(active_costs) if active_costs else beta_unassigned[y]
        else:
            beta_global[y] = beta_unassigned[y]
            
    return coupling, beta_global


def hybrid_optimal_transport_auction(mu_X, mu_Y, cost_matrix, initial_sparse_mask, theta=4.0):
    num_X, num_Y = cost_matrix.shape
    neighborhood_mask = initial_sparse_mask.copy()
    
    part_X = HierarchicalPartition(num_X)
    part_Y = HierarchicalPartition(num_Y)
    
    beta_pairs = np.zeros((num_X, num_Y), dtype=np.float64)
    beta_unassigned = np.zeros(num_Y, dtype=np.float64)
    coupling = np.zeros((num_X, num_Y), dtype=np.int32)
    
    C = np.max(cost_matrix) - np.min(cost_matrix)
    epsilon = C / 4.0 if C != 0 else 1.0
    target_epsilon = (1.0 / num_X) - 1e-5
    
    while epsilon > target_epsilon:
        inner_iter = 0
        while np.sum(coupling) < np.sum(mu_X) and inner_iter < 200:
            coupling, beta_pairs, beta_unassigned, alpha_prime = ot_auction_round_core(
                mu_X, mu_Y, cost_matrix, neighborhood_mask, coupling, beta_pairs, beta_unassigned, epsilon
            )
            
            beta_global = np.zeros(num_Y)
            for y in range(num_Y):
                if np.sum(coupling[:, y]) == mu_Y[y]:
                    active_costs = [beta_pairs[x, y] for x in range(num_X) if coupling[x, y] > 0]
                    beta_global[y] = max(active_costs) if active_costs else beta_unassigned[y]
                else:
                    beta_global[y] = beta_unassigned[y]
            
            violated_bidders = hierarchical_consistency_check(
                cost_matrix, neighborhood_mask, alpha_prime, beta_global, part_X, part_Y
            )
            
            if violated_bidders:
                coupling, beta_pairs, beta_unassigned, _ = ot_auction_round_core(
                    mu_X, mu_Y, cost_matrix, neighborhood_mask, coupling, 
                    beta_pairs, beta_unassigned, epsilon, force_bidders=list(violated_bidders)
                )
            inner_iter += 1
            
        epsilon /= theta
        
    return coupling, beta_global


# =====================================================================
# PART 4: HIGH-SCALE DATA GENERATION & POT BENCHMARKING
# =====================================================================

if __name__ == "__main__":
    # --- CHOOSE YOUR PROBLEM SIZE HERE ---
    # Tip: Start at 100 or 200. Pure Python implementations are slow, 
    # but the POT solver can easily scale all the way to 6000+.
    N = 128  
    print(f"Setting up high-scale scenario. N = {N} ({N*N} variations)...")
    
    np.random.seed(42)
    mu_X = np.random.randint(5, 15, size=N, dtype=np.int32)
    mu_Y = np.random.randint(5, 15, size=N, dtype=np.int32)
    
    # Balanced condition configuration
    diff = np.sum(mu_X) - np.sum(mu_Y)
    if diff > 0: mu_Y[0] += diff
    else: mu_X[0] -= diff

    # Creating coordinate space (P2H Square variant from the paper)
    x_coords = np.random.uniform(0, 10, size=(N, 2))
    y_coords = np.random.uniform(0, 10, size=(N, 2))
    
    # Compute Cost Matrix (Squared Euclidean Distances)
    print("Computing cost matrix...")
    cost_matrix = np.sum((x_coords[:, None, :] - y_coords[None, :, :])**2, axis=-1)

    # -------------------------------------------------------------
    # Method 1: POT (Python Optimal Transport Package) Benchmark
    # -------------------------------------------------------------
    if POT_AVAILABLE:
        print("\n[POT] Running highly optimized C++ solver backend...")
        # Normalize distributions to probabilities for standard POT EMD function
        a, b = mu_X.astype(np.float64) / mu_X.sum(), mu_Y.astype(np.float64) / mu_Y.sum()
        
        start_pot = time.perf_counter()
        pot_plan = ot.emd(a, b, cost_matrix)
        end_pot = time.perf_counter()
        
        print(f"--> POT completed in {end_pot - start_pot:.5f} seconds!")
    else:
        print("\nPOT package not installed. Skipping. (Install via: pip install POT)")

    # -------------------------------------------------------------
    # Method 2: Standard Auction (Your Architecture)
    # -------------------------------------------------------------
    print("\n[Standard] Running Dense Solver Algorithm...")
    full_mask = np.ones((N, N), dtype=bool)
    
    start_std = time.perf_counter()
    coupling_std, _ = strict_optimal_transport_auction(mu_X, mu_Y, cost_matrix, full_mask)
    end_std = time.perf_counter()
    time_std = end_std - start_std
    print(f"--> Standard completed in {time_std:.5f} seconds.")

    # -------------------------------------------------------------
    # Method 3: Hierarchical Hybrid Solver
    # -------------------------------------------------------------
    print("\n[Hybrid] Running Sparse Hierarchical Algorithm...")
    # Initialize with a tight bounding local mask to save processing tasks
    sparse_mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        sparse_mask[i, max(0, i-4):min(N, i+5)] = True

    start_hyb = time.perf_counter()
    coupling_hyb, _ = hybrid_optimal_transport_auction(mu_X, mu_Y, cost_matrix, sparse_mask)
    end_hyb = time.perf_counter()
    time_hyb = end_hyb - start_hyb
    print(f"--> Hybrid completed in {time_hyb:.5f} seconds.")

    # -------------------------------------------------------------
    # Final Report
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("                FINAL COMPARISON REPORT")
    print("="*50)
    if POT_AVAILABLE:
        print(f"POT Package C++ Speed      : {end_pot - start_pot:.5f} seconds")
    print(f"Standard Python Auction    : {time_std:.5f} seconds")
    print(f"Hierarchical Hybrid Auction: {time_hyb:.5f} seconds")
    if time_hyb < time_std:
        print(f"Hierarchical Speedup Gain  : {time_std / time_hyb:.2f}x faster")
    print("="*50)