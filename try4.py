import numpy as np
import time
import scipy.sparse as sp
import ot

class HierarchicalPartition:
    def __init__(self, num_elements, cluster_size=4):
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
    return np.min(cost_matrix[elements_a[:, None], elements_b])


def hierarchical_consistency_check_ot(cost_matrix, neighborhood_mask, alpha_prime_tracker, 
                                      coupling, beta_pairs, beta_unassigned, mu_Y, part_X, part_Y):
                                          
    num_X, num_Y = cost_matrix.shape
    mask_extended = False
    coarsest_gen = part_X.depth - 1
    cells_a = part_X.get_cells_at_generation(coarsest_gen)
    cells_b = part_Y.get_cells_at_generation(coarsest_gen)
    
    beta_effective = np.zeros((num_X, num_Y))
    for y in range(num_Y):
        total_assigned_y = np.sum(coupling[:, y])
        y_has_unassigned = total_assigned_y < mu_Y[y]
        for x in range(num_X):
            if coupling[x, y] > 0:
                beta_effective[x, y] = beta_pairs[x, y]
            elif y_has_unassigned:
                beta_effective[x, y] = beta_unassigned[y]
            else:
                active_b = beta_pairs[:, y][coupling[:, y] > 0]
                beta_effective[x, y] = np.max(active_b) if len(active_b) > 0 else beta_unassigned[y]

    def check_recursive(gen, cell_a, cell_b):
        nonlocal mask_extended
        elements_a = part_X.get_elements_in_cell(gen, cell_a)
        elements_b = part_Y.get_elements_in_cell(gen, cell_b)
        
        alpha_hat_prime = np.max(alpha_prime_tracker[elements_a])
        beta_hat = np.max(beta_effective[elements_a[:, None], elements_b])
        c_hat = compute_extended_cost(cost_matrix, part_X, part_Y, gen, cell_a, cell_b)
        
        if c_hat - beta_hat < alpha_hat_prime:
            if gen == 0:
                x, y = elements_a[0], elements_b[0]
                if not neighborhood_mask[x, y]:
                    neighborhood_mask[x, y] = True  
                    mask_extended = True
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
            
    return mask_extended

def ot_auction_round_core(mu_X, mu_Y, cost_matrix, neighborhood_mask, coupling, 
                          beta_pairs, beta_unassigned, epsilon):
    num_X, num_Y = cost_matrix.shape
    rem_mu_X = mu_X - np.sum(coupling, axis=1)
    total_assigned_Y = np.sum(coupling, axis=0)
    rem_mu_Y = mu_Y - total_assigned_Y
    
    bids_received = {y: [] for y in range(num_Y)}
    alpha_prime_tracker = np.zeros(num_X)
    
    bidders = np.where(rem_mu_X > 0)[0]
    
    for x in bidders:
        if rem_mu_X[x] <= 0:
            continue
        x_neighbors = np.where(neighborhood_mask[x, :])[0]
        if len(x_neighbors) == 0:
            continue
        
        pi_list = []
        for y in x_neighbors:
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

def solve_single_scale_auction(mu_X, mu_Y, cost_matrix, initial_mask, part_X, part_Y, theta=4.0):
    """
    Löst das OT-Problem auf einer dedizierten Skala vollständig. Falls nach einer 
    erfolgreichen Kopplung die Konsistenzprüfung fehlschlägt, wird die Maske erweitert 
    und das Problem auf dieser Skala sauber neu aufgerollt.
    """
    num_X, num_Y = cost_matrix.shape
    neighborhood_mask = initial_mask.copy()
    
    while True:
        beta_pairs = np.zeros((num_X, num_Y), dtype=np.float64)
        beta_unassigned = np.zeros(num_Y, dtype=np.float64)
        coupling = np.zeros((num_X, num_Y), dtype=np.int32)
        
        valid_costs = cost_matrix[neighborhood_mask]
        C = np.max(valid_costs) - np.min(valid_costs) if len(valid_costs) > 0 else 1.0
        epsilon = C / 4.0 if C != 0 else 1.0
        target_epsilon = (1.0 / num_X) - 1e-5
        
        alpha_prime_tracker = np.zeros(num_X)
        while epsilon > target_epsilon:
            total_mass_X = np.sum(mu_X)
            while np.sum(coupling) < total_mass_X:
                coupling, beta_pairs, beta_unassigned, alpha_prime_tracker = ot_auction_round_core(
                    mu_X, mu_Y, cost_matrix, neighborhood_mask, coupling, beta_pairs, beta_unassigned, epsilon
                )
            epsilon /= theta
            
        mask_extended = hierarchical_consistency_check_ot(
            cost_matrix, neighborhood_mask, alpha_prime_tracker, coupling, beta_pairs, beta_unassigned, mu_Y, part_X, part_Y
        )
        
        if not mask_extended:
            return coupling, neighborhood_mask

def hybrid_optimal_transport_auction(mu_X, mu_Y, cost_matrix, theta=4.0):
    """
    Das echte Multiscale-Verfahren (Sect. 4, SS13): Startet grob, löst vollständig, 
    projiziert den Support nach unten und verfeinert sukzessive.
    """
    num_X, num_Y = cost_matrix.shape
    part_X = HierarchicalPartition(num_X)
    part_Y = HierarchicalPartition(num_Y)
    
    coarsest_gen = part_X.depth - 1
    
    current_sparse_mask = np.ones((num_X, num_Y), dtype=bool)
    
    for gen in range(coarsest_gen, -1, -1):
        if gen > 0:
            cells_X = part_X.hierarchy[gen]
            cells_Y = part_Y.hierarchy[gen]
            unique_cells_X = np.unique(cells_X)
            unique_cells_Y = np.unique(cells_Y)
            
            mu_X_coarse = np.array([np.sum(mu_X[cells_X == c]) for c in unique_cells_X], dtype=np.int32)
            mu_Y_coarse = np.array([np.sum(mu_Y[cells_Y == c]) for c in unique_cells_Y], dtype=np.int32)
            
            cost_coarse = np.zeros((len(unique_cells_X), len(unique_cells_Y)))
            for i, ca in enumerate(unique_cells_X):
                for j, cb in enumerate(unique_cells_Y):
                    cost_coarse[i, j] = compute_extended_cost(cost_matrix, part_X, part_Y, gen, ca, cb)
            
            mask_coarse = np.zeros_like(cost_coarse, dtype=bool)
            for i, ca in enumerate(unique_cells_X):
                for j, cb in enumerate(unique_cells_Y):
                    elements_a = part_X.get_elements_in_cell(gen, ca)
                    elements_b = part_Y.get_elements_in_cell(gen, cb)
                    mask_coarse[i, j] = np.any(current_sparse_mask[elements_a[:, None], elements_b])
            
            part_X_coarse = HierarchicalPartition(len(unique_cells_X))
            part_Y_coarse = HierarchicalPartition(len(unique_cells_Y))
            coupling_coarse, _ = solve_single_scale_auction(
                mu_X_coarse, mu_Y_coarse, cost_coarse, mask_coarse, part_X_coarse, part_Y_coarse, theta
            )
            
            current_sparse_mask = np.zeros((num_X, num_Y), dtype=bool)
            active_coarse_pairs = np.argwhere(coupling_coarse > 0)
            for ca_idx, cb_idx in active_coarse_pairs:
                ca = unique_cells_X[ca_idx]
                cb = unique_cells_Y[cb_idx]
                elements_a = part_X.get_elements_in_cell(gen, ca)
                elements_b = part_Y.get_elements_in_cell(gen, cb)
                current_sparse_mask[elements_a[:, None], elements_b] = True
        else:
            final_coupling, _ = solve_single_scale_auction(
                mu_X, mu_Y, cost_matrix, current_sparse_mask, part_X, part_Y, theta
            )
            return final_coupling


def strict_optimal_transport_auction(mu_X, mu_Y, cost_matrix, theta=4.0):
    """ Standard-Referenzlöser auf nativer Ebene ohne Multiscale-Struktur """
    num_X, num_Y = cost_matrix.shape
    full_mask = np.ones((num_X, num_Y), dtype=bool)
    part_X = HierarchicalPartition(num_X)
    part_Y = HierarchicalPartition(num_Y)
    coupling, _ = solve_single_scale_auction(mu_X, mu_Y, cost_matrix, full_mask, part_X, part_Y, theta)
    return coupling

if __name__ == "__main__":
    N = 128  
    print(f"Setting up high-scale scenario. N = {N} ({N*N} variations)...")
    
    np.random.seed(42)
    mu_X = np.random.randint(5, 15, size=N, dtype=np.int32)
    mu_Y = np.random.randint(5, 15, size=N, dtype=np.int32)
    
    diff = np.sum(mu_X) - np.sum(mu_Y)
    if diff > 0: mu_Y[0] += diff
    else: mu_X[0] -= diff

    x_coords = np.random.uniform(0, 10, size=(N, 2))
    y_coords = np.random.uniform(0, 10, size=(N, 2))
    
    print("Computing cost matrix...")
    cost_matrix = np.sum((x_coords[:, None, :] - y_coords[None, :, :])**2, axis=-1)

    print("\n[POT] Running highly optimized C++ solver backend...")
    a, b = mu_X.astype(np.float64) / mu_X.sum(), mu_Y.astype(np.float64) / mu_Y.sum()

    start_pot = time.perf_counter()
    pot_plan = ot.emd(a, b, cost_matrix)
    end_pot = time.perf_counter()

    print(f"--> POT completed in {end_pot - start_pot:.5f} seconds!")

    print("\n[Standard] Running Dense Solver Algorithm...")
    start_std = time.perf_counter()
    coupling_std = strict_optimal_transport_auction(mu_X, mu_Y, cost_matrix)
    end_std = time.perf_counter()
    time_std = end_std - start_std
    print(f"--> Standard completed in {time_std:.5f} seconds.")

    print("\n[Hybrid] Running Pure Sparse Hierarchical Multiscale Algorithm...")
    start_hyb = time.perf_counter()
    coupling_hyb = hybrid_optimal_transport_auction(mu_X, mu_Y, cost_matrix)
    end_hyb = time.perf_counter()
    time_hyb = end_hyb - start_hyb
    print(f"--> Hybrid completed in {time_hyb:.5f} seconds.")

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
