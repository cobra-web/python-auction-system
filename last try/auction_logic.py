# auction_logic.py

from typing import Dict, Set, Callable, List
from dataclasses import dataclass
from models import OTSolverState, Node, Point, QuadNode

# ... now your functions will recognize 'List' ..

from typing import Dict, Set, Callable, List
from dataclasses import dataclass # Don't forget to import this!
from models import OTSolverState, Node, Point

@dataclass
class PiEntry:
    target_y: int
    current_owner_x: int  
    net_cost: float       
    available_mass: int 

# Now the rest of your functions (build_pi_x, etc.) will work.

def build_pi_x(state: OTSolverState, x: Node) -> List[PiEntry]:
    pi_x = []
    
    for y_id, y_node in state.Y.items():
        # Only consider neighbors where a cost exists (c(x, y) < infinity)
        if (x.id, y_id) not in state.c:
            continue
            
        c_xy = state.c[(x.id, y_id)]
        
        # Part 1: Consider unassigned mass at y (the \oslash condition)
        if y_node.unassigned_mass > 0:
            net_cost = c_xy - state.beta_empty[y_id]
            pi_x.append(PiEntry(
                target_y=y_id, 
                current_owner_x=None, 
                net_cost=net_cost, 
                available_mass=y_node.unassigned_mass
            ))
            
        # Part 2: Consider mass at y currently owned by other nodes x'
        # We look for x' != x where \mu(x', y) > 0
        for (x_prime_id, dest_y_id), assigned_mass in state.mu.items():
            if dest_y_id == y_id and x_prime_id != x.id and assigned_mass > 0:
                beta_x_prime_y = state.beta_xy.get((x_prime_id, y_id), 0.0)
                net_cost = c_xy - beta_x_prime_y
                
                pi_x.append(PiEntry(
                    target_y=y_id,
                    current_owner_x=x_prime_id,
                    net_cost=net_cost,
                    available_mass=assigned_mass
                ))
                
    # Sort ascending by net_cost as per Equation 11
    pi_x.sort(key=lambda entry: entry.net_cost)
    return pi_x

def determine_m_and_bids(state: OTSolverState, x: Node, epsilon: float):
    mass_to_assign = int(x.unassigned_mass)
    if mass_to_assign <= 0:
        return [], float('inf') # Was return []
    
    pi_x = build_pi_x(state, x)
    if not pi_x:
        return [], float('inf') # Was return []
        
    accumulated_mass = 0
    m_index = 0
    
    for i, entry in enumerate(pi_x):
        accumulated_mass += int(entry.available_mass)
        if accumulated_mass >= mass_to_assign:
            m_index = i
            break
            
    if accumulated_mass < mass_to_assign:
        alpha_prime_x = float('inf')
    else:
        m_index_for_alpha = m_index + 1 if (m_index + 1) < len(pi_x) else m_index
        alpha_prime_x = pi_x[m_index_for_alpha].net_cost
    
    bids = []
    mass_left_to_bid = mass_to_assign
    
    bound_index = m_index + 1 if accumulated_mass >= mass_to_assign else len(pi_x)
    for i in range(bound_index):
        entry = pi_x[i]
        mass_for_this_bid = min(mass_left_to_bid, int(entry.available_mass))
        
        bid_value = float('-inf') if alpha_prime_x == float('inf') else (state.c[(x.id, entry.target_y)] - alpha_prime_x - epsilon)
        
        bids.append({
            'source_x': x.id,
            'target_y': entry.target_y,
            'targeted_owner_x': entry.current_owner_x,
            'mass': mass_for_this_bid,
            'bid_value': bid_value
        })
        
        mass_left_to_bid -= mass_for_this_bid
        if mass_left_to_bid <= 0:
            break
            
    return bids


# Include build_pi_x helper
# Include get_exact_beta_y (Step 3)

def get_exact_beta_y(state: OTSolverState, y_id: int) -> float:
    max_beta = state.beta_empty.get(y_id, float('-inf'))
    
    for (x_prime_id, dest_y_id), assigned_mass in state.mu.items():
        if dest_y_id == y_id and assigned_mass > 0:
            beta_xy = state.beta_xy.get((x_prime_id, y_id), float('-inf'))
            if beta_xy > max_beta:
                max_beta = beta_xy
                
    return max_beta


# Include get_node_max_alpha_prime (Step 3)

def get_node_max_alpha_prime(node: QuadNode, alpha_prime_dict: Dict[int, float]) -> float:
    if node.is_leaf:
        if not node.points:
            return float('-inf')
        return max(alpha_prime_dict.get(p.id, float('-inf')) for p in node.points)
    return max(get_node_max_alpha_prime(child, alpha_prime_dict) for child in node.children)

def compute_c_hat(node_a, node_b, c_func):
    # 1. Internal helper to recursively collect all leaf points
    def get_all_points(node):
        if node.is_leaf:
            return node.points
        points = []
        for child in node.children:
            points.extend(get_all_points(child))
        return points

    # 2. Get the actual point lists from the hierarchies
    points_a = get_all_points(node_a)
    points_b = get_all_points(node_b)

    # 3. Guard against empty nodes
    if not points_a or not points_b:
        return float('inf')

    # 4. Find the absolute minimum cost between any p_a and p_b
    # This is the 'hat' cost (the lower bound)
    min_cost = min(c_func(pa, pb) for pa in points_a for pb in points_b)
    
    return min_cost

def consistency_check(
    node_a: QuadNode, 
    node_b: QuadNode, 
    state: OTSolverState, 
    alpha_prime_dict: Dict[int, float], 
    c_func: Callable[[Point, Point], float], 
    nodes_to_rebid: Set[int]
):
    if compute_c_hat(node_a, node_b, c_func) - node_b.beta_hat >= get_node_max_alpha_prime(node_a, alpha_prime_dict):
        return 
        
    if node_a.is_leaf and node_b.is_leaf:
        for p_x in node_a.points:
            alpha_prime_x = alpha_prime_dict.get(p_x.id, float('-inf'))
            for p_y in node_b.points:
                exact_c = c_func(p_x, p_y)
                if exact_c - get_exact_beta_y(state, p_y.id) < alpha_prime_x:
                    if (p_x.id, p_y.id) not in state.c:
                        state.c[(p_x.id, p_y.id)] = exact_c
                        nodes_to_rebid.add(p_x.id)
    else:
        for child_a in (node_a.children if not node_a.is_leaf else [node_a]):
            for child_b in (node_b.children if not node_b.is_leaf else [node_b]):
                consistency_check(child_a, child_b, state, alpha_prime_dict, c_func, nodes_to_rebid)
# Include run_hybrid_auction_step (Step 4)

def run_hybrid_auction_step(state, quadtree_X, quadtree_Y, epsilon, c_func):
    while any(x.unassigned_mass > 0 for x in state.X.values()):
        
        all_bids = []
        alpha_prime_dict = {}
        for x in [x for x in state.X.values() if x.unassigned_mass > 0]:
            bids, alpha_prime_x = determine_m_and_bids(state, x, epsilon)
            if bids:
                all_bids.extend(bids)
                alpha_prime_dict[x.id] = alpha_prime_x

        beta_dict = {y_id: get_exact_beta_y(state, y_id) for y_id in state.Y}
        quadtree_Y.update_beta_hat(quadtree_Y.root, beta_dict)
        
        nodes_to_rebid = set()
        consistency_check(quadtree_X.root, quadtree_Y.root, state, 
                          alpha_prime_dict, c_func, nodes_to_rebid)

        if nodes_to_rebid:
            all_bids = [b for b in all_bids if b['source_x'] not in nodes_to_rebid]
            for x_id in nodes_to_rebid:
                new_bids, _ = determine_m_and_bids(state, state.X[x_id], epsilon)
                all_bids.extend(new_bids)

        bids_by_y = {y_id: [] for y_id in state.Y}
        for bid in all_bids:
            bids_by_y[bid['target_y']].append(bid)

        for y_id, bids in bids_by_y.items():
            bids.sort(key=lambda b: b['bid_value'])
            
            for bid in bids:
                source_x = state.X[bid['source_x']]
                target_owner_id = bid['targeted_owner_x']
                
                if target_owner_id is None:
                    take = min(bid['mass'], source_x.unassigned_mass, state.Y[y_id].unassigned_mass)
                    if take > 0:
                        state.mu[(source_x.id, y_id)] = state.mu.get((source_x.id, y_id), 0) + take
                        source_x.assigned_mass += take
                        state.Y[y_id].assigned_mass += take
                        state.beta_empty[y_id] = bid['bid_value']
                        state.beta_xy[(source_x.id, y_id)] = bid['bid_value']
                
                else:
                    target_owner = state.X[target_owner_id]
                    current_owned = state.mu.get((target_owner.id, y_id), 0)
                    take = min(bid['mass'], source_x.unassigned_mass, current_owned)
                    
                    if take > 0:
                        # Direct atomic mutation for consistency
                        state.mu[(target_owner.id, y_id)] -= take
                        target_owner.assigned_mass -= take
                        
                        state.mu[(source_x.id, y_id)] = state.mu.get((source_x.id, y_id), 0) + take
                        source_x.assigned_mass += take
                        
                        state.beta_xy[(source_x.id, y_id)] = bid['bid_value']
                        
                        if state.mu[(target_owner.id, y_id)] == 0:
                            del state.mu[(target_owner.id, y_id)]