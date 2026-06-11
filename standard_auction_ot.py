from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class Node:
    id: int
    total_mass: int
    # Tracks how much mass is currently assigned to/from this node
    assigned_mass: int = 0 
    
    @property
    def unassigned_mass(self) -> int:
        return self.total_mass - self.assigned_mass

class OTSolverState:
    def __init__(self, X_nodes: Dict[int, Node], Y_nodes: Dict[int, Node], cost_matrix: Dict[Tuple[int, int], float]):
        self.X = X_nodes
        self.Y = Y_nodes
        self.c = cost_matrix
        
        # \mu(x, y): Tracks the actual mass assigned between x and y
        self.mu: Dict[Tuple[int, int], int] = {}
        
        # \beta(x, y): Dual variable for specific assigned pairs
        self.beta_xy: Dict[Tuple[int, int], float] = {}
        
        # \beta(\oslash, y): Dual variable for unassigned mass atoms at y
        self.beta_empty: Dict[int, float] = {y: 0.0 for y in self.Y}

    def get_mu(self, x: int, y: int) -> int:
        return self.mu.get((x, y), 0)



@dataclass
class PiEntry:
    target_y: int
    current_owner_x: int  # None represents the empty \oslash state
    net_cost: float       # c(x, y) - \beta(x', y)  OR  c(x,y) - \beta(\oslash, y)
    available_mass: int   # How much mass can be taken from this specific entry

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
        return []
        
    pi_x = build_pi_x(state, x)
    if not pi_x:
        return [] 
        
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
