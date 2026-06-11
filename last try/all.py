#Step 1

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


#Step 2

from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable

@dataclass
class Point:
    id: int
    x: float
    y: float

class QuadNode:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        self.bounds = (x_min, y_min, x_max, y_max)
        self.points: List[Point] = []
        self.children: List['QuadNode'] = []
        
        # Extended dual variables (\hat{\alpha} and \hat{\beta})
        self.alpha_hat: float = float('-inf')
        self.beta_hat: float = float('-inf')

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

class QuadTree:
    def __init__(self, points: List[Point], max_points_per_cell: int = 1):
        self.max_points = max_points_per_cell
        
        if not points:
            self.root = None
            return
            
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        
        # Add a tiny epsilon to max bounds to ensure edge points are strictly strictly inside
        self.root = QuadNode(
            min(x_coords), min(y_coords), 
            max(x_coords) + 1e-9, max(y_coords) + 1e-9
        )
        
        for p in points:
            self._insert(self.root, p)

    def _insert(self, node: QuadNode, point: Point):
        # If it's a leaf and has room, or if it can't be subdivided further (identical points)
        if node.is_leaf and len(node.points) < self.max_points:
            node.points.append(point)
            return
            
        # If it's a leaf but full, we must subdivide it
        if node.is_leaf:
            self._subdivide(node)
            
        # Route the point to the correct child
        for child in node.children:
            x_min, y_min, x_max, y_max = child.bounds
            if x_min <= point.x < x_max and y_min <= point.y < y_max:
                self._insert(child, point)
                return

    def _subdivide(self, node: QuadNode):
        x_min, y_min, x_max, y_max = node.bounds
        mid_x = (x_min + x_max) / 2.0
        mid_y = (y_min + y_max) / 2.0
        
        node.children = [
            QuadNode(x_min, y_min, mid_x, mid_y), # Bottom-Left
            QuadNode(mid_x, y_min, x_max, mid_y), # Bottom-Right
            QuadNode(x_min, mid_y, mid_x, y_max), # Top-Left
            QuadNode(mid_x, mid_y, x_max, y_max)  # Top-Right
        ]
        
        # Re-distribute existing points to children
        existing_points = node.points
        node.points = []
        for p in existing_points:
            self._insert(node, p)

    def update_alpha_hat(self, node: QuadNode, alpha_dict: Dict[int, float]) -> float:
        """
        Recursively computes \hat{\alpha}(a) for the tree (Equation 13).
        Should be called on the root node whenever \alpha values change.
        """
        if node.is_leaf:
            if not node.points:
                node.alpha_hat = float('-inf')
            else:
                node.alpha_hat = max(alpha_dict.get(p.id, float('-inf')) for p in node.points)
        else:
            node.alpha_hat = max(self.update_alpha_hat(child, alpha_dict) for child in node.children)
            
        return node.alpha_hat

    def update_beta_hat(self, node: QuadNode, beta_dict: Dict[int, float]) -> float:
        """
        Recursively computes \hat{\beta}(b) for the tree.
        """
        if node.is_leaf:
            if not node.points:
                node.beta_hat = float('-inf')
            else:
                node.beta_hat = max(beta_dict.get(p.id, float('-inf')) for p in node.points)
        else:
            node.beta_hat = max(self.update_beta_hat(child, beta_dict) for child in node.children)
            
        return node.beta_hat

def compute_c_hat(node_a: QuadNode, node_b: QuadNode, c_func: Callable[[Point, Point], float]) -> float:
    """
    Computes \hat{c}(a,b) as the exact minimum cost between all points in node_a and node_b (Equation 14).
    Note: For specific metrics like squared Euclidean distance, this can be heavily optimized
    by calculating the minimum distance between the bounding boxes instead of point-to-point.
    """
    # Extract all points recursively from both nodes
    def get_all_points(node: QuadNode) -> List[Point]:
        if node.is_leaf:
            return node.points
        points = []
        for child in node.children:
            points.extend(get_all_points(child))
        return points

    points_a = get_all_points(node_a)
    points_b = get_all_points(node_b)
    
    if not points_a or not points_b:
        return float('inf')

    # Explicit min_{x \in a, y \in b} c(x,y)
    return min(c_func(pa, pb) for pa in points_a for pb in points_b)


#Step 3

from typing import Dict, Set, Callable

def get_exact_beta_y(state: OTSolverState, y_id: int) -> float:
    """
    Computes \beta(y) for a specific node y.
    In the generalized OT algorithm, \beta is the maximum of the unassigned 
    \beta(\oslash, y) and any \beta(x', y) for mass currently assigned to y.
    """
    max_beta = state.beta_empty.get(y_id, float('-inf'))
    
    for (x_prime_id, dest_y_id), assigned_mass in state.mu.items():
        if dest_y_id == y_id and assigned_mass > 0:
            beta_xy = state.beta_xy.get((x_prime_id, y_id), float('-inf'))
            if beta_xy > max_beta:
                max_beta = beta_xy
                
    return max_beta

def get_node_max_alpha_prime(node: QuadNode, alpha_prime_dict: Dict[int, float]) -> float:
    """Helper to dynamically compute \hat{\alpha}'(a) for a given partition cell."""
    if node.is_leaf:
        if not node.points:
            return float('-inf')
        return max(alpha_prime_dict.get(p.id, float('-inf')) for p in node.points)
    return max(get_node_max_alpha_prime(child, alpha_prime_dict) for child in node.children)


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


#Step 4

def run_hybrid_auction_step(state, quadtree_X, quadtree_Y, epsilon, c_func):
    # The while loop naturally handles the Mass-Left Leak: 
    # Any node with unassigned_mass > 0 will re-enter determine_m_and_bids
    # in the next iteration. Because the state.mu/state.beta have changed,
    # build_pi_x will re-sort the landscape, naturally updating \alpha'.
    while any(x.unassigned_mass > 0 for x in state.X.values()):
        
        # 1. Bidding Phase (Restricted to \hat{\mathcal{N}})
        all_bids = []
        alpha_prime_dict = {}
        for x in [x for x in state.X.values() if x.unassigned_mass > 0]:
            bids, alpha_prime_x = determine_m_and_bids(state, x, epsilon)
            if bids:
                all_bids.extend(bids)
                alpha_prime_dict[x.id] = alpha_prime_x

        # 2. Consistency Check
        beta_dict = {y_id: get_exact_beta_y(state, y_id) for y_id in state.Y}
        quadtree_Y.update_beta_hat(quadtree_Y.root, beta_dict)
        
        nodes_to_rebid = set()
        consistency_check(quadtree_X.root, quadtree_Y.root, state, 
                          alpha_prime_dict, c_func, nodes_to_rebid)

        # 3. Rebidding
        if nodes_to_rebid:
            all_bids = [b for b in all_bids if b['source_x'] not in nodes_to_rebid]
            for x_id in nodes_to_rebid:
                new_bids, _ = determine_m_and_bids(state, state.X[x_id], epsilon)
                all_bids.extend(new_bids)

        # 4. Assignment Phase
        bids_by_y = {y_id: [] for y_id in state.Y}
        for bid in all_bids:
            bids_by_y[bid['target_y']].append(bid)

        for y_id, bids in bids_by_y.items():
            bids.sort(key=lambda b: b['bid_value'])
            
            for bid in bids:
                # Re-check current mass availability at every step to prevent 
                # over-assignment due to partial fulfillment
                source_x = state.X[bid['source_x']]
                target_owner_id = bid['targeted_owner_x']
                
                # Scenario A: Target Unassigned (\oslash)
                if target_owner_id is None:
                    take = min(bid['mass'], source_x.unassigned_mass, state.Y[y_id].unassigned_mass)
                    if take > 0:
                        state.mu[(source_x.id, y_id)] = state.mu.get((source_x.id, y_id), 0) + take
                        source_x.assigned_mass += take
                        state.Y[y_id].assigned_mass += take
                        state.beta_empty[y_id] = bid['bid_value']
                        state.beta_xy[(source_x.id, y_id)] = bid['bid_value']
                
                # Scenario B: Target Existing Owner
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


#Step 5

class HierarchicalOptimalTransportSolver:
    def __init__(self, quadtree_X: QuadTree, quadtree_Y: QuadTree, 
                 initial_epsilon: float, epsilon_factor: float = 0.5):
        self.qt_X = quadtree_X
        self.qt_Y = quadtree_Y
        self.eps = initial_epsilon
        self.eps_factor = epsilon_factor
        self.active_N: Set[Tuple[int, int]] = set() # Explicitly tracked neighborhood

    def solve(self, state: OTSolverState, c_func: Callable):
        generations = self._get_generations_ordered()

        # Initialization: At the coarsest scale, we must start with 
        # a sufficiently large subset or the full domain.
        self.active_N = self._get_full_domain_neighborhood()

        for gen_idx in range(len(generations)):
            # The epsilon-scaling loop is used at each generation[cite: 256].
            # Resetting variables (like dual variables) between epsilon scales
            # is critical for convergence[cite: 255].
            self._epsilon_scaling_loop(state, c_func)
            
            if gen_idx < len(generations) - 1:
                self._project_support_to_next_scale(state)

    def _project_support_to_next_scale(self, state: OTSolverState):
        """
        Populates \hat{\mathcal{N}} with pairs corresponding to active support.
        """
        # Explicitly define \hat{\mathcal{N}} by taking the support of \mu 
        # from the current scale to initialize the next[cite: 226].
        self.active_N = set(state.mu.keys())
        
        # In a production implementation, the bidding phase (build_pi_x)
        # would be updated to check against this self.active_N set.
