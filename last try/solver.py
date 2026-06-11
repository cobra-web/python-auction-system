from typing import Callable, Set, Tuple, List
from models import OTSolverState, QuadTree, Point
from auction_logic import run_hybrid_auction_step

class HierarchicalOptimalTransportSolver:
    def __init__(self, quadtree_X: QuadTree, quadtree_Y: QuadTree, 
                 initial_epsilon: float, epsilon_factor: float = 0.5):
        self.qt_X = quadtree_X
        self.qt_Y = quadtree_Y
        self.eps = initial_epsilon
        self.eps_factor = epsilon_factor
        self.active_N: Set[Tuple[int, int]] = set()

    def solve(self, state: OTSolverState, c_func: Callable):
        generations = self._get_generations_ordered()
        self.active_N = self._get_full_domain_neighborhood()

        for gen_idx in range(len(generations)):
            self._epsilon_scaling_loop(state, c_func)
            if gen_idx < len(generations) - 1:
                self._project_support_to_next_scale(state)

    def _epsilon_scaling_loop(self, state, c_func):
        current_eps = self.eps
        while current_eps > 1e-6:
            run_hybrid_auction_step(state, self.qt_X, self.qt_Y, current_eps, c_func)
            current_eps *= self.eps_factor
            
    def _project_support_to_next_scale(self, state):
        self.active_N = set(state.mu.keys())

    # THESE ARE NOW CORRECTLY INDENTED INSIDE THE CLASS
    def _get_generations_ordered(self) -> List[int]:
        g = 3 
        return list(range(g, -1, -1))

    def _get_full_domain_neighborhood(self) -> Set[Tuple[int, int]]:
        return {(x.id, y.id) for x in self.qt_X.root.get_all_points() 
                           for y in self.qt_Y.root.get_all_points()}