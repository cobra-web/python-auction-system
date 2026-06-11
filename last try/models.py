from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

@dataclass
class Point:
    id: int
    x: float
    y: float

@dataclass
class Node: # Represents x in X or y in Y
    id: int
    unassigned_mass: int
    assigned_mass: int

class OTSolverState:
    def __init__(self, X: Dict[int, Node], Y: Dict[int, Node]):
        self.X = X
        self.Y = Y
        self.mu: Dict[Tuple[int, int], int] = {}
        self.beta_empty: Dict[int, float] = {y_id: 0.0 for y_id in Y}
        self.beta_xy: Dict[Tuple[int, int], float] = {}
        self.c: Dict[Tuple[int, int], float] = {} # Sparse neighborhood

class QuadNode:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.bounds = (x_min, y_min, x_max, y_max)
        self.points: List[Point] = []
        self.children: List['QuadNode'] = []
        self.alpha_hat: float = float('-inf')
        self.beta_hat: float = float('-inf')
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

# Include QuadTree class here (the full code from Step 2)