from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable

@dataclass
class Point:
    id: int
    x: float
    y: float

@dataclass
class Node: 
    id: int
    total_mass: int
    assigned_mass: int = 0
    
    @property
    def unassigned_mass(self) -> int:
        return self.total_mass - self.assigned_mass

class OTSolverState:
    def __init__(self, X: Dict[int, Node], Y: Dict[int, Node]):
        self.X = X
        self.Y = Y
        self.mu: Dict[Tuple[int, int], int] = {}
        self.beta_empty: Dict[int, float] = {y_id: 0.0 for y_id in Y}
        self.beta_xy: Dict[Tuple[int, int], float] = {}
        self.c: Dict[Tuple[int, int], float] = {}

class QuadNode:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        self.bounds = (x_min, y_min, x_max, y_max)
        self.points: List[Point] = []
        self.children: List['QuadNode'] = []
        self.alpha_hat: float = float('-inf')
        self.beta_hat: float = float('-inf')
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_all_points(self) -> List[Point]:
        """Recursive helper to gather all points in this node and its children."""
        if self.is_leaf:
            return self.points
        points = []
        for child in self.children:
            points.extend(child.get_all_points())
        return points

class QuadTree:
    # ... (Your existing QuadTree class remains exactly as you wrote it) ...
    def __init__(self, points: List[Point], max_points_per_cell: int = 1):
        self.max_points = max_points_per_cell
        if not points:
            self.root = None
            return
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        self.root = QuadNode(min(x_coords), min(y_coords), max(x_coords) + 1e-9, max(y_coords) + 1e-9)
        for p in points:
            self._insert(self.root, p)
    
    def _insert(self, node: QuadNode, point: Point):
        if node.is_leaf and len(node.points) < self.max_points:
            node.points.append(point)
            return
        if node.is_leaf: self._subdivide(node)
        for child in node.children:
            x_min, y_min, x_max, y_max = child.bounds
            if x_min <= point.x < x_max and y_min <= point.y < y_max:
                self._insert(child, point)
                return

    def _subdivide(self, node: QuadNode):
        x_min, y_min, x_max, y_max = node.bounds
        mid_x, mid_y = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        node.children = [QuadNode(x_min, y_min, mid_x, mid_y), QuadNode(mid_x, y_min, x_max, mid_y),
                         QuadNode(x_min, mid_y, mid_x, y_max), QuadNode(mid_x, mid_y, x_max, y_max)]
        for p in node.points: self._insert(node, p)
        node.points = []

    def update_alpha_hat(self, node: QuadNode, alpha_dict: Dict[int, float]) -> float:
        if node.is_leaf:
            node.alpha_hat = max((alpha_dict.get(p.id, float('-inf')) for p in node.points), default=float('-inf'))
        else:
            node.alpha_hat = max(self.update_alpha_hat(child, alpha_dict) for child in node.children)
        return node.alpha_hat

    def update_beta_hat(self, node: QuadNode, beta_dict: Dict[int, float]) -> float:
        if node.is_leaf:
            node.beta_hat = max((beta_dict.get(p.id, float('-inf')) for p in node.points), default=float('-inf'))
        else:
            node.beta_hat = max(self.update_beta_hat(child, beta_dict) for child in node.children)
        return node.beta_hat