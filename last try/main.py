from models import Point, OTSolverState, QuadTree, Node
from solver import HierarchicalOptimalTransportSolver

# 1. Setup Data
points_X = [Point(0, 0.1, 0.1), Point(1, 0.9, 0.9)]
points_Y = [Point(0, 0.2, 0.2), Point(1, 0.8, 0.8)]

# 2. Build Trees & State
qt_X = QuadTree(points_X)
qt_Y = QuadTree(points_Y)
state = OTSolverState(
    X={p.id: Node(p.id, 1, 0) for p in points_X},
    Y={p.id: Node(p.id, 1, 0) for p in points_Y}
)

# 3. Solve
def cost_func(p1, p2): return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
solver = HierarchicalOptimalTransportSolver(qt_X, qt_Y, initial_epsilon=1.0)
solver.solve(state, cost_func)