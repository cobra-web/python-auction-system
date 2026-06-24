import numpy as np
from src.core.lap_auction import AuctionLAP
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

N = 16
np.random.seed(42)

X_points = np.random.rand(N, 2)
Y_points = np.random.rand(N, 2)

C = distance_matrix(X_points, Y_points)

auction = AuctionLAP(C)
assignments, total_cost, iters = auction.solve()

assignment_matrix = np.zeros((N, N))
for x, y in enumerate(assignments):
    assignment_matrix[x, y] = 1

plt.figure(figsize=(6, 6))
plt.imshow(assignment_matrix, cmap='Blues', interpolation='nearest')
plt.title(f"Final Assignment Matrix (N={N})\nCost: {total_cost:.2f}, Iters: {iters}")
plt.xlabel("Points in Y")
plt.ylabel("Points in X")
plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
plt.xticks(np.arange(N))
plt.yticks(np.arange(N))
plt.show()
