# Quick test to verify no deadlocking
import numpy as np
from src.core.ot_auction import AuctionOT

N = 16
C = np.random.rand(N, N)
mu_X = np.random.rand(N) * 2 + 1  # Random supply 1-3
mu_Y = np.random.rand(N) * 2 + 1  # Random demand 1-3

print(f"Total supply: {np.sum(mu_X):.2f}")
print(f"Total demand: {np.sum(mu_Y):.2f}")
print(f"Imbalance: {abs(np.sum(mu_X) - np.sum(mu_Y)):.2f}")

solver = AuctionOT(C, mu_X, mu_Y, epsilon=0.01)
mu, cost, iters = solver.solve()

assigned_X = np.sum(mu, axis=1)
assigned_Y = np.sum(mu, axis=0)

print(f"\nIterations: {iters}")
print(f"Cost: {cost:.6f}")
print(f"Supply satisfied: {np.allclose(assigned_X, mu_X)}")
print(f"Demand satisfied: {np.allclose(assigned_Y, mu_Y)}")
print(f"Max supply deficit: {np.max(np.abs(assigned_X - mu_X)):.2e}")
print(f"Max demand deficit: {np.max(np.abs(assigned_Y - mu_Y)):.2e}")
print("✓ Test passed!" if iters < 50000 else "✗ Deadlocked")
