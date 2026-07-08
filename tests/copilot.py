# Quick sanity check
import numpy as np
from src.core.ot_auction import AuctionOT

# Small test
N = 8
C = np.random.rand(N, N)
mu_X = np.ones(N)
mu_Y = np.ones(N)

solver = AuctionOT(C, mu_X, mu_Y, epsilon=0.1)
mu, cost, iters = solver.solve()

print(f"Final cost: {cost:.6f}")
print(f"Assigned X: {np.sum(mu, axis=1)}")  # Should be all 1.0
print(f"Assigned Y: {np.sum(mu, axis=0)}")  # Should be all 1.0
print(f"Iterations: {iters}")
assert np.allclose(np.sum(mu, axis=1), mu_X), "Supply not satisfied"
assert np.allclose(np.sum(mu, axis=0), mu_Y), "Demand not satisfied"
print("✓ Test passed!")
