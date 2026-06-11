import numpy as np
import ot

print("Section 2 - LAP and OT")
from s2_lap_ot import solve_lap, solve_ot

cost_lap = np.array([[4.,1.,3.],[2.,0.,5.],[3.,2.,2.]])
assignment, total = solve_lap(cost_lap)
print("LAP:", assignment, "cost:", total)

positions = np.array([0.,1.,2.,3.])
cost_ot   = (positions[:,None] - positions[None,:])**2
coupling, total_ot = solve_ot(cost_ot, [0.5,0.3,0.1,0.1], [0.1,0.1,0.3,0.5])
print("OT cost:", round(total_ot, 4))


print("\nSection 3 - Auction Algorithm")
from s3_auction import auction_lap

assignment, total, beta = auction_lap(cost_lap)
print("Auction:", assignment, "cost:", total)


print("\nSection 4 - Hierarchical Multiscale OT")
from s4_hierachical import solve_ot_multiscale

rng  = np.random.default_rng(42)
N    = 40
px   = rng.random((N, 2))
py   = rng.random((N, 2))
mu   = np.ones(N) / N

coup, total, N_hat = solve_ot_multiscale(px, py, mu, mu, verbose=True)
cost_full = np.sum((px[:,None]-py[None,:])**2, axis=2)
cost_dense = float(np.sum(cost_full * ot.emd(mu, mu, cost_full)))
print(f"Dense cost: {cost_dense:.4f}  |  Hierarchical cost: {total:.4f}")
print(f"Pairs used: {len(N_hat)} / {N*N}  ({100*len(N_hat)/N**2:.1f}%)")
