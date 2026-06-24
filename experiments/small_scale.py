import time
import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.cost_functions import squared_euclidean
from src.utils.eps_scaling import EpsScalingManager
from src.core.ot_auction import AuctionOT

class SilencePrints:
    def __enter__(self):
        import sys
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys
        sys.stdout.close()
        sys.stdout = self._original_stdout

def run_lap_test(N=16):
    print("\n--- Running LAP Test (N=16) ---")
    np.random.seed(50)
    X = np.random.rand(N, 2)
    Y = np.random.rand(N, 2)
    C = squared_euclidean(X, Y)
    
    mu_X = np.ones(N, dtype=int)
    mu_Y = np.ones(N, dtype=int)
    
    t0 = time.perf_counter()
    with SilencePrints():
        manager = EpsScalingManager(AuctionOT, C, mu_X=mu_X, mu_Y=mu_Y)
        assignments, total_cost, iters, _ = manager.solve()
    duration = time.perf_counter() - t0
    
    print(f"LAP solved in {duration:.4f} seconds.")
    print(f"Total Iterations: {iters}")
    print(f"Total Cost: {total_cost:.4f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(C, cmap='viridis')
    ax1.set_title(f"Cost Matrix (N={N})")
    ax1.set_xlabel("Points in Y")
    ax1.set_ylabel("Points in X")
    fig.colorbar(im1, ax=ax1)
    
    # Perfekter Kontrast: Da Masse = 1, setzen wir vmax auf 1
    im2 = ax2.imshow(assignments, cmap='Blues', vmin=0, vmax=1)
    ax2.set_title("LAP Final Assignments")
    ax2.set_xlabel("Points in Y")
    ax2.set_ylabel("Points in X")
    fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

def run_ot_test(N=16):
    print("\n--- Running OT Test (N=16) ---")
    np.random.seed(50)
    X = np.random.rand(N, 2)
    Y = np.random.rand(N, 2)
    C = squared_euclidean(X, Y)
    
    mu_X = np.random.randint(1, 5, size=N)
    mu_Y = np.random.randint(1, 5, size=N)
    diff = np.sum(mu_X) - np.sum(mu_Y)
    if diff > 0: mu_Y[0] += diff
    elif diff < 0: mu_X[0] += abs(diff)
    
    t0 = time.perf_counter()
    with SilencePrints():
        manager = EpsScalingManager(AuctionOT, C, mu_X=mu_X, mu_Y=mu_Y)
        coupling_matrix, total_cost, iters, _ = manager.solve()
    duration = time.perf_counter() - t0
    
    print(f"OT solved in {duration:.4f} seconds.")
    print(f"Total Iterations: {iters}")
    print(f"Total Cost: {total_cost:.4f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(C, cmap='viridis')
    ax1.set_title(f"Cost Matrix (N={N})")
    ax1.set_xlabel("Points in Y")
    ax1.set_ylabel("Points in X")
    fig.colorbar(im1, ax=ax1)
    
    # scaling to the max of matrix
    im2 = ax2.imshow(coupling_matrix, cmap='Oranges', vmin=0, vmax=np.max(coupling_matrix))
    ax2.set_title("OT Final Coupling Matrix")
    ax2.set_xlabel("Points in Y")
    ax2.set_ylabel("Points in X")
    fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_lap_test(N=16)
    run_ot_test(N=16)
