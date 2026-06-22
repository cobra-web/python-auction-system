import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Import your newly built modules
from src.core.lap_auction import AuctionLAP
from src.core.ot_auction import AuctionOT
from src.utils.eps_scaling import EpsScalingManager

def plot_matrices(cost_matrix, assignment_matrix, title, N):
    """
    Supervisor Note: "matrix plotten als bild (sollte ähnlich aussehen)"
    Plots the Cost Matrix and the Final Assignment/Coupling Matrix.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Cost Matrix
    im1 = axes[0].imshow(cost_matrix, cmap='viridis')
    axes[0].set_title(f"Cost Matrix (N={N})")
    axes[0].set_xlabel("Points in Y")
    axes[0].set_ylabel("Points in X")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot Assignment/Coupling Matrix
    # Using 'Blues' cmap where darker = more mass/assigned
    im2 = axes[1].imshow(assignment_matrix, cmap='Blues')
    axes[1].set_title(title)
    axes[1].set_xlabel("Points in Y")
    axes[1].set_ylabel("Points in X")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def run_lap_test(N=16):
    print(f"\n--- Running LAP Test (N={N}) ---")
    np.random.seed(42)
    
    # Generate random points and cost matrix
    X_points = np.random.rand(N, 2)
    Y_points = np.random.rand(N, 2)
    C = cdist(X_points, Y_points, metric='sqeuclidean')
    
    # Supervisor Note: "runtime"
    start_time = time.perf_counter()
    
    manager = EpsScalingManager(AuctionLAP, C)
    assignments, total_cost, iters = manager.solve()
    
    end_time = time.perf_counter()
    
    print(f"LAP solved in {end_time - start_time:.4f} seconds.")
    print(f"Total Iterations: {iters}")
    print(f"Total Cost: {total_cost:.4f}")
    
    # Format 1D assignments into a 2D matrix for plotting
    assignment_matrix = np.zeros((N, N))
    for x, y in enumerate(assignments):
        assignment_matrix[x, y] = 1
        
    plot_matrices(C, assignment_matrix, "LAP Final Assignments", N)

def run_ot_test(N=16):
    print(f"\n--- Running OT Test (N={N}) ---")
    np.random.seed(42)
    
    # Generate points and cost
    X_points = np.random.rand(N, 2)
    Y_points = np.random.rand(N, 2)
    C = cdist(X_points, Y_points, metric='sqeuclidean')
    
    # Generate integer masses that sum to the same total
    # We use random integers between 1 and 5
    mu_X = np.random.randint(1, 6, size=N)
    mu_Y = np.random.randint(1, 6, size=N)
    
    # Balance the masses (force sum(mu_X) == sum(mu_Y))
    diff = np.sum(mu_X) - np.sum(mu_Y)
    if diff > 0:
        mu_Y[0] += diff
    elif diff < 0:
        mu_X[0] += abs(diff)
        
    start_time = time.perf_counter()
    
    manager = EpsScalingManager(AuctionOT, C, mu_X=mu_X, mu_Y=mu_Y)
    coupling_matrix, total_cost, iters = manager.solve()
    
    end_time = time.perf_counter()
    
    print(f"OT solved in {end_time - start_time:.4f} seconds.")
    print(f"Total Iterations: {iters}")
    print(f"Total Cost: {total_cost:.4f}")
    
    plot_matrices(C, coupling_matrix, "OT Final Coupling Matrix", N)

if __name__ == "__main__":
    # Supervisor Note: "bei kleiner punktmenge ausprobieren e.g. 16, 32"
    run_lap_test(N=16)
    run_ot_test(N=16)
