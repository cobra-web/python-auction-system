import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from src.utils.eps_scaling import EpsScalingManager
from src.core.ot_auction import AuctionOT
from src.hierarchical.partitions import HierarchicalPartition
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver

def plot_transport_plans(X_pts, Y_pts, dense_mu, hier_mu, title_suffix=""):
    """
    Visualizes the active transport edges for the Dense and Hierarchical solvers,
    along with a direct structural comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Optimal Transport Primal Plan Comparison {title_suffix}", fontsize=16)

    threshold = 1e-5

    # Helper function to extract lines and widths
    def extract_edges(mu_matrix):
        lines = []
        widths = []
        for i in range(mu_matrix.shape[0]):
            for j in range(mu_matrix.shape[1]):
                if mu_matrix[i, j] > threshold:
                    lines.append([X_pts[i], Y_pts[j]])
                    widths.append(mu_matrix[i, j])
        return lines, widths

    dense_lines, dense_widths = extract_edges(dense_mu)
    hier_lines, hier_widths = extract_edges(hier_mu)

    # Normalize widths for visual clarity
    max_mass = max(max(dense_widths) if dense_widths else 1.0, 
                   max(hier_widths) if hier_widths else 1.0)
    
    dense_widths_norm = [w / max_mass * 3 for w in dense_widths]
    hier_widths_norm = [w / max_mass * 3 for w in hier_widths]

    # --- PLOT 1: DENSE OT ---
    ax = axes[0]
    ax.set_title(f"Dense OT Plan\n({len(dense_lines)} Active Pairs)")
    ax.scatter(X_pts[:, 0], X_pts[:, 1], c='blue', label='Supply (X)', zorder=5, s=30)
    ax.scatter(Y_pts[:, 0], Y_pts[:, 1], c='red', marker='s', label='Demand (Y)', zorder=5, s=30)
    lc_dense = LineCollection(dense_lines, linewidths=dense_widths_norm, colors='black', alpha=0.5)
    ax.add_collection(lc_dense)
    ax.legend()

    # --- PLOT 2: HIERARCHICAL OT ---
    ax = axes[1]
    ax.set_title(f"Hierarchical OT Plan\n({len(hier_lines)} Active Pairs)")
    ax.scatter(X_pts[:, 0], X_pts[:, 1], c='blue', zorder=5, s=30)
    ax.scatter(Y_pts[:, 0], Y_pts[:, 1], c='red', marker='s', zorder=5, s=30)
    lc_hier = LineCollection(hier_lines, linewidths=hier_widths_norm, colors='green', alpha=0.5)
    ax.add_collection(lc_hier)

    # --- PLOT 3: DIFFERENCE OVERLAY ---
    ax = axes[2]
    ax.set_title("Edge Discrepancy\n(Grey=Shared, Red=Dense Only, Green=Hier Only)")
    ax.scatter(X_pts[:, 0], X_pts[:, 1], c='blue', zorder=5, s=15, alpha=0.5)
    ax.scatter(Y_pts[:, 0], Y_pts[:, 1], c='red', marker='s', zorder=5, s=15, alpha=0.5)

    shared_lines = []
    dense_only_lines = []
    hier_only_lines = []

    for i in range(dense_mu.shape[0]):
        for j in range(dense_mu.shape[1]):
            in_dense = dense_mu[i, j] > threshold
            in_hier = hier_mu[i, j] > threshold
            
            line = [X_pts[i], Y_pts[j]]
            
            if in_dense and in_hier:
                shared_lines.append(line)
            elif in_dense:
                dense_only_lines.append(line)
            elif in_hier:
                hier_only_lines.append(line)

    lc_shared = LineCollection(shared_lines, linewidths=1.0, colors='grey', alpha=0.3, zorder=1)
    lc_dense_only = LineCollection(dense_only_lines, linewidths=1.5, colors='red', alpha=0.8, zorder=2)
    lc_hier_only = LineCollection(hier_only_lines, linewidths=1.5, colors='green', alpha=0.8, zorder=3)

    ax.add_collection(lc_shared)
    ax.add_collection(lc_dense_only)
    ax.add_collection(lc_hier_only)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

def run_standalone_visualization():
    print("Generating data for visualization...")
    N = 45  # Good scale for clear visualization
    np.random.seed(42)
    
    X_pts = np.random.rand(N, 2)
    Y_pts = np.random.rand(N, 2)
    
    mu_X = np.random.randint(1, 6, size=N)
    mu_Y = np.random.randint(1, 6, size=N)
    
    # Balance mass
    diff = np.sum(mu_X) - np.sum(mu_Y)
    if diff > 0:
        mu_Y[0] += diff
    elif diff < 0:
        mu_X[0] += abs(diff)

    print("Running Dense Solver...")
    dense_manager = EpsScalingManager(AuctionOT, X_pts=X_pts, Y_pts=Y_pts, mu_X=mu_X, mu_Y=mu_Y)
    dense_mu_dict, _, _, _ = dense_manager.solve()
    
    # Reconstruct dense mu
    dense_mu = np.zeros((N, N))
    for x in dense_mu_dict:
        for y, m in dense_mu_dict[x].items():
            dense_mu[x, y] = m

    print("Running Hierarchical Solver...")
    probe_X = HierarchicalPartition(X_pts, max_points_per_cell=1, max_allowed_depth=15)
    probe_Y = HierarchicalPartition(Y_pts, max_points_per_cell=1, max_allowed_depth=15)

    target_depth = max(probe_X.max_depth, probe_Y.max_depth)
    
    tree_X = HierarchicalPartition(X_pts, max_points_per_cell=1, max_allowed_depth=target_depth)
    tree_Y = HierarchicalPartition(Y_pts, max_points_per_cell=1, max_allowed_depth=target_depth)
    
    multiscale_solver = HierarchicalMultiscaleSolver(tree_X, tree_Y, mu_X, mu_Y)
    sparse_hier_mu = multiscale_solver.solve()
    
    # Reconstruct hier mu
    hier_mu = np.zeros((N, N))
    for x, y, mass in sparse_hier_mu:
        hier_mu[x, y] += mass

    print("Rendering Plots...")
    plot_transport_plans(X_pts, Y_pts, dense_mu, hier_mu, title_suffix=f"(N={N})")

if __name__ == "__main__":
    run_standalone_visualization()
