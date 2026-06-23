import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.cost_functions import squared_euclidean
from src.hierarchical.partitions import HierarchicalPartition
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver

# Create a custom visualization subclass so we don't mess up your clean production solver
class VisualizableMultiscaleSolver(HierarchicalMultiscaleSolver):
    def __init__(self, tree_X, tree_Y, cost_matrix, mu_X, mu_Y, output_dir="plots"):
        super().__init__(tree_X, tree_Y, cost_matrix, mu_X, mu_Y)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_generation_state(self, gen, C_fine, mu_fine, N_guess=None):
        """Generates a side-by-side subplot of the generation's matrix state."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left Plot: Coarsened Cost Matrix
        im0 = axes[0].imshow(C_fine, cmap='viridis', origin='upper')
        axes[0].set_title(f"Coarse Cost Matrix (Gen {gen})")
        axes[0].set_xlabel("Y Coarse Cells")
        axes[0].set_ylabel("X Coarse Cells")
        fig.colorbar(im0, ax=axes[0], label="Cost")
        
        # Right Plot: Aggregated Coupling Matrix
        # We overlay a red grid outline over the sparse guess neighborhood if provided
        im1 = axes[1].imshow(mu_fine, cmap='Blues', origin='upper', vmin=0)
        axes[1].set_title(f"Coupling Allocation (Gen {gen})")
        axes[1].set_xlabel("Y Coarse Cells")
        axes[1].set_ylabel("X Coarse Cells")
        fig.colorbar(im1, ax=axes[1], label="Mass Transported")
        
        if N_guess is not None:
            # Highlight the sparse neighborhood boundaries with small red dots
            for (x, y) in N_guess:
                axes[1].plot(y, x, 'rx', markersize=4, alpha=0.4)
                
        plt.suptitle(f"Hierarchical Multiscale Refinement — Generation {gen} ({C_fine.shape[0]}x{C_fine.shape[1]})", fontsize=14)
        plt.tight_layout()
        
        # Save file instead of blocking terminal execution
        file_path = os.path.join(self.output_dir, f"generation_{gen}.png")
        plt.savefig(file_path, dpi=150)
        plt.close()
        print(f"  [Visualizer] Saved snapshot: {file_path}")

    def solve_with_plots(self):
        """Runs the multiscale loop and takes a snapshot at every scale level."""
        coarsest_gen = self.g - 1
        print(f"\n[Visualizer Master Loop] Starting at Coarsest Generation: {coarsest_gen}")
        
        # 1. Base Level Root Solve
        C_fine, mu_X_fine, mu_Y_fine = self._build_coarsened_problem(coarsest_gen)
        from src.utils.eps_scaling import EpsScalingManager
        from src.core.ot_auction import AuctionOT
        
        manager = EpsScalingManager(AuctionOT, C_fine, mu_X=mu_X_fine, mu_Y=mu_Y_fine)
        current_mu, _, _, _ = manager.solve()
        
        # Snapshot the root level state
        self.plot_generation_state(coarsest_gen, C_fine, current_mu)
        
        # 2. Sequential Descent down the tree levels
        for gen in range(coarsest_gen - 1, -1, -1):
            N_guess = self._induce_sparse_neighborhood(current_mu, gen + 1)
            C_fine, mu_X_fine, mu_Y_fine = self._build_coarsened_problem(gen)
            
            while True:
                hybrid_manager = EpsScalingManager(
                    AuctionOT, C_fine, mu_X=mu_X_fine, mu_Y=mu_Y_fine, allowed_edges=N_guess
                )
                current_mu, _, _, final_beta = hybrid_manager.solve()
                
                # Check for boundary dynamic updates
                alpha = np.zeros(len(mu_X_fine))
                for x in range(len(mu_X_fine)):
                    assigned_ys = np.where(current_mu[x] > 0)[0]
                    if len(assigned_ys) > 0:
                        alpha[x] = np.min(C_fine[x, assigned_ys] - final_beta[assigned_ys])
                
                violations = []
                target_eps = hybrid_manager.target_eps
                N_guess_set = set(N_guess)
                for x in range(len(mu_X_fine)):
                    for y in range(len(mu_Y_fine)):
                        if (x, y) not in N_guess_set:
                            if alpha[x] + final_beta[y] > C_fine[x, y] + target_eps + 1e-5:
                                violations.append((x, y))
                if len(violations) == 0:
                    break
                else:
                    N_guess.extend(violations)
            
            # Snapshot the optimized, corrected generation state before moving to the next
            self.plot_generation_state(gen, C_fine, current_mu, N_guess)
            
        print("\nAll generations visualised successfully!")

def run_visualization():
    np.random.seed(101)
    N = 32
    X_pts = np.random.rand(N, 2)
    Y_pts = np.random.rand(N, 2)
    C = squared_euclidean(X_pts, Y_pts)
    
    mu_X = np.random.randint(1, 5, size=N)
    mu_Y = np.random.randint(1, 5, size=N)
    diff = np.sum(mu_X) - np.sum(mu_Y)
    if diff > 0: mu_Y[0] += diff
    elif diff < 0: mu_X[0] += abs(diff)
        
    tree_X = HierarchicalPartition(X_pts)
    tree_Y = HierarchicalPartition(Y_pts)
    
    solver = VisualizableMultiscaleSolver(tree_X, tree_Y, C, mu_X, mu_Y)
    solver.solve_with_plots()

if __name__ == "__main__":
    run_visualization()
