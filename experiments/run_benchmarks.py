import csv
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ot

from src.utils.eps_scaling import EpsScalingManager
from src.core.ot_auction import AuctionOT
from src.hierarchical.partitions import HierarchicalPartition
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCALES = [16, 32, 64, 128, 256, 512, 1024]

# Fewer seeds at the large sizes so a full sweep actually finishes.
SEEDS_BY_N = {
    16:   [42, 50, 100, 2024, 999],
    32:   [42, 50, 100, 2024, 999],
    64:   [42, 50, 100, 2024, 999],
    128:  [42, 50, 100, 2024, 999],
    256:  [42, 50, 100],
    512:  [42, 50, 100],
    1024: [42, 50, 100],
}

CSV_PATH = "benchmark_results.csv"


# ---------------------------------------------------------------------------
# Output suppression
# ---------------------------------------------------------------------------

class Silence:
    """
    Suppress stdout AND stderr.

    The eps-scaling manager prints to stdout, the multiscale solver writes its
    depth and loop diagnostics to stderr. Silencing only stdout leaves the
    [Depth N] lines in the table, which is why the old benchmark was noisy.
    """

    def __enter__(self):
        self._devnull = open(os.devnull, "w")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = self._devnull
        sys.stderr = self._devnull
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self._devnull.close()
        return False


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

def build_matched_trees(X_pts, Y_pts, max_points_per_cell=1, max_allowed_depth=15):
    probe_X = HierarchicalPartition(X_pts, max_points_per_cell=max_points_per_cell,
                                    max_allowed_depth=max_allowed_depth)
    probe_Y = HierarchicalPartition(Y_pts, max_points_per_cell=max_points_per_cell,
                                    max_allowed_depth=max_allowed_depth)
    target_depth = max(probe_X.max_depth, probe_Y.max_depth)
    tree_X = HierarchicalPartition(X_pts, max_points_per_cell=max_points_per_cell,
                                   max_allowed_depth=target_depth)
    tree_Y = HierarchicalPartition(Y_pts, max_points_per_cell=max_points_per_cell,
                                   max_allowed_depth=target_depth)
    return tree_X, tree_Y


def make_instance(N, seed):
    np.random.seed(seed)
    X_pts = np.random.rand(N, 2)
    Y_pts = np.random.rand(N, 2)

    mu_X = np.random.randint(1, 6, size=N).astype(float)
    mu_Y = np.random.randint(1, 6, size=N).astype(float)

    diff = np.sum(mu_X) - np.sum(mu_Y)
    if diff > 0:
        mu_Y[0] += diff
    elif diff < 0:
        mu_X[0] += abs(diff)

    diffs = X_pts[:, None, :] - Y_pts[None, :, :]
    C = np.sum(diffs ** 2, axis=2)

    gmin = np.minimum(X_pts.min(axis=0), Y_pts.min(axis=0))
    gmax = np.maximum(X_pts.max(axis=0), Y_pts.max(axis=0))
    global_max_c = float(np.sum((gmax - gmin) ** 2)) or 1.0

    return X_pts, Y_pts, mu_X, mu_Y, C, global_max_c


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmarks():
    header = (f"| {'N':<5} | {'Method':<13} | {'Time Mean':>10} | {'Time Std':>9} "
              f"| {'Gap Mean':>9} | {'Gap Std':>8} | {'Seeds':>5} |")
    rule = "-" * len(header)

    print()
    print("BACHELOR THESIS: OPTIMAL TRANSPORT SOLVER BENCHMARK")
    print(rule)
    print(header)
    print(rule)

    results = {
        "N": [],
        "dense_time_mean": [], "dense_time_std": [],
        "hier_time_mean": [], "hier_time_std": [],
        "dense_gap_mean": [], "dense_gap_std": [],
        "hier_gap_mean": [], "hier_gap_std": [],
    }

    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "N", "seeds",
            "dense_time_mean", "dense_time_std", "dense_gap_mean", "dense_gap_std",
            "hier_time_mean", "hier_time_std", "hier_gap_mean", "hier_gap_std",
        ])

    for N in SCALES:
        seeds = SEEDS_BY_N.get(N, [42, 50, 100])
        dense_times, hier_times = [], []
        dense_gaps, hier_gaps = [], []

        tight_target = 0.5 / (N + 1)
        tight_min = 1e-5

        for seed in seeds:
            X_pts, Y_pts, mu_X, mu_Y, C, global_max_c = make_instance(N, seed)

            try:
                exact_cost = ot.emd2(mu_X, mu_Y, C)
            except Exception:
                continue

            tree_X, tree_Y = build_matched_trees(X_pts, Y_pts)

            # Dense baseline
            t0 = time.perf_counter()
            with Silence():
                dense_mgr = EpsScalingManager(
                    AuctionOT, X_pts=X_pts, Y_pts=Y_pts, mu_X=mu_X, mu_Y=mu_Y,
                    normalize=False, max_c=global_max_c,
                    target_eps=tight_target, min_eps=tight_min,
                )
                dense_mu_dict, _, _, _ = dense_mgr.solve()
            dense_times.append(time.perf_counter() - t0)

            dense_cost = 0.0
            for x in dense_mu_dict:
                for y, m in dense_mu_dict[x].items():
                    dense_cost += m * C[x, y]
            dense_gaps.append((dense_cost - exact_cost) / exact_cost * 100.0)

            # Hierarchical solver
            t1 = time.perf_counter()
            with Silence():
                hier_solver = HierarchicalMultiscaleSolver(
                    tree_X, tree_Y, mu_X, mu_Y,
                    max_c=global_max_c,
                    target_eps=tight_target, min_eps=tight_min,
                )
                sparse_hier_mu = hier_solver.solve()
            hier_times.append(time.perf_counter() - t1)

            hier_cost = sum(mass * C[x, y] for x, y, mass in sparse_hier_mu)
            hier_gaps.append((hier_cost - exact_cost) / exact_cost * 100.0)

        if not dense_times:
            print(f"| {N:<5} | {'SKIPPED':<13} | {'':>10} | {'':>9} | {'':>9} | {'':>8} | {0:>5} |")
            print(rule)
            continue

        # ddof=1 gives the sample standard deviation, which is what you want
        # with three to five seeds.
        ddof = 1 if len(dense_times) > 1 else 0
        row = {
            "dense_time_mean": float(np.mean(dense_times)),
            "dense_time_std": float(np.std(dense_times, ddof=ddof)),
            "dense_gap_mean": float(np.mean(dense_gaps)),
            "dense_gap_std": float(np.std(dense_gaps, ddof=ddof)),
            "hier_time_mean": float(np.mean(hier_times)),
            "hier_time_std": float(np.std(hier_times, ddof=ddof)),
            "hier_gap_mean": float(np.mean(hier_gaps)),
            "hier_gap_std": float(np.std(hier_gaps, ddof=ddof)),
        }

        results["N"].append(N)
        for key, value in row.items():
            results[key].append(value)

        print(f"| {N:<5} | {'DENSE OT':<13} | {row['dense_time_mean']:>10.4f} "
              f"| {row['dense_time_std']:>9.4f} | {row['dense_gap_mean']:>8.3f}% "
              f"| {row['dense_gap_std']:>7.3f}% | {len(dense_times):>5} |")
        print(f"| {N:<5} | {'HIERARCH. OT':<13} | {row['hier_time_mean']:>10.4f} "
              f"| {row['hier_time_std']:>9.4f} | {row['hier_gap_mean']:>8.3f}% "
              f"| {row['hier_gap_std']:>7.3f}% | {len(hier_times):>5} |")
        print(rule)

        # Checkpoint after every N so a Ctrl+C does not lose the whole sweep.
        with open(CSV_PATH, "a", newline="") as fh:
            csv.writer(fh).writerow([
                N, len(dense_times),
                row["dense_time_mean"], row["dense_time_std"],
                row["dense_gap_mean"], row["dense_gap_std"],
                row["hier_time_mean"], row["hier_time_std"],
                row["hier_gap_mean"], row["hier_gap_std"],
            ])

    print_summary(results)
    return results


def print_summary(results):
    if len(results["N"]) < 2:
        return

    N_arr = np.array(results["N"], dtype=float)
    dense = np.array(results["dense_time_mean"])
    hier = np.array(results["hier_time_mean"])

    print()
    print("SCALING SUMMARY")
    line = f"| {'N':<5} | {'hier/dense':>10} | {'dense slope':>11} | {'hier slope':>10} |"
    print("-" * len(line))
    print(line)
    print("-" * len(line))

    for i, N in enumerate(results["N"]):
        ratio = hier[i] / dense[i]
        if i == 0:
            d_slope = h_slope = ""
        else:
            step = np.log(N_arr[i] / N_arr[i - 1])
            d_slope = f"{np.log(dense[i] / dense[i - 1]) / step:>11.2f}"
            h_slope = f"{np.log(hier[i] / hier[i - 1]) / step:>10.2f}"
        print(f"| {N:<5} | {ratio:>10.2f} | {d_slope:>11} | {h_slope:>10} |")

    print("-" * len(line))

    # Overall fitted exponents across the whole sweep.
    d_fit = np.polyfit(np.log(N_arr), np.log(dense), 1)[0]
    h_fit = np.polyfit(np.log(N_arr), np.log(hier), 1)[0]
    print(f"Fitted exponent, dense:        N^{d_fit:.2f}")
    print(f"Fitted exponent, hierarchical: N^{h_fit:.2f}")
    print(f"Results written to {CSV_PATH}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(results):
    if len(results["N"]) < 2:
        return

    N_arr = np.array(results["N"], dtype=float)

    plt.figure(figsize=(8, 6))
    for key_mean, key_std, label, marker in [
        ("dense_time_mean", "dense_time_std", "Dense Auction", "o"),
        ("hier_time_mean", "hier_time_std", "Hierarchical Auction", "s"),
    ]:
        mean = np.array(results[key_mean])
        std = np.array(results[key_std])
        plt.loglog(N_arr, mean, marker=marker, label=label, linewidth=2)
        plt.fill_between(N_arr, mean - std, mean + std, alpha=0.2)

    plt.title("Computation Time vs Problem Size")
    plt.xlabel("Number of Points (N)")
    plt.ylabel("Time (seconds)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("thesis_scaling_plot.pdf")
    plt.close()

    plt.figure(figsize=(8, 6))
    for key_mean, key_std, label, marker in [
        ("dense_gap_mean", "dense_gap_std", "Dense", "o"),
        ("hier_gap_mean", "hier_gap_std", "Hierarchical", "s"),
    ]:
        mean = np.array(results[key_mean])
        std = np.array(results[key_std])
        plt.semilogx(N_arr, mean, marker=marker, label=label, linewidth=2)
        plt.fill_between(N_arr, mean - std, mean + std, alpha=0.2)

    plt.title("Optimality Gap vs Exact Reference (POT)")
    plt.xlabel("Number of Points (N)")
    plt.ylabel("Relative Gap (%)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("thesis_gap_plot.pdf")
    plt.close()


if __name__ == "__main__":
    final_results = run_benchmarks()
    plot_results(final_results)
