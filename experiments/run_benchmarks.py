import csv
import os
import sys
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ot

from src.utils.eps_scaling import EpsScalingManager
from src.core.ot_auction import AuctionOT
from src.hierarchical.partitions import HierarchicalPartition
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver

SCALES = [16, 32, 64, 128, 256, 512, 1024, 2048, 6000]

SEEDS_BY_N = {
    16:   [40, 50, 100, 2024, 999],
    32:   [40, 50, 100, 2024, 999],
    64:   [40, 50, 100, 2024, 999],
    128:  [40, 50, 100, 2024, 999],
    256:  [40, 50, 100],
    512:  [40, 50, 100],
    1024: [40, 50, 100],
}

# eps = ACCURACY / M with M the total mass. A final eps-CS plan satisfies
#   cost <= OPT + M * eps * max_c = OPT + ACCURACY * max_c      (raw cost units).
# The relative gap against POT therefore stays far below 1 percent. A larger eps is faster but
# the gap is then only bounded, not close to zero.
ACCURACY = 1e-3
MARGINAL_TOL = 1e-6

CSV_PATH = "benchmark_results.csv"


class Silence:
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


def build_matched_trees(X_pts, Y_pts, max_points_per_cell=1, max_allowed_depth=30):
    """Both trees with leaves of one point (required by the hierarchical solver)."""
    tree_X = HierarchicalPartition(X_pts, max_points_per_cell=max_points_per_cell,
                                   max_allowed_depth=max_allowed_depth)
    tree_Y = HierarchicalPartition(Y_pts, max_points_per_cell=max_points_per_cell,
                                   max_allowed_depth=max_allowed_depth)
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


def marginal_error(plan_dense, mu_X, mu_Y):
    return max(float(np.abs(plan_dense.sum(1) - mu_X).max()),
               float(np.abs(plan_dense.sum(0) - mu_Y).max()))


def run_benchmarks():
    header = (f"| {'N':<5} | {'Method':<13} | {'Time Mean':>10} | {'Time Std':>9} "
              f"| {'Gap Mean':>9} | {'Gap Std':>8} | {'Seeds':>5} | {'Fail':>4} |")
    rule = "-" * len(header)

    print()
    print("BACHELOR THESIS: OPTIMAL TRANSPORT SOLVER BENCHMARK")
    print(f"accuracy parameter: eps = {ACCURACY:g} / M   (cost <= OPT + {ACCURACY:g} * max_c)")
    print(rule)
    print(header)
    print(rule)

    keys = ["pot_time", "dense_time", "hier_time", "dense_gap", "hier_gap"]
    results = {"N": []}
    for k in keys:
        results[k + "_mean"] = []
        results[k + "_std"] = []

    with open(CSV_PATH, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["N", "seeds", "failures"]
            + [f"{k}_{s}" for k in keys for s in ("mean", "std")])

    for N in SCALES:
        seeds = SEEDS_BY_N.get(N, [42, 50, 100])
        data = {k: [] for k in keys}
        failures = 0

        for seed in seeds:
            X_pts, Y_pts, mu_X, mu_Y, C, global_max_c = make_instance(N, seed)
            M = float(mu_X.sum())
            target = ACCURACY / M

            try:
                # POT reference, timed including its own cost matrix
                t0 = time.perf_counter()
                C_pot = ot.dist(X_pts, Y_pts, metric="sqeuclidean")
                exact_cost = float(ot.emd2(mu_X, mu_Y, C_pot))
                pot_time = time.perf_counter() - t0

                # Dense auction with eps-scaling
                t0 = time.perf_counter()
                with Silence():
                    dense_mgr = EpsScalingManager(
                        AuctionOT, X_pts=X_pts, Y_pts=Y_pts, mu_X=mu_X, mu_Y=mu_Y,
                        normalize=False, max_c=global_max_c, target_eps=target)
                    dense_mu, _, _, _ = dense_mgr.solve()
                dense_time = time.perf_counter() - t0

                P = np.zeros((N, N))
                for x in dense_mu:
                    for y, m in dense_mu[x].items():
                        P[x, y] = m
                if marginal_error(P, mu_X, mu_Y) > MARGINAL_TOL:
                    raise RuntimeError("dense plan violates the marginals")
                dense_cost = float((P * C).sum())

                # Hierarchical auction; tree construction is part of the timed work
                t0 = time.perf_counter()
                with Silence():
                    tree_X, tree_Y = build_matched_trees(X_pts, Y_pts)
                    hier_solver = HierarchicalMultiscaleSolver(
                        tree_X, tree_Y, mu_X, mu_Y, max_c=global_max_c, target_eps=target)
                    plan = hier_solver.solve()
                hier_time = time.perf_counter() - t0

                Ph = np.zeros((N, N))
                for x, y, m in plan:
                    Ph[x, y] += m
                if marginal_error(Ph, mu_X, mu_Y) > MARGINAL_TOL:
                    raise RuntimeError("hierarchical plan violates the marginals")
                hier_cost = float((Ph * C).sum())

            except Exception:
                failures += 1
                print(f"[N={N}, seed={seed}] FAILED:", file=sys.__stderr__)
                traceback.print_exc(file=sys.__stderr__)
                continue

            data["pot_time"].append(pot_time)
            data["dense_time"].append(dense_time)
            data["hier_time"].append(hier_time)
            data["dense_gap"].append((dense_cost - exact_cost) / exact_cost * 100.0)
            data["hier_gap"].append((hier_cost - exact_cost) / exact_cost * 100.0)

        n_ok = len(data["dense_time"])
        if n_ok == 0:
            print(f"| {N:<5} | {'ALL FAILED':<13} | {'':>10} | {'':>9} | {'':>9} | {'':>8} "
                  f"| {0:>5} | {failures:>4} |")
            print(rule)
            continue

        ddof = 1 if n_ok > 1 else 0
        row = {}
        for k in keys:
            row[k + "_mean"] = float(np.mean(data[k]))
            row[k + "_std"] = float(np.std(data[k], ddof=ddof))

        results["N"].append(N)
        for key, value in row.items():
            results[key].append(value)

        for label, k in (("POT EMD", "pot"), ("DENSE OT", "dense"), ("HIERARCH. OT", "hier")):
            gap_m = f"{row[k + '_gap_mean']:>8.4f}%" if k != "pot" else f"{'-':>9}"
            gap_s = f"{row[k + '_gap_std']:>7.4f}%" if k != "pot" else f"{'-':>8}"
            print(f"| {N:<5} | {label:<13} | {row[k + '_time_mean']:>10.4f} "
                  f"| {row[k + '_time_std']:>9.4f} | {gap_m} | {gap_s} | {n_ok:>5} | {failures:>4} |")
        print(rule)

        with open(CSV_PATH, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [N, n_ok, failures]
                + [row[k + "_" + s] for k in keys for s in ("mean", "std")])

    print_summary(results)
    return results


def print_summary(results):
    if len(results["N"]) < 2:
        return

    N_arr = np.array(results["N"], dtype=float)
    series = {name: np.array(results[name + "_time_mean"])
              for name in ("pot", "dense", "hier")}

    print()
    print("SCALING SUMMARY")
    line = (f"| {'N':<5} | {'hier/dense':>10} | {'pot slope':>9} | {'dense slope':>11} "
            f"| {'hier slope':>10} |")
    print("-" * len(line))
    print(line)
    print("-" * len(line))

    for i, N in enumerate(results["N"]):
        ratio = series["hier"][i] / series["dense"][i]
        if i == 0:
            slopes = ["", "", ""]
        else:
            step = np.log(N_arr[i] / N_arr[i - 1])
            slopes = [f"{np.log(series[n][i] / series[n][i - 1]) / step:.2f}"
                      for n in ("pot", "dense", "hier")]
        print(f"| {N:<5} | {ratio:>10.2f} | {slopes[0]:>9} | {slopes[1]:>11} | {slopes[2]:>10} |")

    print("-" * len(line))
    for name, label in (("pot", "POT"), ("dense", "dense"), ("hier", "hierarchical")):
        fit = np.polyfit(np.log(N_arr), np.log(series[name]), 1)[0]
        print(f"Fitted exponent, {label}: N^{fit:.2f}")
    print(f"Results written to {CSV_PATH}")


def plot_results(results):
    if len(results["N"]) < 2:
        return

    N_arr = np.array(results["N"], dtype=float)

    plt.figure(figsize=(8, 6))
    for name, label, marker in [("pot", "POT (EMD)", "^"),
                                ("dense", "Dense Auction", "o"),
                                ("hier", "Hierarchical Auction", "s")]:
        mean = np.array(results[name + "_time_mean"])
        std = np.array(results[name + "_time_std"])
        plt.loglog(N_arr, mean, marker=marker, label=label, linewidth=2)
        plt.fill_between(N_arr, np.maximum(mean - std, 1e-9), mean + std, alpha=0.2)

    plt.title("Computation Time vs Problem Size")
    plt.xlabel("Number of Points (N)")
    plt.ylabel("Time (seconds)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("thesis_scaling_plot.pdf")
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, label, marker in [("dense", "Dense", "o"), ("hier", "Hierarchical", "s")]:
        mean = np.array(results[name + "_gap_mean"])
        std = np.array(results[name + "_gap_std"])
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
