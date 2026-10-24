import os

# Single-threaded BLAS/OpenMP, so that timings are not distorted by NumPy threading.
# Must be set before NumPy is imported (also in spawned child processes, which re-import this file).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
import multiprocessing as mp
import platform
import sys
import time
import traceback
import warnings
from functools import lru_cache

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ot

from src.utils.eps_scaling import EpsScalingManager
from src.core.ot_auction import AuctionOT
from src.hierarchical.partitions import HierarchicalPartition
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver

# ----------------------------------------------------------------------------- settings
SCALES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
FAMILIES = ("uniform_legacy", "uniform", "clustered", "unit", "glyph", "image")
DEFAULT_FAMILY = "uniform_legacy"

SEEDS_SMALL = [40, 50, 100, 2024, 999]   # N <= SMALL_N_MAX
SEEDS_LARGE = [40, 50, 100]
SMALL_N_MAX = 128
# Seeds of the original run, so that it can be reproduced exactly (the first seed at the
# largest sizes was 42 there).
LEGACY_SEEDS_BY_N = {
    16: SEEDS_SMALL, 32: SEEDS_SMALL, 64: SEEDS_SMALL, 128: SEEDS_SMALL,
    256: SEEDS_LARGE, 512: SEEDS_LARGE, 1024: SEEDS_LARGE,
}
LEGACY_SEEDS_DEFAULT = [42, 50, 100]

# eps* = ACCURACY / M with M the total mass. An eps*-CS plan satisfies
#   cost <= OPT + M * eps* * C_max = OPT + ACCURACY * C_max      (raw cost units)
# (Propositions 3.5 and 3.12), with C_max the normalisation constant of the family
# (max_{x,y} c(x,y), or the bounding-box diagonal for uniform_legacy; both are upper bounds).
ACCURACY = 1e-3
SWEEP_N = 1024
SWEEP_ACCURACIES = [1e-1, 1e-2, 1e-3, 1e-4]

MARGINAL_TOL = 1e-6
BOUND_RTOL = 1e-9            # slack for floating point when checking the proven bound
POT_MAX_ITER = 100_000_000   # POT's default (100000) is too small for N >= 4096
DENSE_MAX_N_DEFAULT = 2048   # above this the dense auction is skipped unless --dense-max-n
FIT_MIN_N = 256              # exponents are fitted on N >= FIT_MIN_N only (overhead below)

GLYPH_DEFAULT = ("\u9f99", "\u51e4")   # two Chinese characters (long, feng)
GLYPH_CANVAS = 256

LOG_PATH = "benchmark_solver.log"
META_PATH = "benchmark_meta.txt"
SWEEP_PATH = "benchmark_sweep.csv"

RAW_HEADER = ["N", "seed", "family", "pot_time", "dense_time", "dense_status", "dense_cap",
              "hier_time", "speedup", "dense_gap", "hier_gap", "dense_bound_ok",
              "hier_bound_ok", "hier_edges"]
KEYS = ["pot_time", "dense_time", "hier_time", "dense_gap", "hier_gap", "hier_edges", "speedup"]
NAN = float("nan")


def raw_path(label):
    return f"benchmark_raw_{label}.csv"


def csv_path(label):
    return f"benchmark_results_{label}.csv"


def seeds_for(N, family):
    if family == "uniform_legacy":
        return LEGACY_SEEDS_BY_N.get(N, LEGACY_SEEDS_DEFAULT)
    return SEEDS_SMALL if N <= SMALL_N_MAX else SEEDS_LARGE


# ----------------------------------------------------------------------------- utilities
class Capture:
    """Redirect solver output to a log file instead of discarding it."""

    def __init__(self, tag):
        self.tag = tag

    def __enter__(self):
        self._fh = open(LOG_PATH, "a")
        self._fh.write(f"\n=== {self.tag} ===\n")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = self._fh
        sys.stderr = self._fh
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self._fh.close()
        return False


def progress(msg):
    """Always visible, also while a solver output is captured."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.__stderr__, flush=True)


def write_metadata(extra=None):
    lines = [
        f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"python: {sys.version.split()[0]}",
        f"numpy: {np.__version__}",
        f"POT: {ot.__version__}",
        f"platform: {platform.platform()}",
        f"processor: {platform.processor() or platform.machine()}",
        f"cpu_count: {os.cpu_count()}",
        f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}",
        f"OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS')}",
        f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS')}",
        f"ACCURACY: {ACCURACY}",
        f"FIT_MIN_N: {FIT_MIN_N}",
        f"SCALES: {SCALES}",
    ]
    lines += [f"{k}: {v}" for k, v in (extra or {}).items()]
    with open(META_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def build_matched_trees(X_pts, Y_pts, max_points_per_cell=1, max_allowed_depth=30):
    """Both trees with leaves of one point (required by the hierarchical solver)."""
    tree_X = HierarchicalPartition(X_pts, max_points_per_cell=max_points_per_cell,
                                   max_allowed_depth=max_allowed_depth)
    tree_Y = HierarchicalPartition(Y_pts, max_points_per_cell=max_points_per_cell,
                                   max_allowed_depth=max_allowed_depth)
    return tree_X, tree_Y


# ----------------------------------------------------------------------------- instances
def _find_font(explicit=None):
    candidates = [
        explicit,
        os.environ.get("BENCH_FONT"),
        r"C:\Windows\Fonts\msyh.ttc",       # Microsoft YaHei
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No CJK font found. Pass one with --font (for example C:\\Windows\\Fonts\\msyh.ttc) "
        "or set the environment variable BENCH_FONT.")


@lru_cache(maxsize=8)
def _render_glyph(char, font_path):
    """Grayscale image in [0, 1] of one character, white ink on black."""
    from PIL import Image, ImageDraw, ImageFont
    size = GLYPH_CANVAS
    font = ImageFont.truetype(font_path, int(size * 0.85))
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).text((size / 2, size / 2), char, fill=255, font=font, anchor="mm")
    return np.asarray(img, dtype=float) / 255.0


@lru_cache(maxsize=8)
def _load_image(path):
    """Grayscale ink image in [0, 1]; dark-on-light images are inverted."""
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0
    return 1.0 - a if a.mean() > 0.5 else a


def _sample_ink(img, N, rng):
    """N points drawn from the ink: pixel with probability proportional to intensity, then a
    uniform position inside the pixel (so all points are distinct). Coordinates lie in [0, 1]^2."""
    h, w = img.shape
    p = img.ravel().astype(float)
    total = p.sum()
    if total <= 0:
        raise ValueError("the image contains no ink")
    idx = rng.choice(p.size, size=N, p=p / total)
    rows, cols = np.divmod(idx, w)
    s = float(max(h, w))
    return np.column_stack([(cols + rng.random(N)) / s, (h - rows - rng.random(N)) / s])


def _points(rng, N, family, spec):
    spec = spec or {}
    if family in ("uniform", "unit"):
        return rng.random((N, 2)), rng.random((N, 2))
    if family == "clustered":
        k = 5
        out = []
        for _ in range(2):
            centres = rng.random((k, 2))
            labels = rng.integers(0, k, size=N)
            out.append(centres[labels] + 0.03 * rng.standard_normal((N, 2)))
        return out[0], out[1]
    if family == "glyph":
        font = _find_font(spec.get("font"))
        gx, gy = spec.get("glyph_x", GLYPH_DEFAULT[0]), spec.get("glyph_y", GLYPH_DEFAULT[1])
        return (_sample_ink(_render_glyph(gx, font), N, rng),
                _sample_ink(_render_glyph(gy, font), N, rng))
    if family == "image":
        if not spec.get("image_x") or not spec.get("image_y"):
            raise ValueError("family 'image' needs --image-x and --image-y")
        return (_sample_ink(_load_image(spec["image_x"]), N, rng),
                _sample_ink(_load_image(spec["image_y"]), N, rng))
    raise ValueError(f"unknown family {family!r}")


def _legacy_instance(N, seed):
    """Exactly the generator of the original thesis run."""
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
    C_max = float(np.sum((gmax - gmin) ** 2)) or 1.0
    return X_pts, Y_pts, mu_X, mu_Y, C, C_max


def make_instance(N, seed, family=DEFAULT_FAMILY, spec=None):
    """Returns X_pts, Y_pts, mu_X, mu_Y, C, C_max."""
    if family == "uniform_legacy":
        return _legacy_instance(N, seed)

    rng = np.random.default_rng(seed)
    X_pts, Y_pts = _points(rng, N, family, spec)

    if family in ("unit", "glyph", "image"):
        mu_X = np.ones(N)
        mu_Y = np.ones(N)
    else:
        mu_X = rng.integers(1, 6, size=N).astype(float)
        mu_Y = rng.integers(1, 6, size=N).astype(float)
        # Balance the total mass by spreading the surplus randomly (no single heavy point).
        diff = int(mu_X.sum() - mu_Y.sum())
        recipient, k = (mu_Y, diff) if diff > 0 else (mu_X, -diff)
        np.add.at(recipient, rng.integers(0, N, size=k), 1.0)

    diffs = X_pts[:, None, :] - Y_pts[None, :, :]
    C = np.sum(diffs ** 2, axis=2)
    C_max = float(C.max()) or 1.0
    return X_pts, Y_pts, mu_X, mu_Y, C, C_max


def marginal_error(plan_dense, mu_X, mu_Y):
    return max(float(np.abs(plan_dense.sum(1) - mu_X).max()),
               float(np.abs(plan_dense.sum(0) - mu_Y).max()))


# ----------------------------------------------------------------------------- dense solver job
def _dense_job(args):
    """Dense auction on one instance. Returns (time, cost). Module level so a spawned child can
    run it; the instance is rebuilt from (N, seed, family, spec), nothing large is pickled."""
    N, seed, family, spec, accuracy = args
    X_pts, Y_pts, mu_X, mu_Y, C, C_max = make_instance(N, seed, family, spec)
    target = accuracy / float(mu_X.sum())
    t0 = time.perf_counter()
    with Capture(f"dense N={N} family={family} seed={seed}"):
        mgr = EpsScalingManager(
            AuctionOT, X_pts=X_pts, Y_pts=Y_pts, mu_X=mu_X, mu_Y=mu_Y,
            normalize=False, max_c=C_max, target_eps=target)
        dense_mu, _, _, _ = mgr.solve()
    elapsed = time.perf_counter() - t0

    P = np.zeros((len(X_pts), len(Y_pts)))
    for x in dense_mu:
        for y, m in dense_mu[x].items():
            P[x, y] = m
    if marginal_error(P, mu_X, mu_Y) > MARGINAL_TOL:
        raise RuntimeError("dense plan violates the marginals")
    return elapsed, float((P * C).sum())


def _dense_worker(args, queue):
    try:
        queue.put(("ok",) + _dense_job(args))
    except BaseException:
        queue.put(("error", traceback.format_exc()))


def run_dense(args, timeout):
    """Returns (status, time, cost). status is 'ok' or 'timeout'. With a timeout the solver runs
    in a child process that is terminated when the time is up (a censored result)."""
    if timeout is None:
        elapsed, cost = _dense_job(args)
        return "ok", elapsed, cost

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_dense_worker, args=(args, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return "timeout", float(timeout), NAN
    result = queue.get(timeout=30)
    if result[0] == "error":
        raise RuntimeError("dense solver failed in child process:\n" + result[1])
    return "ok", result[1], result[2]


# ----------------------------------------------------------------------------- one instance
def solve_instance(N, seed, family, accuracy, dense_max_n, dense_timeout, spec):
    """POT, hierarchical auction and (optionally) dense auction on one instance."""
    X_pts, Y_pts, mu_X, mu_Y, C, C_max = make_instance(N, seed, family, spec)
    M = float(mu_X.sum())
    target = accuracy / M
    tag = f"N={N} family={family} seed={seed}"

    # POT reference, timed including its own cost matrix
    t0 = time.perf_counter()
    C_pot = ot.dist(X_pts, Y_pts, metric="sqeuclidean")
    exact_cost, pot_log = ot.emd2(mu_X, mu_Y, C_pot, numItermax=POT_MAX_ITER, log=True)
    pot_time = time.perf_counter() - t0
    if pot_log.get("result_code") != 1:
        # POT stopped early: its cost is NOT optimal, so no gap can be computed.
        progress(f"{tag}: POT NOT OPTIMAL ({pot_log.get('warning')}); gap not available")
        exact_cost = NAN
    exact_cost = float(exact_cost)
    progress(f"{tag}: POT done ({pot_time:.2f}s)")

    # Hierarchical auction; tree construction is part of the timed work
    t0 = time.perf_counter()
    with Capture(f"hierarchical {tag}"):
        tree_X, tree_Y = build_matched_trees(X_pts, Y_pts)
        hier_solver = HierarchicalMultiscaleSolver(
            tree_X, tree_Y, mu_X, mu_Y, max_c=C_max, target_eps=target)
        plan = hier_solver.solve()
    hier_time = time.perf_counter() - t0
    hier_edges = len(hier_solver.last_N_guess)
    progress(f"{tag}: hierarchical done ({hier_time:.1f}s, {hier_edges} admissible pairs)")

    Ph = np.zeros((len(X_pts), len(Y_pts)))
    for x, y, m in plan:
        Ph[x, y] += m
    if marginal_error(Ph, mu_X, mu_Y) > MARGINAL_TOL:
        raise RuntimeError("hierarchical plan violates the marginals")
    hier_cost = float((Ph * C).sum())

    # Dense auction: skipped above dense_max_n, censored at dense_timeout seconds
    dense_status, dense_time, dense_cost = "skipped", NAN, NAN
    dense_cap = NAN if dense_timeout is None else float(dense_timeout)
    if N <= dense_max_n:
        progress(f"{tag}: running dense auction"
                 + (f" (cap {dense_timeout:.0f}s)" if dense_timeout else ""))
        dense_status, dense_time, dense_cost = run_dense(
            (N, seed, family, spec, accuracy), dense_timeout)
        progress(f"{tag}: dense {dense_status} ({dense_time:.1f}s)")
    else:
        progress(f"{tag}: dense skipped (N > {dense_max_n})")

    # Gaps and the proven bound  cost - OPT <= accuracy * C_max
    bound = accuracy * C_max

    def gap(cost):
        return (cost - exact_cost) / exact_cost * 100.0

    def bound_ok(cost):
        if np.isnan(cost) or np.isnan(exact_cost):
            return NAN
        return float(cost - exact_cost <= bound * (1.0 + BOUND_RTOL) + 1e-12)

    # speed-up of hierarchical over dense; for a timeout it is only a lower bound
    speedup = dense_time / hier_time if dense_status in ("ok", "timeout") else NAN

    return {
        "pot_time": pot_time, "dense_time": dense_time, "dense_status": dense_status,
        "dense_cap": dense_cap, "hier_time": hier_time, "speedup": speedup,
        "dense_gap": gap(dense_cost), "hier_gap": gap(hier_cost),
        "dense_bound_ok": bound_ok(dense_cost), "hier_bound_ok": bound_ok(hier_cost),
        "hier_edges": float(hier_edges),
    }


# ----------------------------------------------------------------------------- raw file I/O
def _done_pairs(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {(int(r["N"]), int(r["seed"])) for r in csv.DictReader(fh)}


def _append_raw(path, family, N, seed, r):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(RAW_HEADER)
        w.writerow([N, seed, family] + [r[k] for k in RAW_HEADER[3:]])


def run_benchmarks(family, label, spec, scales, seeds_override, accuracy,
                   dense_max_n, dense_timeout, fresh):
    rpath = raw_path(label)
    if fresh and os.path.exists(rpath):
        os.remove(rpath)
    write_metadata({"family": label, "dense_max_n": dense_max_n, "dense_timeout": dense_timeout,
                    "spec": spec})
    done = _done_pairs(rpath)

    for N in scales:
        seeds = list(seeds_override) if seeds_override else seeds_for(N, family)
        for seed in seeds:
            if (N, seed) in done:
                progress(f"N={N} family={label} seed={seed}: already in {rpath}, skipped")
                continue
            progress(f"N={N} family={label} seed={seed}: start")
            try:
                r = solve_instance(N, seed, family, accuracy, dense_max_n, dense_timeout, spec)
            except Exception:
                print(f"[N={N}, seed={seed}] FAILED:", file=sys.__stderr__)
                traceback.print_exc(file=sys.__stderr__)
                continue
            _append_raw(rpath, family, N, seed, r)
            if r["dense_bound_ok"] == 0.0 or r["hier_bound_ok"] == 0.0:
                progress(f"N={N} seed={seed}: WARNING proven bound violated")


# ----------------------------------------------------------------------------- aggregation
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def aggregate_raw(label):
    """Per-N means and standard deviations from the raw file. Timed-out dense runs enter the
    dense mean with their cap, so that mean is a lower bound (dense_censored > 0)."""
    path = raw_path(label)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run the benchmark first")
    by_n = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            by_n.setdefault(int(r["N"]), []).append(r)

    results = {"N": [], "n_seeds": [], "dense_censored": [], "bound_violations": []}
    for k in KEYS:
        results[k + "_mean"] = []
        results[k + "_std"] = []

    for N in sorted(by_n):
        rows = by_n[N]
        data = {k: [] for k in KEYS}
        censored = 0
        violations = 0
        for r in rows:
            status = r["dense_status"]
            if status == "ok":
                dense_val = _f(r["dense_time"])
            elif status == "timeout":
                dense_val = _f(r["dense_cap"])
                censored += 1
            else:
                dense_val = NAN
            data["pot_time"].append(_f(r["pot_time"]))
            data["dense_time"].append(dense_val)
            data["hier_time"].append(_f(r["hier_time"]))
            data["dense_gap"].append(_f(r["dense_gap"]) if status == "ok" else NAN)
            data["hier_gap"].append(_f(r["hier_gap"]))
            data["hier_edges"].append(_f(r["hier_edges"]))
            data["speedup"].append(_f(r["speedup"]))
            violations += sum(1 for b in (r["dense_bound_ok"], r["hier_bound_ok"])
                              if b not in ("", "nan") and _f(b) == 0.0)

        n = len(rows)
        ddof = 1 if n > 1 else 0
        results["N"].append(N)
        results["n_seeds"].append(n)
        results["dense_censored"].append(censored)
        results["bound_violations"].append(violations)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # all-NaN columns (skipped dense, POT not optimal)
            for k in KEYS:
                results[k + "_mean"].append(float(np.nanmean(data[k])))
                results[k + "_std"].append(float(np.nanstd(data[k], ddof=ddof)))
    return results


def write_summary_csv(results, label):
    with open(csv_path(label), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "seeds", "dense_censored", "bound_violations"]
                   + [f"{k}_{s}" for k in KEYS for s in ("mean", "std")])
        for i, N in enumerate(results["N"]):
            w.writerow([N, results["n_seeds"][i], results["dense_censored"][i],
                        results["bound_violations"][i]]
                       + [results[f"{k}_{s}"][i] for k in KEYS for s in ("mean", "std")])


# ----------------------------------------------------------------------------- printing
def print_table(results, label, accuracy=ACCURACY):
    header = (f"| {'N':<5} | {'Method':<13} | {'Time Mean':>11} | {'Time Std':>10} "
              f"| {'Gap Mean':>9} | {'Gap Std':>8} | {'Seeds':>5} |")
    rule = "-" * len(header)
    print()
    print(f"BACHELOR THESIS: OPTIMAL TRANSPORT SOLVER BENCHMARK   family = {label}")
    print(f"accuracy parameter: eps* = {accuracy:g} / M   (cost <= OPT + {accuracy:g} * C_max)")
    print(rule)
    print(header)
    print(rule)
    for i, N in enumerate(results["N"]):
        censored = results["dense_censored"][i]
        for name, k in (("POT EMD", "pot"), ("DENSE OT", "dense"), ("HIERARCH. OT", "hier")):
            t = results[f"{k}_time_mean"][i]
            t_txt = (">" if (k == "dense" and censored) else "") + f"{t:.4f}"
            gm = f"{results[f'{k}_gap_mean'][i]:>8.4f}%" if k != "pot" else f"{'-':>9}"
            gs = f"{results[f'{k}_gap_std'][i]:>7.4f}%" if k != "pot" else f"{'-':>8}"
            print(f"| {N:<5} | {name:<13} | {t_txt:>11} | {results[f'{k}_time_std'][i]:>10.4f} "
                  f"| {gm} | {gs} | {results['n_seeds'][i]:>5} |")
        note = f"|       bound violations: {results['bound_violations'][i]}"
        if censored:
            note += f"   dense timeouts: {censored} (dense time is a lower bound)"
        print(note)
        print(rule)


def _fit_exponent(N_arr, y):
    mask = (N_arr >= FIT_MIN_N) & np.isfinite(y) & (y > 0)
    if mask.sum() < 2:
        return NAN
    return float(np.polyfit(np.log(N_arr[mask]), np.log(y[mask]), 1)[0])


def print_summary(results, label):
    if len(results["N"]) < 2:
        return

    N_arr = np.array(results["N"], dtype=float)
    series = {name: np.array(results[name + "_time_mean"]) for name in ("pot", "dense", "hier")}

    print()
    print(f"SCALING SUMMARY   family = {label}")
    line = (f"| {'N':<5} | {'dense/hier':>10} | {'pot slope':>9} | {'dense slope':>11} "
            f"| {'hier slope':>10} | {'pairs |N|':>10} |")
    print("-" * len(line))
    print(line)
    print("-" * len(line))

    with np.errstate(all="ignore"):
        for i, N in enumerate(results["N"]):
            ratio = series["dense"][i] / series["hier"][i]
            if i == 0:
                slopes = ["", "", ""]
            else:
                step = np.log(N_arr[i] / N_arr[i - 1])
                slopes = [f"{np.log(series[n][i] / series[n][i - 1]) / step:.2f}"
                          for n in ("pot", "dense", "hier")]
            edges = results["hier_edges_mean"][i]
            print(f"| {N:<5} | {ratio:>10.2f} | {slopes[0]:>9} | {slopes[1]:>11} "
                  f"| {slopes[2]:>10} | {edges:>10.0f} |")

    print("-" * len(line))
    for name, text in (("pot", "POT"), ("dense", "dense"), ("hier", "hierarchical")):
        fit = _fit_exponent(N_arr, series[name])
        print(f"Fitted exponent, {text} (N >= {FIT_MIN_N}): N^{fit:.2f}")
    if sum(results["dense_censored"]):
        print("Note: dense runs that hit the time cap enter with the cap, so the dense "
              "exponent and the ratio dense/hier are lower bounds.")
    print(f"Total violations of the proven bound: {sum(results['bound_violations'])}")
    print(f"Summary written to {csv_path(label)}")


# ----------------------------------------------------------------------------- plots
def plot_results(results, label):
    if len(results["N"]) < 2:
        return

    N_arr = np.array(results["N"], dtype=float)
    censored = np.array(results["dense_censored"]) > 0

    plt.figure(figsize=(8, 6))
    for name, text, marker in [("pot", "POT (EMD)", "D"),
                               ("dense", "Dense Auction", "o"),
                               ("hier", "Hierarchical Auction", "s")]:
        mean = np.array(results[name + "_time_mean"])
        std = np.array(results[name + "_time_std"])
        plt.loglog(N_arr, mean, marker=marker, label=text, linewidth=2)
        plt.fill_between(N_arr, np.maximum(mean - std, 1e-9), mean + std, alpha=0.2)
    if censored.any():
        dense = np.array(results["dense_time_mean"])
        plt.loglog(N_arr[censored], dense[censored], linestyle="none", marker="^", ms=12,
                   mfc="none", mec="k", label="dense: time cap reached (lower bound)")

    hier = np.array(results["hier_time_mean"])
    idx = np.where((N_arr >= FIT_MIN_N) & np.isfinite(hier))[0]
    if len(idx):
        n0, t0 = N_arr[idx[0]], hier[idx[0]]
        for p, style in ((1, ":"), (2, "--"), (3, "-.")):
            plt.loglog(N_arr, t0 * (N_arr / n0) ** p, style, color="gray", alpha=0.6,
                       label=f"$N^{p}$")

    plt.title(f"Computation Time vs Problem Size ({label})")
    plt.xlabel("Number of Points (N)")
    plt.ylabel("Time (seconds)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"thesis_scaling_plot_{label}.pdf")
    plt.close()

    # speed-up of the hierarchical over the dense auction
    speed = np.array(results["speedup_mean"])
    if np.isfinite(speed).sum() >= 2:
        plt.figure(figsize=(8, 5))
        plt.loglog(N_arr, speed, marker="o", linewidth=2, label="dense time / hierarchical time")
        if censored.any():
            plt.loglog(N_arr[censored], speed[censored], linestyle="none", marker="^", ms=12,
                       mfc="none", mec="k", label="dense hit the time cap (lower bound)")
        plt.axhline(1.0, color="gray", ls="--", label="break-even")
        plt.title(f"Speed-up of the Hierarchical Auction ({label})")
        plt.xlabel("Number of Points (N)")
        plt.ylabel("Speed-up factor")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"thesis_speedup_plot_{label}.pdf")
        plt.close()

    # sparsity of the final admissible set
    edges = np.array(results["hier_edges_mean"])
    if np.isfinite(edges).sum() >= 2:
        plt.figure(figsize=(8, 5))
        plt.loglog(N_arr, edges, marker="s", linewidth=2, label="admissible pairs |N| (final level)")
        e0, n0 = edges[np.isfinite(edges)][0], N_arr[np.isfinite(edges)][0]
        plt.loglog(N_arr, e0 * (N_arr / n0), ":", color="gray", label="$N$")
        plt.loglog(N_arr, e0 * (N_arr / n0) ** 2, "--", color="gray", label="$N^2$ (dense)")
        plt.title(f"Sparsity of the Hierarchical Solver ({label})")
        plt.xlabel("Number of Points (N)")
        plt.ylabel("Number of pairs")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"thesis_sparsity_plot_{label}.pdf")
        plt.close()

    plt.figure(figsize=(8, 6))
    for name, text, marker in [("dense", "Dense", "o"), ("hier", "Hierarchical", "s")]:
        mean = np.array(results[name + "_gap_mean"])
        std = np.array(results[name + "_gap_std"])
        plt.semilogx(N_arr, mean, marker=marker, label=text, linewidth=2)
        plt.fill_between(N_arr, mean - std, mean + std, alpha=0.2)
    plt.title(f"Optimality Gap vs Exact Reference (POT), {label}")
    plt.xlabel("Number of Points (N)")
    plt.ylabel("Relative Gap (%)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"thesis_gap_plot_{label}.pdf")
    plt.close()


# ----------------------------------------------------------------------------- accuracy sweep
def run_sweep(family, label, spec, N, accuracies, seeds, dense_max_n, dense_timeout):
    """Runtime and gap against the accuracy parameter at fixed N."""
    write_metadata({"family": label, "sweep_N": N, "dense_max_n": dense_max_n,
                    "dense_timeout": dense_timeout})
    sweep_header = ["family", "N", "accuracy", "seed", "pot_time", "dense_time", "dense_status",
                    "hier_time", "dense_gap", "hier_gap", "dense_bound_ok", "hier_bound_ok"]
    with open(SWEEP_PATH, "w", newline="") as fh:
        csv.writer(fh).writerow(sweep_header)

    print()
    print(f"ACCURACY SWEEP   family = {label}, N = {N}")
    line = (f"| {'accuracy':>9} | {'pot [s]':>9} | {'dense [s]':>10} | {'hier [s]':>9} "
            f"| {'dense gap %':>11} | {'hier gap %':>10} |")
    print("-" * len(line))
    print(line)
    print("-" * len(line))

    table = []
    for acc in accuracies:
        rows = []
        for seed in seeds:
            progress(f"sweep accuracy={acc:g} seed={seed}")
            try:
                r = solve_instance(N, seed, family, acc, dense_max_n, dense_timeout, spec)
            except Exception:
                print(f"[sweep accuracy={acc:g}, seed={seed}] FAILED:", file=sys.__stderr__)
                traceback.print_exc(file=sys.__stderr__)
                continue
            rows.append(r)
            with open(SWEEP_PATH, "a", newline="") as fh:
                csv.writer(fh).writerow([label, N, acc, seed] + [r[k] for k in sweep_header[4:]])
        if not rows:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = {k: float(np.nanmean([r[k] for r in rows]))
                 for k in ("pot_time", "dense_time", "hier_time", "dense_gap", "hier_gap")}
        table.append((acc, m))
        print(f"| {acc:>9.0e} | {m['pot_time']:>9.3f} | {m['dense_time']:>10.3f} "
              f"| {m['hier_time']:>9.3f} | {m['dense_gap']:>11.5f} | {m['hier_gap']:>10.5f} |")
    print("-" * len(line))

    if len(table) >= 2:
        acc_arr = np.array([a for a, _ in table])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for name, text, marker in [("pot_time", "POT (exact, no accuracy parameter)", "D"),
                                   ("dense_time", "Dense auction", "o"),
                                   ("hier_time", "Hierarchical auction", "s")]:
            ax[0].loglog(acc_arr, [m[name] for _, m in table], marker=marker, label=text)
        ax[0].set_xlabel("accuracy parameter")
        ax[0].set_ylabel("time (s)")
        ax[0].set_title(f"Runtime vs accuracy (N = {N})")
        ax[0].grid(True, which="both", ls="--", alpha=0.5)
        ax[0].legend()
        for name, text, marker in [("dense_gap", "Dense auction", "o"),
                                   ("hier_gap", "Hierarchical auction", "s")]:
            ax[1].loglog(acc_arr, [max(m[name], 1e-12) for _, m in table],
                         marker=marker, label=text)
        ax[1].set_xlabel("accuracy parameter")
        ax[1].set_ylabel("relative gap (%)")
        ax[1].set_title("Gap vs accuracy")
        ax[1].grid(True, which="both", ls="--", alpha=0.5)
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(f"thesis_sweep_plot_{label}.pdf")
        plt.close(fig)
    print(f"Sweep written to {SWEEP_PATH}")


# ----------------------------------------------------------------------------- command line
def _parse_glyph(text):
    text = text.strip()
    if text.upper().startswith("U+"):
        return chr(int(text[2:], 16))
    return text[0]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="OT solver benchmark",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--family", default=DEFAULT_FAMILY, choices=list(FAMILIES) + ["all"],
                    help=f"instance family (default: {DEFAULT_FAMILY})")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    ap.add_argument("--scales", type=int, nargs="+", help="only these N (default: SCALES)")
    ap.add_argument("--seeds", type=int, nargs="+", help="only these seeds")
    ap.add_argument("--accuracy", type=float, default=ACCURACY,
                    help="accuracy parameter, eps* = accuracy / M")
    ap.add_argument("--dense-max-n", type=int, default=DENSE_MAX_N_DEFAULT,
                    help="run the dense auction only for N up to this value "
                         f"(default {DENSE_MAX_N_DEFAULT}; 0 switches it off)")
    ap.add_argument("--dense-timeout", type=float, default=None,
                    help="seconds after which a dense run is stopped and recorded as censored")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the raw file of this family before running")
    ap.add_argument("--plot-only", action="store_true",
                    help="no solving: tables and plots from the raw file")
    ap.add_argument("--sweep", action="store_true",
                    help=f"accuracy sweep at N = {SWEEP_N} instead of the scaling run")
    ap.add_argument("--font", default=None, help="CJK font file for --family glyph")
    ap.add_argument("--glyphs", nargs=2, default=None, metavar=("X", "Y"),
                    help="two characters or code points (for example U+9F99 U+51E4)")
    ap.add_argument("--image-x", default=None, help="image file for --family image")
    ap.add_argument("--image-y", default=None, help="image file for --family image")
    args = ap.parse_args()

    spec = {"font": args.font, "image_x": args.image_x, "image_y": args.image_y}
    if args.glyphs:
        spec["glyph_x"], spec["glyph_y"] = (_parse_glyph(g) for g in args.glyphs)
    else:
        spec["glyph_x"], spec["glyph_y"] = GLYPH_DEFAULT

    families = [f for f in FAMILIES if f != "uniform_legacy"] if args.family == "all" \
        else [args.family]
    scales = args.scales if args.scales else SCALES

    for fam in families:
        label = fam + (f"_{args.tag}" if args.tag else "")
        if args.sweep:
            run_sweep(fam, label, spec, SWEEP_N, SWEEP_ACCURACIES,
                      args.seeds or SEEDS_LARGE, args.dense_max_n, args.dense_timeout)
            continue
        if not args.plot_only:
            run_benchmarks(fam, label, spec, scales, args.seeds, args.accuracy,
                           args.dense_max_n, args.dense_timeout, args.fresh)
        final_results = aggregate_raw(label)
        write_summary_csv(final_results, label)
        print_table(final_results, label, args.accuracy)
        print_summary(final_results, label)
        plot_results(final_results, label)
