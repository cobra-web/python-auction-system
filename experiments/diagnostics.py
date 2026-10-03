"""
diagnostics.py: correctness checks for the auction solvers. Run from the repo root (BA\\Python):

    python -m experiments.diagnostics            # runs all checks
    python -m experiments.diagnostics t3         # runs one check

    t1  dense auction: effective beta is not discarded, eps-CS holds on the FULL matrix
    t2  marginals of dense and hierarchical plans (also unbalanced sizes n != m for the dense one)
    t3  cost gap against POT, for shrinking eps (must go to 0 for both solvers)
    t4  eps-CS of the final hierarchical plan on the FULL matrix (soundness of the pruning)
    t5  forced miss: delete edges that the optimal plan needs, the consistency loop must re-add them
    t6  LAP auction against scipy.optimize.linear_sum_assignment

All costs are checked in the solver's normalised units: C / max_c.
"""
import contextlib
import io
import sys

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment

from src.core.ot_auction import AuctionOT
from src.core.lap_auction import AuctionLAP
from src.utils.eps_scaling import EpsScalingManager
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver
from experiments.run_benchmarks import make_instance, build_matched_trees

TOL = 1e-7
FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILED.append(name)


def dict_to_dense(mu_dict, n, m):
    P = np.zeros((n, m))
    for x in mu_dict:
        for y, mass in mu_dict[x].items():
            P[x, y] = mass
    return P


def list_to_dense(triples, n, m):
    P = np.zeros((n, m))
    for x, y, mass in triples:
        P[x, y] += mass
    return P


def eps_cs_violation(C_norm, P, beta, eps):
    """Largest amount by which eps-CS is violated on any used pair (min over ALL targets)."""
    net = C_norm - beta[None, :]
    best = net.min(axis=1)
    viol = np.where(P > TOL, net - best[:, None] - eps, -np.inf)
    return float(viol.max())


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def t1(sizes=(16, 32, 64), seeds=(40, 50, 100), eps=1e-3):
    print("t1: dense auction, eps-CS on the full matrix")
    for N in sizes:
        for seed in seeds:
            X, Y, mu_X, mu_Y, C, cmax = make_instance(N, seed)
            s = AuctionOT(X, Y, mu_X, mu_Y, epsilon=eps, normalize=False, max_c=cmax)
            mu, _, _ = s.solve()
            beta = s.get_effective_beta()
            P = dict_to_dense(mu, N, N)
            v = eps_cs_violation(C / cmax, P, beta, eps)
            check(f"N={N} seed={seed}", v <= TOL and np.abs(beta).max() > 0,
                  f"violation {v:.1e}, max|beta| {np.abs(beta).max():.3f}")


def t2(N=64, seed=40):
    print("t2: marginals")
    X, Y, mu_X, mu_Y, C, cmax = make_instance(N, seed)

    with quiet():
        mgr = EpsScalingManager(AuctionOT, X_pts=X, Y_pts=Y, mu_X=mu_X, mu_Y=mu_Y,
                                normalize=False, max_c=cmax)
        mu, _, _, _ = mgr.solve()
    Pd = dict_to_dense(mu, N, N)
    check("dense", np.abs(Pd.sum(1) - mu_X).max() < 1e-6 and np.abs(Pd.sum(0) - mu_Y).max() < 1e-6)

    with quiet():
        tx, ty = build_matched_trees(X, Y)
        plan = HierarchicalMultiscaleSolver(tx, ty, mu_X, mu_Y, max_c=cmax).solve()
    Ph = list_to_dense(plan, N, N)
    check("hierarchical (k=1)",
          np.abs(Ph.sum(1) - mu_X).max() < 1e-6 and np.abs(Ph.sum(0) - mu_Y).max() < 1e-6)

    try:
        tx8, ty8 = build_matched_trees(X, Y, max_points_per_cell=8, max_allowed_depth=2)
        HierarchicalMultiscaleSolver(tx8, ty8, mu_X, mu_Y, max_c=cmax)
        check("leaves with several points are rejected", False)
    except ValueError:
        check("leaves with several points are rejected", True)

    # unbalanced sizes n != m, dense solver
    rng = np.random.RandomState(1)
    n, m = 40, 25
    Xu, Yu = rng.rand(n, 2), rng.rand(m, 2)
    a = rng.randint(1, 6, n).astype(float)
    b = rng.randint(1, 6, m).astype(float)
    if a.sum() > b.sum():
        b[0] += a.sum() - b.sum()
    else:
        a[0] += b.sum() - a.sum()
    Cu = ot.dist(Xu, Yu)
    with quiet():
        mgr = EpsScalingManager(AuctionOT, X_pts=Xu, Y_pts=Yu, mu_X=a, mu_Y=b,
                                normalize=True, accuracy=1e-4)
        mu, _, _, _ = mgr.solve()
    Pu = dict_to_dense(mu, n, m)
    exact = ot.emd2(a, b, Cu)
    gap = abs((Pu * Cu).sum() - exact) / exact
    check("dense, n != m", np.abs(Pu.sum(1) - a).max() < 1e-6 and np.abs(Pu.sum(0) - b).max() < 1e-6
          and gap < 1e-3, f"gap {gap:.2e}")


def t3(N=64, seeds=(40, 50, 100)):
    print("t3: gap against POT for shrinking eps (accuracy parameter)")
    for seed in seeds:
        X, Y, mu_X, mu_Y, C, cmax = make_instance(N, seed)
        exact = ot.emd2(mu_X, mu_Y, C)
        M = mu_X.sum()
        for acc in (1.0, 1e-2, 1e-3):
            eps = acc / M
            with quiet():
                mgr = EpsScalingManager(AuctionOT, X_pts=X, Y_pts=Y, mu_X=mu_X, mu_Y=mu_Y,
                                        normalize=False, max_c=cmax, target_eps=eps)
                mu, _, _, _ = mgr.solve()
                tx, ty = build_matched_trees(X, Y)
                plan = HierarchicalMultiscaleSolver(tx, ty, mu_X, mu_Y, max_c=cmax,
                                                    target_eps=eps).solve()
            cd = sum(m * C[x, y] for x in mu for y, m in mu[x].items())
            ch = sum(m * C[x, y] for x, y, m in plan)
            bound = acc * cmax
            ok = cd - exact <= bound + 1e-9 and ch - exact <= bound + 1e-9 \
                and cd >= exact - 1e-9 and ch >= exact - 1e-9
            check(f"seed {seed} accuracy {acc:g}", ok,
                  f"dense gap {(cd - exact) / exact * 100:+.4f}%, hier gap {(ch - exact) / exact * 100:+.4f}%, "
                  f"bound {bound / exact * 100:.3f}%")


def t4(N=64, seeds=(40, 50)):
    print("t4: eps-CS of the final hierarchical plan on the full matrix")
    for seed in seeds:
        X, Y, mu_X, mu_Y, C, cmax = make_instance(N, seed)
        eps = 1e-3 / mu_X.sum()
        with quiet():
            tx, ty = build_matched_trees(X, Y)
            sol = HierarchicalMultiscaleSolver(tx, ty, mu_X, mu_Y, max_c=cmax, target_eps=eps)
            sol.solve()
        fx = tx.get_active_cells_at_depth(sol.max_depth)
        fy = ty.get_active_cells_at_depth(sol.max_depth)
        P = np.zeros((N, N))
        beta_pt = np.zeros(N)
        for i in sol.last_mu:
            for j, m in sol.last_mu[i].items():
                P[fx[i].point_indices[0], fy[j].point_indices[0]] += m
        for j, cell in enumerate(fy):
            beta_pt[cell.point_indices[0]] = sol.last_beta[j]
        v = eps_cs_violation(C / cmax, P, beta_pt, sol.target_eps)
        check(f"seed {seed}", v <= TOL, f"violation {v:.1e}, max|beta| {np.abs(beta_pt).max():.3f}")


def t5(N=48, seed=40):
    print("t5: bad but feasible neighbourhood, the consistency loop must repair it")
    X, Y, mu_X, mu_Y, C, cmax = make_instance(N, seed)
    eps = 1e-3 / mu_X.sum()
    exact = ot.emd2(mu_X, mu_Y, C)
    rng = np.random.RandomState(7)
    with quiet():
        tx, ty = build_matched_trees(X, Y)
        sol = HierarchicalMultiscaleSolver(tx, ty, mu_X, mu_Y, max_c=cmax, target_eps=eps)
        original = sol._induce_sparse_neighborhood

        def sabotaged(mu_hat, cells_X_coarse, cells_Y_coarse, cells_X_fine, cells_Y_fine, **kw):
            # replace the coarse plan by an optimal plan for RANDOM costs: feasible, sparse, but wrong
            a = np.array([mu_X[c.point_indices].sum() for c in cells_X_coarse])
            b = np.array([mu_Y[c.point_indices].sum() for c in cells_Y_coarse])
            G = ot.emd(a, b, rng.rand(len(a), len(b)))
            bad = {i: {j: G[i, j] for j in range(len(b)) if G[i, j] > 1e-9} for i in range(len(a))}
            return original(bad, cells_X_coarse, cells_Y_coarse, cells_X_fine, cells_Y_fine, **kw)

        sol._induce_sparse_neighborhood = sabotaged
        plan = sol.solve()
    cost = sum(m * C[x, y] for x, y, m in plan)
    check("gap stays within the eps bound", -1e-9 <= cost - exact <= 1e-3 * cmax + 1e-9,
          f"gap {(cost - exact) / exact * 100:+.4f}%, final edge set {len(sol.last_N_guess)} "
          f"of {N * N}")


def t6(sizes=(1, 2, 5, 30, 100), seeds=(0, 1, 2)):
    print("t6: LAP auction against scipy (integer costs, eps < 1/N gives the exact optimum)")
    for N in sizes:
        for seed in seeds:
            rng = np.random.RandomState(seed)
            C = rng.randint(0, 50, size=(N, N)).astype(float)
            _, cost, _ = AuctionLAP(C, epsilon=1.0 / (N + 1)).solve()
            r, c = linear_sum_assignment(C)
            check(f"N={N} seed={seed}", abs(cost - C[r, c].sum()) < 1e-9,
                  f"auction {cost:.0f}, scipy {C[r, c].sum():.0f}")


ALL = {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5, "t6": t6}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        ALL[n]()
    print()
    print("ALL PASSED" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
