"""
diagnostics.py
Run ONE check at a time from the repo root (same folder as run_benchmarks.py):

    python diagnostics.py t1      # effective beta + eps-CS of the dense auction
    python diagnostics.py t2      # marginals of dense and hierarchical plans
    python diagnostics.py t3      # cost gap vs ot.emd2 for leaf size k = 1 and k = 8
    python diagnostics.py t4      # eps-CS of the final multiscale plan on the FULL matrix

All costs are checked in the solver's normalised units: C / global_max_c.
"""
import sys
import numpy as np
import ot

from src.core.ot_auction import AuctionOT
from src.utils.eps_scaling import EpsScalingManager
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver
from experiments.run_benchmarks import make_instance, build_matched_trees

TOL = 1e-7


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
    """Largest amount by which eps-CS is violated on any used pair.
    <= ~1e-7 means eps-CS holds. Uses ALL targets in the min (full matrix)."""
    net = C_norm - beta[None, :]
    best = net.min(axis=1)
    used = P > TOL
    viol = np.where(used, net - best[:, None] - eps, -np.inf)
    return float(viol.max())


def t1(N=32, seed=40, eps=1e-3):
    X, Y, mu_X, mu_Y, C, cmax = make_instance(N, seed)
    s = AuctionOT(X, Y, mu_X, mu_Y, epsilon=eps, normalize=False, max_c=cmax)
    mu, cost, iters = s.solve()
    beta = s.get_effective_beta()
    P = dict_to_dense(mu, N, N)
    print(f"iterations           : {iters}")
    print(f"max |effective beta| : {np.abs(beta).max():.3e}   (0 means the duals are being thrown away)")
    print(f"beta_diamond max abs : {np.abs(s.beta_diamond).max():.3e}")
    print(f"eps-CS violation     : {eps_cs_violation(C / cmax, P, beta, eps):.3e}   (should be <= 1e-7)")


def t2(N=64, seed=40):
    X, Y, mu_X, mu_Y, C, cmax = make_instance(N, seed)
    tight = 0.5 / (N + 1)

    mgr = EpsScalingManager(AuctionOT, X_pts=X, Y_pts=Y, mu_X=mu_X, mu_Y=mu_Y,
                            normalize=False, max_c=cmax, target_eps=tight, min_eps=1e-5)
    mu, _, _, _ = mgr.solve()
    Pd = dict_to_dense(mu, N, N)
    print(f"dense : max row err {np.abs(Pd.sum(1) - mu_X).max():.2e}, "
          f"max col err {np.abs(Pd.sum(0) - mu_Y).max():.2e}")

    for k in (1, 8):
        tx, ty = build_matched_trees(X, Y, max_points_per_cell=k)
        sol = HierarchicalMultiscaleSolver(tx, ty, mu_X, mu_Y, max_c=cmax,
                                           target_eps=tight, min_eps=1e-5)
        Ph = list_to_dense(sol.solve(), N, N)
        print(f"hier k={k}: max row err {np.abs(Ph.sum(1) - mu_X).max():.2e}, "
              f"max col err {np.abs(Ph.sum(0) - mu_Y).max():.2e}")


def t3(N=64, seeds=(40, 50, 100)):
    for seed in seeds:
        X, Y, mu_X, mu_Y, C, cmax = make_instance(N, seed)
        exact = ot.emd2(mu_X, mu_Y, C)
        tight = 0.5 / (N + 1)
        for k in (1, 8):
            tx, ty = build_matched_trees(X, Y, max_points_per_cell=k)
            sol = HierarchicalMultiscaleSolver(tx, ty, mu_X, mu_Y, max_c=cmax,
                                               target_eps=tight, min_eps=1e-5)
            trip = sol.solve()
            cost = sum(m * C[x, y] for x, y, m in trip)
            print(f"seed {seed} k={k}: gap {(cost - exact) / exact * 100:+.4f}%  "
                  f"(negative gap = marginals violated)")


def t4(N=64, seed=40):
    """Needs two lines at the END of HierarchicalMultiscaleSolver.solve(), before the return:
           self.last_mu = current_mu
           self.last_beta = final_beta
    Only meaningful for k = 1 (every leaf is a single point)."""
    X, Y, mu_X, mu_Y, C, cmax = make_instance(N, seed)
    tx, ty = build_matched_trees(X, Y, max_points_per_cell=1)
    eps = 0.5 / (N + 1)
    sol = HierarchicalMultiscaleSolver(tx, ty, mu_X, mu_Y, max_c=cmax,
                                       target_eps=eps, min_eps=1e-5)
    sol.solve()
    fx = tx.get_active_cells_at_depth(sol.max_depth)
    fy = ty.get_active_cells_at_depth(sol.max_depth)
    assert all(len(c.point_indices) == 1 for c in fx + fy), "leaves are not singletons"
    P = np.zeros((N, N))
    beta_pt = np.zeros(N)
    for i in sol.last_mu:
        for j, m in sol.last_mu[i].items():
            P[fx[i].point_indices[0], fy[j].point_indices[0]] += m
    for j, cell in enumerate(fy):
        beta_pt[cell.point_indices[0]] = sol.last_beta[j]
    print(f"max |beta|       : {np.abs(beta_pt).max():.3e}")
    print(f"eps-CS violation : {eps_cs_violation(C / cmax, P, beta_pt, eps):.3e}   (should be <= 1e-7)")

def t1b(N=32, seed=40, eps=1e-3, top=5):
    X, Y, mu_X, mu_Y, C, cmax = make_instance(N, seed)
    s = AuctionOT(X, Y, mu_X, mu_Y, epsilon=eps, normalize=False, max_c=cmax)
    mu, cost, iters = s.solve()
    beta = s.get_effective_beta()
    Cn = C / cmax
    P = dict_to_dense(mu, N, N)
    print("unassigned left :", s.unassigned_X.sum())
    print("max row err     :", np.abs(P.sum(1) - mu_X).max(),
          " max col err:", np.abs(P.sum(0) - mu_Y).max())
    net = Cn - beta[None, :]
    best = net.min(axis=1)
    rows = []
    for x in range(N):
        for y in range(N):
            if P[x, y] > TOL:
                rows.append((net[x, y] - best[x] - eps, x, y))
    rows.sort(reverse=True)
    n_viol = sum(1 for r in rows if r[0] > 1e-7)
    print(f"{n_viol} of {len(rows)} used pairs violate eps-CS")
    for v, x, y in rows[:top]:
        yb = int(np.argmin(net[x]))
        print(f"x={x} y={y} viol={v:.3e} | y full={s.assigned_Y[y] >= s.mu_Y[y] - TOL}"
              f" owners={len(s._get_active_xs_for_y(y))}"
              f" own btilde={s._get_beta_tilde(x, y):.4f} eff beta={beta[y]:.4f}"
              f" net_used={net[x, y]:.4f}"
              f" | best y*={yb} net*={best[x]:.4f}"
              f" y* full={s.assigned_Y[yb] >= s.mu_Y[yb] - TOL} eff beta*={beta[yb]:.4f}")


def t1c(N=32, seed=40, eps=1e-3):
    X, Y, mu_X, mu_Y, C, cmax = make_instance(N, seed)
    Cn = C / cmax
    s = AuctionOT(X, Y, mu_X, mu_Y, epsilon=eps, normalize=False, max_c=cmax)
    st = {"log": [], "prev_x": None, "done": False}
    orig_place, orig_slots = s._place, s._sorted_slots

    def place(x, y, owner_xp, amount, nbt):
        got = orig_place(x, y, owner_xp, amount, nbt)
        if got > 0:
            st["log"].append((x, y, int(owner_xp), round(got, 4), round(nbt, 5)))
            st["prev_x"] = x
        return got

    def slots(x):
        if not st["done"] and st["log"]:
            P = np.array([[s._get_mu(a, b) for b in range(N)] for a in range(N)])
            beta = s.get_effective_beta()
            net = Cn - beta[None, :]
            best = net.min(axis=1)
            viol = np.where(P > TOL, net - best[:, None] - eps, -np.inf)
            if viol.max() > 1e-7:
                st["done"] = True
                bx, by = np.unravel_index(np.argmax(viol), viol.shape)
                print(f"FIRST VIOLATION before buyer x={x}'s turn, previous bidder = {st['prev_x']}")
                print(f"  violating pair x={bx} y={by}: viol={viol[bx, by]:.3e}")
                print(f"  placements of previous bidder (x, y, owner_xp, amount, new_btilde):")
                for rec in st["log"]:
                    print("   ", rec, f" implied alpha' = {Cn[rec[0], rec[1]] - rec[4] - eps:.5f}")
                yb = int(np.argmin(net[bx]))
                print(f"  best target y*={yb}: eff beta={beta[yb]:.5f}, "
                      f"owners={[(o, round(s._get_beta_tilde(o, yb), 5)) for o in s._get_active_xs_for_y(yb)]}")
                print(f"  used target y={by}: btilde(x,y)={s._get_beta_tilde(bx, by):.5f}, eff beta={beta[by]:.5f}")
            st["log"] = []
        return orig_slots(x)

    s._place, s._sorted_slots = place, slots
    s.solve()
    if not st["done"]:
        print("eps-CS held before every turn; the violation appears only after the last bid.")


if __name__ == "__main__":
    {"t1": t1, "t1b": t1b, "t1c": t1c, "t2": t2, "t3": t3, "t4": t4}[sys.argv[1]]()
