import time
import os
import sys
import numpy as np
from src.utils.cost_functions import squared_euclidean
from src.utils.eps_scaling import EpsScalingManager
from src.core.ot_auction import AuctionOT
from src.core.lap_auction import AuctionLAP

from src.hierarchical.partitions import HierarchicalPartition
from src.hierarchical.multiscale_solver import HierarchicalMultiscaleSolver

class SilencePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def build_matched_trees(X_pts, Y_pts, max_points_per_cell=8, max_allowed_depth=10):
    probe_X = HierarchicalPartition(X_pts, max_points_per_cell=max_points_per_cell,
                                     max_allowed_depth=max_allowed_depth)
    probe_Y = HierarchicalPartition(Y_pts, max_points_per_cell=max_points_per_cell,
                                     max_allowed_depth=max_allowed_depth)

    target_g = max(probe_X.g, probe_Y.g)
    
    # Erneuter Aufbau mit gematchter Tiefe
    tree_X = HierarchicalPartition(X_pts, max_points_per_cell=max_points_per_cell,
                                   max_allowed_depth=target_g)
    tree_Y = HierarchicalPartition(Y_pts, max_points_per_cell=max_points_per_cell,
                                   max_allowed_depth=target_g)
    return tree_X, tree_Y


def run_comprehensive_benchmarks():
    print("\n================================================================================")
    print("BACHELOR THESIS: OPTIMAL TRANSPORT SOLVER BENCHMARK (WITH PROFESSOR DIAGNOSTICS)")
    print("================================================================================\n")
    
    # Test-Skalen (N Punkte)
    scales = [16, 32, 64, 128]
    
    for N in scales:
        print("-" * 80)
        print(f"STARTE DIAGNOSE FÜR SKALA N = {N}")
        print("-" * 80)
        
        # 1. Daten-Generierung (Gleichmäßige Verteilung im Raum)
        np.random.seed(42) # Reproduzierbarkeit
        X_pts = np.random.rand(N, 2)
        Y_pts = np.random.rand(N, 2)
        
        # Generierung von zufälligen Integer-Massen (z.B. zwischen 1 und 5)
        mu_X = np.random.randint(1, 6, size=N)
        mu_Y = np.random.randint(1, 6, size=N)
        
        # Massen-Ausgleich erzwingen, da OT balanciert sein muss
        diff = np.sum(mu_X) - np.sum(mu_Y)
        if diff > 0:
            mu_Y[0] += diff
        elif diff < 0:
            mu_X[0] += abs(diff)
            
        total_problem_mass = np.sum(mu_X)
        
        # Originale, unskalierte Distanzmatrix berechnen
        C = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                C[i, j] = np.sum((X_pts[i] - Y_pts[j])**2)
                
        # --- BENCHMARK 1: REINES LAP AUCTION (Referenz für 1-zu-1 Matching) ---
        try:
            t0 = time.perf_counter()
            with SilencePrints():
                # LAP ignoriert Gewichte per Definition
                lap_solver = AuctionLAP(C)
                lap_assignment = lap_solver.solve()
            t_lap = time.perf_counter() - t0
            
            # Rekonstruktion der Matrix aus dem LAP-Vektor für den manuellen Check
            lap_mu = np.zeros_like(C)
            for x, y in enumerate(lap_assignment):
                if y != -1: lap_mu[x, y] = 1.0
                
            manual_lap_cost = np.sum(lap_mu * C)
            print(f"[LAP Reference]   Zeit: {t_lap:.5f}s | Gemeldete Kosten: {manual_lap_cost:.4f}")
        except Exception as e:
            print(f"[LAP Reference]   FEHLGESCHLAGEN")
            
        # --- BENCHMARK 2: DENSE OT SOLVER ---
        dense_mu = None
        try:
            t1 = time.perf_counter()
            with SilencePrints():
                dense_manager = EpsScalingManager(AuctionOT, C, mu_X=mu_X, mu_Y=mu_Y)
                dense_mu, dense_reported_cost, _, _ = dense_manager.solve()
            t_dense = time.perf_counter() - t1
            
            # --- PROFESSOR DIAGNOSTICS FÜR DENSE ---
            manual_dense_cost = np.sum(dense_mu * C)
            actual_dense_mass = np.sum(dense_mu)
            active_dense_pairs = np.sum(dense_mu > 1e-5)
            
            print(f"\n[DENSE OT RESULT]")
            print(f"  -> Runtime: {t_dense:.5f}s")
            print(f"  -> Gemeldete Solver-Kosten:    {dense_reported_cost:.4f}")
            print(f"  -> Manuell nachgerechnete Kosten: {manual_dense_cost:.4f}  <-- Check 3")
            print(f"  -> Transportierte Gesamtmasse:   {actual_dense_mass:.2f} (Soll: {total_problem_mass}) <-- Check 2")
            print(f"  -> Anzahl aktiver Paare (>1e-5): {active_dense_pairs} (Theor. Max: {2*N-1}) <-- Check 1")
        except Exception as e:
            print(f"\n[DENSE OT] FEHLGESCHLAGEN")
            import traceback; traceback.print_exc()

        # --- BENCHMARK 3: HIERARCHICAL MULTISCALE SOLVER ---
        try:
            tree_X, tree_Y = build_matched_trees(X_pts, Y_pts)

            t2 = time.perf_counter()
            with SilencePrints():
                multiscale_solver = HierarchicalMultiscaleSolver(tree_X, tree_Y, C, mu_X, mu_Y)
                hier_mu = multiscale_solver.solve()
            t_hier = time.perf_counter() - t2
            
            # --- PROFESSOR DIAGNOSTICS FÜR HIERARCHICAL ---
            manual_hier_cost = np.sum(hier_mu * C)
            actual_hier_mass = np.sum(hier_mu)
            active_hier_pairs = np.sum(hier_mu > 1e-5)
            
            print(f"\n[HIERARCHICAL OT RESULT]")
            print(f"  -> Runtime: {t_hier:.5f}s")
            print(f"  -> Manuell nachgerechnete Kosten: {manual_hier_cost:.4f}  <-- Check 3")
            print(f"  -> Transportierte Gesamtmasse:   {actual_hier_mass:.2f} (Soll: {total_problem_mass}) <-- Check 2")
            print(f"  -> Anzahl aktiver Paare (>1e-5): {active_hier_pairs} (Theor. Max: {2*N-1}) <-- Check 1")
            
            # Direkter struktureller Vergleich der Matrizen
            matrices_identical = np.allclose(dense_mu, hier_mu, atol=1e-3) if dense_mu is not None else False
            print(f"  -> Struktur-Abgleich: Sind Primalpläne identisch? {matrices_identical}")
            
        except Exception as e:
            print(f"\n[HIERARCHICAL OT] FEHLGESCHLAGEN")
            import traceback; traceback.print_exc()
            
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    run_comprehensive_benchmarks()
