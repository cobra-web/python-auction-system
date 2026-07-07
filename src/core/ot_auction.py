import numpy as np
import sys
from src.utils.data_structures import SparseNeighborhood

class AuctionOT:
    def __init__(self, cost_matrix, mu_X, mu_Y, epsilon=None, allowed_edges=None):
        self.C = np.array(cost_matrix, dtype=float)
        self.mu_X = np.array(mu_X, dtype=int)
        self.mu_Y = np.array(mu_Y, dtype=int)
        self.N_X, self.N_Y = self.C.shape

        assert np.sum(self.mu_X) == np.sum(self.mu_Y), "Total mass in X must equal total mass in Y"

        if epsilon is None:
            self.epsilon = 1.0 / (np.sum(self.mu_X) + 1.0)
        else:
            self.epsilon = epsilon

        self.mu = np.zeros((self.N_X, self.N_Y), dtype=int)  # coupling matrix

        # Track mass fragments for each y target
        self.y_atoms = []
        for y in range(self.N_Y):
            self.y_atoms.append([{'x': -1, 'mass': self.mu_Y[y], 'beta': 0.0}])

        if allowed_edges is not None:
            self.sparse = SparseNeighborhood(self.N_X, self.N_Y, allowed_edges)
        else:
            self.sparse = None

    def solve(self):
        iterations = 0
        while True:
            assigned_X = np.sum(self.mu, axis=1)
            unassigned_X = self.mu_X - assigned_X

            if np.sum(unassigned_X) == 0:
                break

            bids = self._bidding_phase(unassigned_X)
            self._assignment_phase(bids)
            iterations += 1

            if iterations > 20000:
                raise RuntimeError(f"AuctionOT did not converge after {iterations} iterations - likely a cycling/dropped-bid bug")

        total_cost = np.sum(self.mu * self.C)
        return self.mu, total_cost, iterations

    def _bidding_phase(self, unassigned_X):
        bids = {y: [] for y in range(self.N_Y)}

        for x in range(self.N_X):
            mass_to_assign = unassigned_X[x]
            if mass_to_assign <= 0:
                continue

            Pi_x = []
            if self.sparse is not None:
                valid_y_targets = self.sparse.get_allowed_y(x)
            else:
                valid_y_targets = range(self.N_Y)

            for y in valid_y_targets:
                for atom in self.y_atoms[y]:
                    if atom['x'] == x:
                        continue
                    eff_cost = self.C[x, y] - atom['beta']
                    Pi_x.append((eff_cost, y, atom['mass'], atom['x'], atom['beta'], atom))

            if not Pi_x:
                continue

            # Sort targets by lowest effective cost
            Pi_x.sort(key=lambda item: item[0])

            accumulated_mass = 0
            bid_targets = []
            alpha_prime = 0

            for i, item in enumerate(Pi_x):
                eff_cost, y, mass_avail, x_prime, old_beta, atom_ref = item
                claim_amount = min(mass_avail, mass_to_assign - accumulated_mass)
                if claim_amount > 0:
                    bid_targets.append({'y': y, 'mass': claim_amount, 'eff_cost': eff_cost, 'x_prime': x_prime, 'atom_ref': atom_ref})
                    accumulated_mass += claim_amount

                if accumulated_mass >= mass_to_assign:
                    if i + 1 < len(Pi_x):
                        alpha_prime = Pi_x[i+1][0]
                    else:
                        alpha_prime = eff_cost
                    break

            for target in bid_targets:
                y = target['y']
                bid_value = self.C[x, y] - alpha_prime - self.epsilon
                bids[y].append({'x': x, 'mass': target['mass'], 'bid_value': bid_value, 'x_prime': target['x_prime'], 'atom_ref': target['atom_ref']})

        return bids

    def _assignment_phase(self, bids):
        for y, y_bids in bids.items():
            if not y_bids:
                continue

            # Group bids by the unique object memory ID of the specific atom fragment being targeted
            by_atom = {}
            for bid in y_bids:
                by_atom.setdefault(id(bid['atom_ref']), []).append(bid)

            # Map atom object memory IDs for robust lookup
            atom_map = {id(atom): atom for atom in self.y_atoms[y]}

            for atom_id, atom_bids in by_atom.items():
                if atom_id not in atom_map:
                    continue
                
                atom = atom_map[atom_id]
                x_prime = atom['x']
                available_mass = atom['mass']

                # Lower bid_value means more competitive price adjustment; sort ascending
                atom_bids.sort(key=lambda b: b['bid_value'])

                satisfied = []
                unmet_bids = []

                for bid in atom_bids:
                    if available_mass > 0:
                        claim = min(bid['mass'], available_mass)
                        satisfied.append({'x': bid['x'], 'mass': claim, 'bid_value': bid['bid_value']})
                        available_mass -= claim
                        if claim < bid['mass']:
                            unmet_bids.append({'bid_value': bid['bid_value']})
                    else:
                        unmet_bids.append(bid)

                min_unmet_val = min(b['bid_value'] for b in unmet_bids) if unmet_bids else None

                if x_prime != -1:
                    total_won = sum(sat['mass'] for sat in satisfied)
                    self.mu[x_prime, y] -= total_won

                for sat in satisfied:
                    x = sat['x']
                    m_won = sat['mass']
                    new_beta = sat['bid_value']
                    
                    # Ensure price drops cleanly to clear the market smoothly
                    if min_unmet_val is not None:
                        new_beta = min(new_beta, min_unmet_val - self.epsilon)

                    self.mu[x, y] += m_won
                    self.y_atoms[y].append({'x': x, 'mass': m_won, 'beta': new_beta})

                atom['mass'] = available_mass
                if min_unmet_val is not None and atom['mass'] > 0:
                    atom['beta'] = min(atom['beta'] - self.epsilon, min_unmet_val - self.epsilon)

            # Housekeeping: remove completely drained fragments
            self.y_atoms[y] = [a for a in self.y_atoms[y] if a['mass'] > 0]
