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

        self.mu = np.zeros((self.N_X, self.N_Y), dtype=int) # coupling matrix

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
            # calculate unassigned mass for each x
            assigned_X = np.sum(self.mu, axis=1)
            unassigned_X = self.mu_X - assigned_X

            # if all mass is assigned, we are done
            if np.sum(unassigned_X) == 0:
                break

            bids = self._bidding_phase(unassigned_X)
            self._assignment_phase(bids)
            iterations += 1
            
            if iterations <= 50 or iterations % 500 == 0:
                print(f"iter {iterations}: unassigned={np.sum(unassigned_X)}, total_atoms={sum(len(a) for a in self.y_atoms)}", file=sys.stderr)

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
                    x_prime = atom['x']
                    if x_prime == x:
                        continue

                    eff_cost = self.C[x, y] - atom['beta']
                    Pi_x.append((eff_cost, y, atom['mass'], x_prime, atom['beta']))

            if not Pi_x:
                continue

            Pi_x.sort(key=lambda item: item[0])

            accumulated_mass = 0
            bid_targets = []
            alpha_prime = 0

            for i, item in enumerate(Pi_x):
                eff_cost, y, mass_avail, x_prime, old_beta = item

                claim_amount = min(mass_avail, mass_to_assign - accumulated_mass)
                if claim_amount > 0:
                    bid_targets.append({'y': y, 'mass': claim_amount, 'eff_cost': eff_cost, 'x_prime': x_prime})
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
                bids[y].append({'x': x, 'mass': target['mass'], 'bid_value': bid_value, 'x_prime': target['x_prime']})

        return bids

    def _assignment_phase(self, bids):
        for y, y_bids in bids.items():
            if not y_bids:
                continue

            by_atom = {}
            for bid in y_bids:
                by_atom.setdefault(bid['x_prime'], []).append(bid)

            for x_prime, atom_bids in by_atom.items():
                atom_idx = None
                for i, atom in enumerate(self.y_atoms[y]):
                    if atom['x'] == x_prime:
                        atom_idx = i
                        break

                if atom_idx is None:
                    continue

                atom = self.y_atoms[y][atom_idx]
                available_mass = atom['mass']

                # Higher bid_value means higher willingness to pay
                atom_bids.sort(key=lambda b: b['bid_value'], reverse=True)

                satisfied = []
                unmet_highest_bid_val = None

                for bid in atom_bids:
                    if available_mass > 0:
                        claim = min(bid['mass'], available_mass)
                        satisfied.append({'x': bid['x'], 'mass': claim, 'beta': bid['bid_value']})
                        available_mass -= claim
                        
                        # If a single bidder's demand wasn't completely filled, the remainder is unmet
                        if claim < bid['mass'] and unmet_highest_bid_val is None:
                            unmet_highest_bid_val = bid['bid_value']
                    else:
                        if unmet_highest_bid_val is None:
                            unmet_highest_bid_val = bid['bid_value']

                # Execute transitions on assignments
                for sat in satisfied:
                    x = sat['x']
                    m_won = sat['mass']
                    new_beta = sat['beta']

                    if x_prime != -1:
                        self.mu[x_prime, y] -= m_won

                    self.mu[x, y] += m_won
                    atom['mass'] -= m_won

                    self.y_atoms[y].append({'x': x, 'mass': m_won, 'beta': new_beta})

                # Strict price increase mechanism for the remaining unsatisfied atom block
                if unmet_highest_bid_val is not None and atom['mass'] > 0:
                    atom['beta'] = max(atom['beta'] + self.epsilon, unmet_highest_bid_val)

            # Housekeeping
            self.y_atoms[y] = [a for a in self.y_atoms[y] if a['mass'] > 0]
