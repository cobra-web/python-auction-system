import numpy as np
import sys
from src.utils.data_structures import SparseNeighborhood

class AuctionOT:
    def __init__(self, cost_matrix, mu_X, mu_Y, epsilon=None, allowed_edges=None, initial_beta=None):
        self.C = np.array(cost_matrix, dtype=float)
        self.mu_X = np.array(mu_X, dtype=float) # Support float tolerances
        self.mu_Y = np.array(mu_Y, dtype=float)
        self.N_X, self.N_Y = self.C.shape

        if epsilon is None:
            self.epsilon = 1.0 / (np.sum(self.mu_X) + 1.0)
        else:
            self.epsilon = epsilon

        self.mu = np.zeros((self.N_X, self.N_Y), dtype=float)

        self.y_atoms = []
        for y in range(self.N_Y):
            # Accept initial beta if passed directly
            start_price = 0.0 if initial_beta is None else initial_beta[y]
            self.y_atoms.append([{'x': -1, 'mass': self.mu_Y[y], 'beta': start_price}])

        if allowed_edges is not None:
            self.sparse = SparseNeighborhood(self.N_X, self.N_Y, allowed_edges)
        else:
            self.sparse = None

    def solve(self):
        iterations = 0
        while True:
            assigned_X = np.sum(self.mu, axis=1)
            unassigned_X = self.mu_X - assigned_X

            # Floating point tolerance for unassigned mass
            if np.sum(unassigned_X[unassigned_X > 1e-8]) == 0:
                break

            bids = self._bidding_phase(unassigned_X)
            self._assignment_phase(bids)
            iterations += 1
            
            if iterations <= 50 or iterations % 500 == 0:
                unassigned_val = np.sum(unassigned_X[unassigned_X > 1e-8])
                total_atoms = sum(len(a) for a in self.y_atoms)
                print(f"iter {iterations}: unassigned={unassigned_val:.2f}, total_atoms={total_atoms}", file=sys.stderr)

            if iterations > 20000:
                raise RuntimeError(f"AuctionOT did not converge after {iterations} iterations.")

        total_cost = np.sum(self.mu * self.C)
        return self.mu, total_cost, iterations

    def _bidding_phase(self, unassigned_X):
        bids = {y: [] for y in range(self.N_Y)}

        for x in range(self.N_X):
            mass_to_assign = unassigned_X[x]
            if mass_to_assign <= 1e-8:
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
                
                if claim_amount > 1e-8:
                    bid_targets.append({'y': y, 'mass': claim_amount, 'eff_cost': eff_cost, 'x_prime': x_prime})
                    accumulated_mass += claim_amount

                if accumulated_mass >= mass_to_assign - 1e-8:
                    if i + 1 < len(Pi_x):
                        alpha_prime = Pi_x[i+1][0]
                    else:
                        alpha_prime = eff_cost 
                    break

            for target in bid_targets:
                y = target['y']
                # Price strictly decreases by epsilon
                bid_value = self.C[x, y] - alpha_prime - self.epsilon
                bids[y].append({'x': x, 'mass': target['mass'], 'bid_value': bid_value, 'x_prime': target['x_prime']})

        return bids

    def _assignment_phase(self, bids):
        atom_bids = {}
        for y, y_bids in bids.items():
            for bid in y_bids:
                key = (y, bid['x_prime'])
                atom_bids.setdefault(key, []).append(bid)

        for (y, x_prime), claims in atom_bids.items():
            atom = None
            for a in self.y_atoms[y]:
                if a['x'] == x_prime:
                    atom = a
                    break
            if atom is None:
                continue

            claims.sort(key=lambda b: b['bid_value'])

            remaining = atom['mass']
            accepted = []

            for bid in claims:
                if remaining <= 1e-8:
                    break
                grant = min(bid['mass'], remaining)
                accepted.append((bid, grant))
                remaining -= grant

            for bid, grant in accepted:
                if grant <= 1e-8:
                    continue
                x = bid['x']
                new_beta = bid['bid_value']
                
                if x_prime != -1:
                    self.mu[x_prime, y] -= grant
                self.mu[x, y] += grant
                atom['mass'] -= grant
                self.y_atoms[y].append({'x': x, 'mass': grant, 'beta': new_beta})

        for y in range(self.N_Y):
            consolidated = {}
            for a in self.y_atoms[y]:
                if a['mass'] <= 1e-8:
                    continue
                owner = a['x']
                if owner not in consolidated:
                    consolidated[owner] = a
                else:
                    consolidated[owner]['mass'] += a['mass']
                    consolidated[owner]['beta'] = min(consolidated[owner]['beta'], a['beta'])
            self.y_atoms[y] = list(consolidated.values())
