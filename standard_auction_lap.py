def hierarchical_auction_lap(X, Y, costs, epsilon):
    """
    Auction Algorithm for the Linear Assignment Problem.
    Aligned with Schmitzer & Schnörr (2013).
    
    Args:
        X: List of source elements.
        Y: List of target elements.
        costs: A dictionary of dictionaries representing sparse neighborhoods. 
               costs[x][y] gives the cost c(x,y). If y is not in costs[x], 
               it is not in the neighborhood N(x).
        epsilon: A positive float for the minimum bidding increment.
    """
    
    # Initialization [cite: 103-104]
    # S represents the assignment, mapping target y to its assigned source x
    S = {} 
    unassigned_x = list(X)
    
    # Dual variables beta(y) initialized to arbitrary values (e.g., 0.0) [cite: 104]
    beta = {y: 0.0 for y in Y}

    # The algorithm repeats until S is complete [cite: 123]
    while unassigned_x:
        
        # --- 1. BIDDING PHASE ---
        # Different x do not interact during this phase [cite: 101]
        bids = {y: [] for y in Y}

        for x in unassigned_x:
            # N(x) is the set of neighbors for x where cost < infinity [cite: 60-61, 107]
            N_x = list(costs[x].keys())
            
            if not N_x:
                continue # Edge case: x has no valid neighbors
                
            # Calculate effective cost: c(x,y) - beta(y) for y in N(x) [cite: 107]
            eff_costs = [(costs[x][y] - beta[y], y) for y in N_x]
            
            # Find a minimizer y* and the corresponding alpha(x) [cite: 109]
            eff_costs.sort(key=lambda item: item[0])
            best_eff_cost, y_star = eff_costs[0]
            
            # Determine the slack of the second 'nearest' constraint alpha'(x) [cite: 109-110]
            if len(eff_costs) > 1:
                alpha_prime = eff_costs[1][0]
            else:
                # If only one neighbor exists, the second best is conceptually infinity.
                # We set it to best_eff_cost to force an aggressive bid.
                alpha_prime = best_eff_cost 
                
            # Element x bids for element y* with value b_xy* [cite: 111-112]
            # b_xy* = c(x,y*) - alpha'(x) - epsilon
            bid_value = costs[x][y_star] - alpha_prime - epsilon
            
            # Submit the bid to the target [cite: 115]
            bids[y_star].append((x, bid_value))

        # --- 2. ASSIGNMENT PHASE ---
        # Different y do not interact during this phase [cite: 101]
        for y in Y:
            P_y = bids[y] # P(y) is the set of bids received [cite: 115-116]
            
            if P_y:
                # Find the lowest bid [cite: 117-118]
                # FIXED: The index is now 1 to access 'bid_value' from the (x, bid_value) tuple
                winning_x, lowest_bid = min(P_y, key=lambda item: item[1])
                
                # Decrease beta(y) to the lowest bid [cite: 117-118]
                beta[y] = lowest_bid
                
                # Remove any existing pair (x, y) from assignment S [cite: 121]
                if y in S:
                    old_x = S[y]
                    unassigned_x.append(old_x)
                    
                # Add to S the pair (x*, y) [cite: 121]
                S[y] = winning_x
                unassigned_x.remove(winning_x)
                
            # If P(y) is empty, beta(y) is left unchanged [cite: 122]

    return S, beta