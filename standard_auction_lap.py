def hierarchical_auction_lap(X, Y, costs, epsilon):
    S = {} 
    unassigned_x = list(X)
    
    beta = {y: 0.0 for y in Y}

    while unassigned_x:
        
        #1. Bidding Phase
        bids = {y: [] for y in Y}

        for x in unassigned_x:
            N_x = list(costs[x].keys())
            
            if not N_x:
                continue
                
            eff_costs = [(costs[x][y] - beta[y], y) for y in N_x]
            
            eff_costs.sort(key=lambda item: item[0])
            best_eff_cost, y_star = eff_costs[0]
            
            if len(eff_costs) > 1:
                alpha_prime = eff_costs[1][0]
            else:
                alpha_prime = best_eff_cost 
                
            bid_value = costs[x][y_star] - alpha_prime - epsilon
            
            bids[y_star].append((x, bid_value))

        # 2. Assignment Phase
        for y in Y:
            P_y = bids[y] # set of bids received
            
            if P_y:
                winning_x, lowest_bid = min(P_y, key=lambda item: item[1])
                
                beta[y] = lowest_bid
                
                if y in S:
                    old_x = S[y]
                    unassigned_x.append(old_x)
                    
                S[y] = winning_x
                unassigned_x.remove(winning_x)
                
    return S, beta
