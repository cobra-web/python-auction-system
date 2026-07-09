import numpy as np
import numpy as np
from src.core.ot_auction import AuctionOT, TOL
from src.utils.eps_scaling import EpsScalingManager
from src.hierarchical.consistency import ConsistencyChecker

min_all = np.min(C_fine[x] - beta)
min_sparse = np.min(C_fine[x,valid] - beta[valid])

print(min_sparse-min_all)
