import numpy as np
from scipy.special import logsumexp

def log_sinkhorn(cost_matrix, mu_X, mu_Y, gamma=0.05, max_iters=1000, tol=1e-5):
    """
    Log-domain stabilized Sinkhorn Algorithm for Entropy-Regularized OT.
    Solves: min <mu, C> + gamma * H(mu)
    """
    C = np.array(cost_matrix, dtype=float)
    a = np.array(mu_X, dtype=float)
    b = np.array(mu_Y, dtype=float)
    
    # Normalize masses to probability distributions for Sinkhorn
    a_norm = a / np.sum(a)
    b_norm = b / np.sum(b)
    
    N_X, N_Y = C.shape
    
    # Initialize dual potentials f and g in log-domain
    f = np.zeros(N_X)
    g = np.zeros(N_Y)
    
    log_a = np.log(a_norm)
    log_b = np.log(b_norm)
    
    for it in range(max_iters):
        f_old = np.copy(f)
        
        # Update potential f (rows)
        for i in range(N_X):
            f[i] = log_a[i] - logsumexp(g - C[i, :] / gamma)
            
        # Update potential g (columns)
        for j in range(N_Y):
            g[j] = log_b[j] - logsumexp(f - C[:, j] / gamma)
            
        # Check convergence on the potentials
        if np.linalg.norm(f - f_old) < tol:
            break
            
    # Reconstruct the optimal coupling matrix
    log_P = (f[:, None] + g[None, :] - C) / gamma
    P = np.exp(log_P)
    
    # Scale back up to match original integer masses
    coupling = P * np.sum(mu_X)
    
    true_cost = np.sum(coupling * C)
    return coupling, true_cost, it
