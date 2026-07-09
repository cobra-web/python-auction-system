min_all = np.min(C_fine[x] - beta)
min_sparse = np.min(C_fine[x,valid] - beta[valid])

print(min_sparse-min_all)
