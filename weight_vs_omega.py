import matplotlib.pylab as plt
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np

weights = pd.read_csv('./results/weights.csv').to_numpy()
omega = pd.read_csv('./results/naive_omega.csv').to_numpy()
weights = weights[:, 1:]
omega = omega[:, 1:]

omega_flt = omega.flatten()
weights_flt = weights.flatten()
valid_indx = np.where(omega_flt != -1)
weights_ = weights_flt[valid_indx]
omega_ = omega_flt[valid_indx]
sorted_ind = np.argsort(weights_)


weights_ = weights_[sorted_ind]
omega_ = omega_[sorted_ind]

fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[1:, :-1])
main_ax.scatter(weights_, omega_, marker='o', alpha=0.2)
plt.xlabel("weights")
plt.ylabel("Omega")
x_hist = fig.add_subplot(grid[0, :-1], yticklabels=[], sharex=main_ax)
x_hist.hist(weights_, orientation='vertical', density=True)

y_hist = fig.add_subplot(grid[1:, -1], xticklabels=[], sharey=main_ax)
y_hist.hist(omega_, histtype='stepfilled', orientation='horizontal', density=False)
y_hist.invert_yaxis()

plt.savefig("./results/weights_vs_omega.png")
