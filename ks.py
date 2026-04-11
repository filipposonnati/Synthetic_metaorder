import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, ks_2samp

# 1. Generate two discrete samples
mu1, mu2 = 3, 4
size = 10000
np.random.seed(42)

sample1 = poisson.rvs(mu1, size=size)
sample2 = poisson.rvs(mu2, size=size)

# 2. Sort data for ECDF calculation
sorted_1 = np.sort(sample1)
sorted_2 = np.sort(sample2)
y = np.arange(1, size + 1) / size

# 3. Create Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left Plot: Histograms (Frequency)
ax1.hist(sample1, bins=range(max(sample2)+2), alpha=0.5, label=f'Poisson ($\mu={mu1}$)', density=True, color='blue', edgecolor='black')
ax1.hist(sample2, bins=range(max(sample2)+2), alpha=0.5, label=f'Poisson ($\mu={mu2}$)', density=True, color='orange', edgecolor='black')
ax1.set_title('Probability Density (Histogram)')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.legend()

# Right Plot: ECDF (The basis for K-S Test)
ax2.step(sorted_1, y, label=f'ECDF Sample 1 ($\mu={mu1}$)', where='post', color='blue')
ax2.step(sorted_2, y, label=f'ECDF Sample 2 ($\mu={mu2}$)', where='post', color='orange')

# Logic to find and plot the max distance D
all_vals = np.sort(np.concatenate([sample1, sample2]))
ecdf1 = np.searchsorted(sorted_1, all_vals, side='right') / size
ecdf2 = np.searchsorted(sorted_2, all_vals, side='right') / size
d_vals = np.abs(ecdf1 - ecdf2)
max_idx = np.argmax(d_vals)
x_at_max_d = all_vals[max_idx]

ax2.vlines(x_at_max_d, ecdf1[max_idx], ecdf2[max_idx], color='red', linestyle='--', label='KS Distance ($D$)')
ax2.set_title('Empirical Cumulative Distribution Function (ECDF)')
ax2.set_xlabel('Value')
ax2.set_ylabel('Cumulative Probability')
ax2.legend()

plt.tight_layout()
plt.savefig('images\\ks_test_visualization.png')

# Run the test
stat, p = ks_2samp(sample1, sample2)
print(f"KS Statistic (D): {stat}")
print(f"P-value: {p}")