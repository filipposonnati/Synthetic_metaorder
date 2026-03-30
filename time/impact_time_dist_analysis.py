import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# --- Graphics Configuration ---
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 14})
# Color gradient for the first 5 bins
colors = plt.cm.viridis(np.linspace(0, 0.8, 5)) 

# --- Data Loading ---
# Ensure the path matches your local directory structure
df = pd.read_csv(f'..\\database\\post_trades\\20_power_2.0_0.csv', sep=',')

# --- Bin Definition ---
n_bins = 30
# Non-linear binning using t^3 to cluster more bins near t=0
bin_edges = np.linspace(0.0, 1.0, n_bins + 1) ** 3
n_to_plot = 5 # Number of early bins to visualize in histograms

# =====================================================
# PART 1: Distribution Comparison (First 5 Bins)
# =====================================================
fig_dist, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(12, 12))

for i in range(n_to_plot):
    t_start, t_end = bin_edges[i], bin_edges[i+1]
    mask = (df['NormalizedTime'] >= t_start) & (df['NormalizedTime'] < t_end)
    bin_data = df.loc[mask, 'Ratio'].dropna()
    
    lo, hi = bin_data.quantile(0.05), bin_data.quantile(0.95)
    bin_data = bin_data[(bin_data >= lo) & (bin_data <= hi)]

    label = f'{i} [{t_start:.1e}, {t_end:.1e}]'
    mean_val = bin_data.mean()
    
    # Linear Plot (using density=True for PDF comparison)
    ax_lin.hist(bin_data, bins=80, density=True, histtype='step', 
                linewidth=2, color=colors[i], label=label)
    ax_lin.axvline(mean_val, color=colors[i], linestyle='--', alpha=0.7)

    # Logarithmic Plot (Y-axis)
    ax_log.hist(bin_data, bins=80, density=True, histtype='step', 
                linewidth=2, color=colors[i], label=label)
    ax_log.axvline(mean_val, color=colors[i], linestyle='--', alpha=0.7)

# Formatting Linear Distribution
ax_lin.set_ylabel("Probability Density")
ax_lin.legend()
ax_lin.grid(True, alpha=0.3)

# Formatting Log Distribution
ax_log.set_yscale('log')
ax_log.set_xlabel(r'$I(t) / \sqrt{Q}$')
ax_log.set_ylabel("Log Density")
ax_log.legend()
ax_log.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig('..\\images\\distributions_comparison.png')

# =====================================================
# PART 2: Statistical Moments Evolution (All Bins)
# =====================================================
stats_list = []

for i in range(len(bin_edges)-1):
    t_start, t_end = bin_edges[i], bin_edges[i+1]
    t_mid = (t_start + t_end) / 2
    
    mask = (df['NormalizedTime'] >= t_start) & (df['NormalizedTime'] < t_end)
    data = df.loc[mask, 'Ratio'].dropna()
    
    if not data.empty:
        lo, hi = data.quantile(0.05), data.quantile(0.95)
        trimmed = data[(data >= lo) & (data <= hi)]

        stats_list.append({
            'time': t_mid,
            'mean': trimmed.mean(),
            'median': trimmed.median(),
            'std': trimmed.std(),
            'skew': skew(trimmed),
            'kurt': kurtosis(trimmed),
            'count': len(trimmed)
        })

df_stats = pd.DataFrame(stats_list)

# Plotting the Evolution of Moments
fig_moments, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Mean & Median (Central Tendency)
axes[0, 0].plot(df_stats['time'], df_stats['mean'], '-o', label='Mean', color='crimson')
axes[0, 0].plot(df_stats['time'], df_stats['median'], '-s', label='Median', color='darkorange')
axes[0, 0].set_title('Central Tendency')
axes[0, 0].set_ylabel('Ratio Value')
axes[0, 0].legend()

# 2. Standard Deviation (Dispersion/Volatility)
axes[0, 1].plot(df_stats['time'], df_stats['std'], '-o', color='forestgreen')
axes[0, 1].set_title('Standard Deviation')
axes[0, 1].set_ylabel('Sigma')

# 3. Skewness (Asymmetry)
axes[1, 0].plot(df_stats['time'], df_stats['skew'], '-o', color='rebeccapurple')
axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[1, 0].set_title('Skewness')
axes[1, 0].set_xlabel('Normalized Time')
axes[1, 0].set_ylabel('Skewness')

# 4. Kurtosis (Tail Heaviness)
axes[1, 1].plot(df_stats['time'], df_stats['kurt'], '-o', color='sienna')
axes[1, 1].set_title('Kurtosis')
axes[1, 1].set_xlabel('Normalized Time')
axes[1, 1].set_ylabel('Excess Kurtosis')

# Apply log scale to X-axis for all moment plots to see early-time behavior
for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    #ax.set_xscale('log') 

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('..\\images\\moments_evolution.png')
plt.show()