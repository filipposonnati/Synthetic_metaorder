import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

# --- Chart Configuration ---
# Set global matplotlib style parameters for consistent, readable plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

# --- Fitting Functions ---

def square_root(x, Y):
    """Square root model: y = Y * sqrt(x). Used to fit the impact vs. time curve."""
    return Y * np.sqrt(x)

def power_law(x, A, B):
    """Generic power law model: y = A * x^B. Used as an alternative fit."""
    return A * np.power(x, B)


# --- Data Loading ---

# Define the dataset identifier and construct the path to the trades directory
function = '20_power_2.0'
dir = '..\\database\\trades_' + function
paths = np.array(listdir(dir))

trades_total = None

# Iterate over all CSV files in the directory and concatenate them into a single DataFrame
for path in paths:
    trades = pd.read_csv(dir + f'\\{path}', sep=',')
    # Select only the relevant columns
    cols = ['Ratio_pre', 'BeginMid', 'MetaDuration', 'ElapsedTime', 'NbChild', 'MetaVolume']
    daily_trades = trades[cols].copy()
    trades_total = pd.concat([trades_total, daily_trades])

# Rename column for clarity
trades_total.rename(columns={'Ratio_pre': 'Ratio'}, inplace=True)

# --- Filtering ---

# Keep only meta-orders with more than one child order
trades_total = trades_total[trades_total['NbChild'] > 1]
# Remove rows where no time has elapsed (avoid division by zero or degenerate cases)
trades_total = trades_total[trades_total['ElapsedTime'] > 0.0]

# Compute normalized time: fraction of meta-order duration elapsed at each child order
trades_total['NormalizedTime'] = trades_total['ElapsedTime'] / trades_total['MetaDuration']


# --- Time Binning Setup ---

# Create 12 non-uniform bins in [0, 1] using a cubic scale (denser bins near 0)
time_bins = np.linspace(0.0, 1.0, 13)**3

n_bins        = len(time_bins) - 1   # Total number of bins (12)
n_plots       = 4                    # Number of subplots for histogram grid
bins_per_plot = n_bins // n_plots    # Bins shown per subplot (3)


# --- Diagnostics: Zero Ratio ---

# Count how many trades have exactly zero partial impact
n_total = len(trades_total)
n_zero  = (trades_total['Ratio'] == 0.0).sum()
pct     = n_zero / n_total * 100

print('\nPartialImpact Diagnostics:')
print(f'  Total trades              : {n_total:,}')
print(f'  PartialImpact == 0 (exact): {n_zero:,}  ({pct:.2f}%)')

# Assign each trade to a time bin
trades_total['time_bin_idx'] = pd.cut(
    trades_total['NormalizedTime'],
    bins=time_bins,
    labels=False,
    include_lowest=True
)

# Compute the fraction of zero-Ratio trades in each bin and print the first 10 bins
trades_total['is_zero'] = trades_total['Ratio'] == 0.0
zero_by_bin = trades_total.groupby('time_bin_idx', observed=True)['is_zero'].mean() * 100
print('\n  % of Ratio == 0 per bin (first 10 bins):')
print(zero_by_bin.head(10).to_string())


# --- Color Palette ---

# 5 distinct colors for the subplots (one per bin group within each subplot)
palette = [plt.get_cmap("tab10")(i) for i in range(5)]


# ---------------------------------------------------------------
# HISTOGRAMS: Distribution of Ratio across time bins
# ---------------------------------------------------------------

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
axes2 = axes2.flatten()

for plot_idx in range(n_plots):
    ax        = axes2[plot_idx]
    start_bin = plot_idx * bins_per_plot
    end_bin   = start_bin + bins_per_plot

    for j, i in enumerate(range(start_bin, end_bin)):
        # Select trades falling in bin i
        subset = trades_total.loc[trades_total['time_bin_idx'] == i, 'Ratio'].dropna()
        if len(subset) < 10:
            continue  # Skip bins with insufficient data

        mean   = subset.mean()
        t_low  = time_bins[i]
        t_high = time_bins[i + 1]
        color  = palette[j]

        # Plot histogram as a step curve (no fill) and mark the mean with a dashed vertical line
        ax.hist(subset, bins=200, density=True, histtype='step',
                linewidth=1.5, color=color,
                label=f'[{t_low:.2e}, {t_high:.2e}]')
        ax.axvline(mean, color=color, linewidth=1.2, linestyle='--')

    # Determine x-axis limits using the 1st and 99th percentiles to avoid outlier distortion
    subplot_data = trades_total.loc[
        trades_total['time_bin_idx'].between(start_bin, end_bin - 1), 'Ratio'
    ].dropna()
    ax_xmin = subplot_data.quantile(0.01)
    ax_xmax = subplot_data.quantile(0.99)

    t_start = time_bins[start_bin]
    t_end   = time_bins[end_bin]
    ax.set_title(f'$t/T$ ∈ [{t_start:.2e}, {t_end:.2e}]')
    ax.set_xlabel(r'$I_i / \sqrt{Q}$')
    ax.set_ylabel('Density')
    ax.legend(title=r'$t/T$')
    ax.set_xlim(ax_xmin, ax_xmax)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join('..\\images', 'impact_time_curve', f'{function}_hist.png'), dpi=150, bbox_inches='tight')
plt.close()


# ---------------------------------------------------------------
# DISTRIBUTION OF Ratio BY NbChild — fixed time bin
# ---------------------------------------------------------------

TARGET_BIN = 1  # Change this value to explore other time bins

# Define NbChild groups with filtering lambdas
nbchild_groups = {
    '2':   lambda x: x == 2,
    '3':   lambda x: x == 3,
    '4':   lambda x: x == 4,
    '5':   lambda x: x == 5,
    '6+':  lambda x: x >= 6,
}
group_colors = [plt.get_cmap("tab10")(i) for i in range(len(nbchild_groups))]

# Subset data to the selected time bin
bin_data = trades_total[trades_total['time_bin_idx'] == TARGET_BIN].copy()

print(f"\nBin {TARGET_BIN} - NbChild distribution:")
print(bin_data['NbChild'].value_counts().sort_index().head(10).to_string())

t_low_label  = time_bins[TARGET_BIN]
t_high_label = time_bins[TARGET_BIN + 1]

# Clip x-axis to avoid extreme outliers
xb_min = bin_data['Ratio'].quantile(0.01)
xb_max = bin_data['Ratio'].quantile(0.99)

fig_nb, ax_hist = plt.subplots(1, 1, figsize=(8, 5))
fig_nb.suptitle(
    f"Distribution of " + r"$I_i \sqrt{{Q}}$" + f" for bin {TARGET_BIN}",
    fontsize=15
)

# Plot histogram for each NbChild group
for (label, condition), color in zip(nbchild_groups.items(), group_colors):
    subset = bin_data.loc[condition(bin_data['NbChild']), 'Ratio'].dropna()
    if len(subset) < 10:
        continue  # Skip groups with too few samples
    mean = subset.mean()
    n    = len(subset)

    ax_hist.hist(subset, bins=100, density=True, histtype='step',
                 linewidth=1.5, color=color,
                 label=f'NbChild={label} (n={n:,})')
    ax_hist.axvline(mean, color=color, linewidth=1.2, linestyle='--')

ax_hist.set_xlim(xb_min, xb_max)
ax_hist.set_xlabel(r'$I_i / \sqrt{Q}$')
ax_hist.set_ylabel('Density')
ax_hist.set_title('Histogram')
ax_hist.legend(fontsize=9)
ax_hist.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join('..\\images', 'impact_time_curve', f'{function}_nchild_dist.png'), dpi=150, bbox_inches='tight')
plt.close()


# ---------------------------------------------------------------
# LOG-LOG PLOTS: Impact vs. MetaOrder Volume — one plot per time bin
# ---------------------------------------------------------------
# For each of the 12 time bins:
#   - histogram of MetaVolume on a twin right axis (frequency)
#   - binned mean impact with error bars (all bins same color)
#   - square root fit: I_i = Y * sqrt(Q), one free parameter Y per time bin
# The fitted Y values are collected and used as the y-axis of the
# impact-time-curve plot that follows.

# Keep only rows with positive MetaVolume and positive Ratio
loglog_data = trades_total.copy()

# Number of log-spaced volume bins for the binned-mean computation
N_VOL_BINS = 20

# Will be filled inside the loop; used by the impact-time-curve plot below
Y_per_bin     = np.full(n_bins, np.nan)   # fitted square-root amplitude per time bin
Y_err_per_bin = np.full(n_bins, np.nan)   # its uncertainty
t_mean_per_bin = np.full(n_bins, np.nan)  # mean normalised time per bin (x-axis)
t_err_per_bin  = np.full(n_bins, np.nan)  # standard error on the mean time

# Layout: 4 rows x 3 columns = 12 subplots
fig_ll, axes_ll = plt.subplots(4, 3, figsize=(18, 20))
axes_ll = axes_ll.flatten()

for bin_idx in range(n_bins):
    ax = axes_ll[bin_idx]

    # Select all trades in the current time bin
    subset = loglog_data[loglog_data['time_bin_idx'] == bin_idx]

    t_low  = time_bins[bin_idx]
    t_high = time_bins[bin_idx + 1]
    ax.set_title(f'$t/T$ in [{t_low:.3f}, {t_high:.3f}]', fontsize=11)

    if len(subset) < 20:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')
        ax.set_xlabel('$Q$')
        ax.set_ylabel(r'$I_i$')
        continue

    vol    = subset['MetaVolume'].to_numpy()
    # y-axis: actual impact I_i = Ratio * sqrt(Q), so the model is I_i = Y * sqrt(Q)
    impact = subset['Ratio'].to_numpy() * np.sqrt(vol)

    # Store mean normalised time for this bin (used as x in impact-time-curve)
    t_mean_per_bin[bin_idx] = subset['NormalizedTime'].mean()
    t_err_per_bin[bin_idx]  = subset['NormalizedTime'].std() / np.sqrt(len(subset))

    # ── Twin axis: histogram of MetaVolume (frequency) on the right ──────
    ax_hist = ax.twinx()
    vol_hist_edges = np.logspace(np.log10(vol.min()), np.log10(vol.max()), N_VOL_BINS + 1)
    ax_hist.hist(vol, bins=vol_hist_edges, color='steelblue', alpha=0.25,
                 label='Volume freq.')
    ax_hist.set_ylabel('Frequency', fontsize=9)
    ax_hist.tick_params(axis='y', labelsize=8)
    ax_hist.set_xscale('log')
    # Keep histogram behind the data points
    ax_hist.set_zorder(ax.get_zorder() - 1)
    ax.set_facecolor('none')  # make main axis background transparent so histogram shows

    # ── Pass 1: count raw points per volume bin (for the threshold filter) ─
    bin_counts = np.array([
        ((vol >= vol_hist_edges[k]) & (vol < vol_hist_edges[k + 1])).sum()
        for k in range(N_VOL_BINS)
    ])
    max_count       = bin_counts.max() if bin_counts.max() > 0 else 1
    count_threshold = 0.5 * max_count  # bins with <= 50% of peak count are excluded from fit

    # ── Pass 2: compute binned means for ALL bins; flag which enter the fit ─
    all_centers, all_means, all_errs = [], [], []
    fit_centers, fit_means, fit_errs = [], [], []

    for k in range(N_VOL_BINS):
        mask  = (vol >= vol_hist_edges[k]) & (vol < vol_hist_edges[k + 1])
        n_raw = mask.sum()
        if n_raw < 2:
            continue  # need at least 2 points for std

        vals   = impact[mask]
        center = np.sqrt(vol_hist_edges[k] * vol_hist_edges[k + 1])  # geometric mean
        mean   = vals.mean()
        err    = vals.std() / np.sqrt(n_raw)

        all_centers.append(center)
        all_means.append(mean)
        all_errs.append(err)

        if n_raw > count_threshold:
            fit_centers.append(center)
            fit_means.append(mean)
            fit_errs.append(err)

    all_centers = np.array(all_centers)
    all_means   = np.array(all_means)
    all_errs    = np.array(all_errs)
    fit_centers = np.array(fit_centers)
    fit_means   = np.array(fit_means)
    fit_errs    = np.array(fit_errs)

    print(f'Bin {bin_idx:2d} | t/T in [{t_low:.3f}, {t_high:.3f}] '
          f'| Volume bins used for fit: {len(fit_centers)} / {len(all_centers)} '
          f'(threshold: {count_threshold:.0f} raw points)')

    # ── Plot ALL binned means in a single colour ──────────────────────────
    if len(all_centers) >= 1:
        ax.errorbar(all_centers, all_means, yerr=all_errs,
                    fmt='o', color='crimson', markersize=5,
                    linewidth=1.2, capsize=3, label='Binned mean', zorder=5)

    # ── Square root fit on the selected (well-populated) bins only ────────
    # Model: I_i = Y * sqrt(Q)  →  one free parameter Y
    if len(fit_centers) >= 2:
        try:
            popt_sr, pcov_sr = curve_fit(
                square_root, fit_centers, fit_means,
                sigma=fit_errs, absolute_sigma=True,
                p0=[fit_means.mean() / np.sqrt(fit_centers.mean())],
                maxfev=5000
            )
            Y_fit     = popt_sr[0]
            Y_fit_err = np.sqrt(pcov_sr[0][0])

            # Store for the impact-time-curve plot
            Y_per_bin[bin_idx]     = Y_fit
            Y_err_per_bin[bin_idx] = Y_fit_err

            # Draw the fit curve over the full volume range
            v_plot = np.logspace(np.log10(vol.min()), np.log10(vol.max()), 200)
            ax.plot(v_plot, square_root(v_plot, Y_fit),
                    linestyle='--', color='black', linewidth=1.5,
                    label=f'$Y\\sqrt{{Q}}$, $Y$={Y_fit:.3f}±{Y_fit_err:.3f}', zorder=6)

            print(f'         Sqrt fit: Y = {Y_fit:.4f} +- {Y_fit_err:.4f}')
        except RuntimeError:
            print(f'Bin {bin_idx:2d}: square root fit did not converge.')

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$Q$')
    ax.set_ylabel(r'$I_i$')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, which='both', ls='--', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join('..\\images', 'impact_time_curve', f'{function}_impact_single.png'),
            dpi=150, bbox_inches='tight')
plt.close()


# ---------------------------------------------------------------
# MAIN PIPELINE: Impact-time curve using Y fitted per time bin
# ---------------------------------------------------------------
# x-axis: mean normalised time t/T per bin  (from the data)
# y-axis: fitted square-root amplitude Y    (from the log-log fit above)
# Fit: Y(t/T) = C * sqrt(t/T), one free parameter C

# Keep only bins where the log-log fit converged
valid = ~np.isnan(Y_per_bin)

x     = t_mean_per_bin[valid]
y     = Y_per_bin[valid]
x_err = t_err_per_bin[valid]
y_err = Y_err_per_bin[valid]


# --- Square Root Fit (iterative, propagating x uncertainty) ---

popt, pcov = curve_fit(square_root, x, y, sigma=y_err, absolute_sigma=True)
C     = popt[0]
C_err = np.sqrt(pcov[0][0])

for i in range(5):
    err_eff = np.sqrt(y_err**2 + (C * 0.5 * x**(-0.5) * x_err)**2)
    popt, pcov = curve_fit(square_root, x, y, sigma=err_eff, absolute_sigma=True)
    C     = popt[0]
    C_err = np.sqrt(pcov[0][0])

print(f'\nImpact-time curve fit: C = {C:.4f} +- {C_err:.4f}')


# --- Plot ---

fig3, ax3 = plt.subplots(figsize=(8, 6))

ax3.errorbar(x, y, yerr=y_err, xerr=x_err,
             linestyle='', marker='o', color='tab:blue',
             label='$Y$ from fit', zorder=3, capsize=2)

x_plot = np.linspace(0.0, 1.0, 200)
ax3.plot(x_plot, square_root(x_plot, C),
         label=rf'$C\sqrt{{t/T}}$',
         linestyle='--', color='black', zorder=2)

ax3.set_xlabel(r'$t/T$')
ax3.set_ylabel(r'$Y$')
ax3.legend()
ax3.grid(True, which='both', ls='-', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join('..\\images', 'impact_time_curve', f'{function}_impact_time_curve.png'))
plt.close()