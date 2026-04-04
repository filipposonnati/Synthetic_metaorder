import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit

# --- CONFIGURATION AND FILTERING LIMITS ---
# These limits define the liquidity regime where the power law is most valid
PATH = 'meta_20_power_2.0.csv'
DIR = 'database\\meta'

# Bounds used to isolate the linear regime in log-log space
LOG_A_MIN, LOG_A_MAX = -1.8, -0.1  # Participation rate range
LOG_B_MIN, LOG_B_MAX = -4.5, -1.5  # Relative volume range

# Significance filter to exclude bins with very few samples to avoid noise
SAMPLE_THRESHOLD_PCT = 0.01        

# --- MODEL DEFINITIONS ---

def log_function_3_ab(log_x, log_Y, delta, alpha):
    """
    Linearized Power Law Model where log10(I) equals log10(Y) plus delta times log10(a) plus alpha times log10(b)
    This translates mathematically to I equals Y times a raised to the delta times b raised to the alpha
    """
    log_a, log_b = log_x
    return log_Y + delta * log_a + alpha * log_b

def safe_err(std_col, count_col, mean_col, fallback_rel=1.0):
    """
    Calculates Standard Error as standard deviation divided by the square root of n
    If a bin has zero or one samples it assigns a fallback relative error to prevent division by zero during fitting
    """
    err = std_col / np.sqrt(count_col)
    bad = (err == 0) | np.isnan(err)
    err[bad] = fallback_rel * mean_col[bad]
    return err

# --- DATA LOADING AND PRE-PROCESSING ---

synthetic_meta = pd.read_csv(f'{DIR}\\{PATH}', sep=',', parse_dates=['BeginTime', 'EndTime'])
df_res = synthetic_meta[['MetaVolume', 'TradedVolume', 'MetaImpact']].copy()

# Define dimensionless market variables where a is participation rate and b is relative volume
df_res['a'] = df_res['MetaVolume'] / df_res['TradedVolume']
df_res['b'] = df_res['TradedVolume']

# Remove non-positive values to allow for correct logarithmic transformation
df_res = df_res[(df_res['a'] > 0) & (df_res['b'] > 0) & (df_res['MetaImpact'] > 0)].copy()

# --- 2D LOGARITHMIC BINNING ---

# Create log-spaced bins for each dimension to establish a two-dimensional grid
bins_a = np.logspace(np.log10(df_res['a'].min()), np.log10(df_res['a'].max()), 31)
bins_b = np.logspace(np.log10(df_res['b'].min()), np.log10(df_res['b'].max()), 31)

df_res['bin_a'] = pd.cut(df_res['a'], bins=bins_a, include_lowest=True)
df_res['bin_b'] = pd.cut(df_res['b'], bins=bins_b, include_lowest=True)

# Calculate mean impact and associated errors for each 2D bin
grouped = df_res.groupby(['bin_a', 'bin_b'], observed=True).agg({
    'a': ['mean', 'std'],
    'b': ['mean', 'std'],
    'MetaImpact': ['mean', 'std', 'count']
}).dropna()

grouped.columns = ['a_mean', 'a_std', 'b_mean', 'b_std', 'impact_mean', 'impact_std', 'sample_count']

# Propagate errors for visualization purposes and weighted regression analysis
grouped['err']   = safe_err(grouped['impact_std'], grouped['sample_count'], grouped['impact_mean'])
grouped['a_err'] = safe_err(grouped['a_std'],      grouped['sample_count'], grouped['a_mean'])
grouped['b_err'] = safe_err(grouped['b_std'],      grouped['sample_count'], grouped['b_mean'])

# --- VISUALIZATION OF INITIAL BINNED DATA ---

fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.dist = 8

log_a = np.log10(grouped['a_mean'])
log_b = np.log10(grouped['b_mean'])
log_i = np.log10(grouped['impact_mean'])

# Transform errors to log-space using the derivative approximation method
zerr = grouped['err'].values   / (grouped['impact_mean'].values * np.log(10))
xerr = grouped['a_err'].values / (grouped['a_mean'].values      * np.log(10))
yerr = grouped['b_err'].values / (grouped['b_mean'].values      * np.log(10))

ax1.scatter(log_a, log_b, log_i, marker='.', s=50, alpha=0.8)

# Manually draw three-dimensional error bars as individual lines across each axis
for xi, yi, zi, xe, ye, ze in zip(log_a, log_b, log_i, xerr, yerr, zerr):
    ax1.plot([xi-xe, xi+xe], [yi, yi], [zi, zi], lw=0.6, alpha=0.6)
    ax1.plot([xi, xi], [yi-ye, yi+ye], [zi, zi], lw=0.6, alpha=0.6)
    ax1.plot([xi, xi], [yi, yi], [zi-ze, zi+ze], lw=0.6, alpha=0.6)

ax1.set_xlabel(r'$\log_{10}(V/V_P)$')
ax1.set_ylabel(r'$\log_{10}(V_P/V_D)$')
ax1.set_zlabel(r'$\log_{10}(I)$')
ax1.view_init(elev=5, azim=-170)

plt.tight_layout()
plt.savefig('images\\part_rate\\part_rate_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# --- VISUALIZATION OF DATA PROJECTION ---

fig_proj, ax_proj = plt.subplots(figsize=(8, 6))

# Scatter plot representing the projection onto the relative volume and impact plane
ax_proj.scatter(log_b, log_i, marker='.', s=50, zorder=3)

# Error bars for the two-dimensional projection
for yi, zi, ye, ze in zip(log_b, log_i, yerr, zerr):
    ax_proj.plot([yi-ye, yi+ye], [zi, zi], lw=0.6, alpha=0.6)
    ax_proj.plot([yi, yi], [zi-ze, zi+ze], lw=0.6, alpha=0.6)

ax_proj.set_xlabel(r'$\log_{10}(V_P/V_D)$')
ax_proj.set_ylabel(r'$\log_{10}(I)$')
ax_proj.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('images\\part_rate\\part_rate_projection.png', dpi=300, bbox_inches='tight')
plt.close()

# --- VISUALIZATION OF DETAILED 3D HISTOGRAM ---

n_hist = 25
log_a_all = np.log10(df_res['a'].to_numpy())
log_b_all = np.log10(df_res['b'].to_numpy())

# Generate two-dimensional histogram data for density visualization
counts, x_edges, y_edges = np.histogram2d(log_a_all, log_b_all, bins=n_hist)

# Prepare grid coordinates for the three-dimensional bar plot
x_pos, y_pos = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing="ij")
x_pos, y_pos = x_pos.ravel(), y_pos.ravel()
dz = counts.ravel()
dx = (x_edges[1] - x_edges[0]) * 0.85
dy = (y_edges[1] - y_edges[0]) * 0.85

fig2 = plt.figure(figsize=(14, 10))
ax2  = fig2.add_subplot(111, projection='3d')

# Apply the Turbo colormap to effectively highlight sample density peaks
norm = plt.Normalize(dz[dz > 0].min(), dz.max())
colors = plt.cm.turbo(norm(dz))

ax2.bar3d(x_pos, y_pos, 0, dx, dy, dz, color=colors, alpha=0.85, shade=True)

# Create a matching colorbar to indicate frequency levels
mappable = cm.ScalarMappable(norm=norm, cmap=plt.cm.turbo)
mappable.set_array(dz)
fig2.colorbar(mappable, ax=ax2, shrink=0.5, aspect=10, pad=0.1, label='Sample Density')

ax2.set_xlabel(r'$\log_{10}(V/V_P)$')
ax2.set_ylabel(r'$\log_{10}(V_P/V_D)$')
ax2.set_zlabel('Frequency')
ax2.view_init(elev=30, azim=-135)

plt.savefig('images\\part_rate\\part_rate_hist.png', bbox_inches='tight')
plt.close()

# --- VISUALIZATION OF HISTOGRAM PROJECTION ---

# Reconstruct the two-dimensional grid for the heatmap visualization
unique_x, unique_y = np.unique(x_pos), np.unique(y_pos)
dz_grid = dz.reshape(len(unique_y), len(unique_x))
X_grid, Y_grid = np.meshgrid(np.append(unique_x, unique_x[-1] + (x_edges[1]-x_edges[0])), 
                             np.append(unique_y, unique_y[-1] + (y_edges[1]-y_edges[0])))

fig3, ax3 = plt.subplots(figsize=(10, 8))
im = ax3.pcolormesh(X_grid, Y_grid, dz_grid, cmap='turbo', shading='auto')

ax3.set_xlabel(r'$\log_{10}(V/V_P)$')
ax3.set_ylabel(r'$\log_{10}(V_P/V_D)$')
fig3.colorbar(im, ax=ax3, label='Frequency')

plt.savefig('images\\part_rate\\part_rate_hist_2d.png', dpi=300, bbox_inches='tight')
plt.close()

# --- REGRESSION PREPARATION AND FILTERING ---

# Filter individual bins based on regime limits and statistical significance
max_samples = grouped['sample_count'].max()
mask_range = (log_a >= LOG_A_MIN) & (log_a <= LOG_A_MAX)
mask_significance = (grouped['sample_count'] >= (max_samples * SAMPLE_THRESHOLD_PCT))

grouped_fit = grouped[mask_range & mask_significance].copy()

# Prepare necessary variables for the curve fitting process
a_f, b_f, i_f = grouped_fit['a_mean'].values, grouped_fit['b_mean'].values, grouped_fit['impact_mean'].values
log_a_f, log_b_f, log_i_f = np.log10(a_f), np.log10(b_f), np.log10(i_f)
zerr_f = zerr[mask_range & mask_significance]
xerr_f = xerr[mask_range & mask_significance]
yerr_f = yerr[mask_range & mask_significance]

# --- ITERATIVE WEIGHTED REGRESSION ---

# Initial guesses for the model parameters
popt = [np.log10(np.median(i_f)), 0.5, 0.5] 

for _ in range(10):
    _, d, al = popt
    # Effective Variance Method including errors from all three axes
    sig_eff = np.sqrt(zerr_f**2 + (d**2 * xerr_f**2) + (al**2 * yerr_f**2))
    popt, _ = curve_fit(log_function_3_ab, (log_a_f, log_b_f), log_i_f, p0=popt, sigma=sig_eff, absolute_sigma=True)

log_Y_fit, d_fit, al_fit = popt
print(f"Fit Results: Y={10**log_Y_fit:.4g}, delta={d_fit:.4f}, alpha={al_fit:.4f}")

# --- VISUALIZATION OF FINAL MODEL SURFACE ---

fig4 = plt.figure(figsize=(12, 10))
ax4 = fig4.add_subplot(111, projection='3d')

# Plot only the filtered data points that were utilized for the regression
ax4.scatter(log_a_f, log_b_f, log_i_f, marker='.', s=50, alpha=0.8)

# Generate and plot the fitted regression surface over the specified range
a_s, b_s = np.meshgrid(np.linspace(LOG_A_MIN, LOG_A_MAX, 20), np.linspace(LOG_B_MIN, LOG_B_MAX, 20))
z_s = log_Y_fit + d_fit * a_s + al_fit * b_s
ax4.plot_surface(a_s, b_s, z_s, alpha=0.8, edgecolor='none')

ax4.set_xlabel(r'$\log_{10}(V/V_P)$')
ax4.set_ylabel(r'$\log_{10}(V_P/V_D)$')
ax4.set_zlabel(r'$\log_{10}(I)$')
ax4.view_init(elev=15, azim=20)

plt.savefig('images\\part_rate\\part_rate_fit_1.png', dpi=300, bbox_inches='tight')
plt.close()

# --- FIT AND PLOT FOR DATA WHERE log_a > -0.1 ---

# 1. Create the mask for the high participation rate regime
mask_high_a = (log_a > -0.1) & mask_significance

grouped_high = grouped[mask_high_a].copy()

if not grouped_high.empty:
    # Prepare variables
    a_h, b_h, i_h = grouped_high['a_mean'].values, grouped_high['b_mean'].values, grouped_high['impact_mean'].values
    log_a_h, log_b_h, log_i_h = np.log10(a_h), np.log10(b_h), np.log10(i_h)
    zerr_h = zerr[mask_high_a]
    xerr_h = xerr[mask_high_a]
    yerr_h = yerr[mask_high_a]

    # Iterative Weighted Regression
    popt_h = [np.log10(np.median(i_h)), 0.5, 0.5]
    for _ in range(10):
        _, d_h, al_h = popt_h
        sig_eff_h = np.sqrt(zerr_h**2 + (d_h**2 * xerr_h**2) + (al_h**2 * yerr_h**2))
        popt_h, _ = curve_fit(log_function_3_ab, (log_a_h, log_b_h), log_i_h, 
                               p0=popt_h, sigma=sig_eff_h, absolute_sigma=True)

    log_Y_h, d_h, al_h = popt_h
    print(f"High Participation Fit (log_a > -0.1): Y={10**log_Y_h:.4g}, delta={d_h:.4f}, alpha={al_h:.4f}")

    # --- PLOTTING ---
    fig6 = plt.figure(figsize=(12, 10))
    ax6 = fig6.add_subplot(111, projection='3d')

    # Scatter high-a data
    ax6.scatter(log_a_h, log_b_h, log_i_h, s=50, alpha=0.8)

    # Generate surface grid for this specific range
    a_range_h = np.linspace(log_a_h.min(), log_a_h.max(), 20)
    b_range_h = np.linspace(log_b_h.min(), log_b_h.max(), 20)
    A_h, B_h = np.meshgrid(a_range_h, b_range_h)
    Z_h = log_Y_h + d_h * A_h + al_h * B_h

    # Plot surface
    ax6.plot_surface(A_h, B_h, Z_h, alpha=0.1, edgecolor='none')

    ax6.set_xlabel(r'$\log_{10}(V/V_P)$')
    ax6.set_ylabel(r'$\log_{10}(V_P/V_D)$')
    ax6.set_zlabel(r'$\log_{10}(I)$')
    ax6.view_init(elev=30, azim=-110)

    plt.savefig('images\\part_rate\\part_rate_fit_2.png', dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("No data points found with log_a > -0.1 and sufficient significance.")