import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit

# --- CONFIGURATION AND FILTERING LIMITS ---
# These limits define the liquidity regime where the power law is most valid
PATH = 'meta_20_power_2.0.csv'
DIR = 'database\\meta'

# Significance filter to exclude bins with very few samples to avoid noise
SAMPLE_THRESHOLD_PCT = 0.001     

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
df_res['a'] = df_res['MetaVolume']# / df_res['TradedVolume']
df_res['b'] = df_res['TradedVolume']

x_label = 'V/V_D'
y_label = 'V_P/V_D'

# --- 2D LOGARITHMIC BINNING ---
# Create log-spaced bins for each dimension to establish a two-dimensional grid
bins_a = np.logspace(np.log10(df_res['a'].min()), np.log10(df_res['a'].max()), 31)
bins_b = np.logspace(np.log10(df_res['b'].min()), np.log10(df_res['b'].max()), 31)

df_res['bin_a'] = pd.cut(df_res['a'], bins=bins_a, include_lowest=True)
df_res['bin_b'] = pd.cut(df_res['b'], bins=bins_b, include_lowest=True)

df_res_zero = df_res[df_res['a'] == 1]
#df_res = df_res[df_res['a'] < 1]

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

# Remove non-positive values to allow for correct logarithmic transformation
grouped = grouped[(grouped['a_mean'] > 0) & (grouped['b_mean'] > 0) & (grouped['impact_mean'] > 0)].copy()

log_a = np.log10(grouped['a_mean'])
log_b = np.log10(grouped['b_mean'])
log_i = np.log10(grouped['impact_mean'])

# Transform errors to log-space using the derivative approximation method
zerr = grouped['err'].values   / (grouped['impact_mean'].values * np.log(10))
xerr = grouped['a_err'].values / (grouped['a_mean'].values      * np.log(10))
yerr = grouped['b_err'].values / (grouped['b_mean'].values      * np.log(10))

fig1 = plt.figure(figsize=(12, 10))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.dist = 8

ax1.plot_trisurf(
    log_a, log_b, log_i,
    cmap='Blues',      # Color of the triangles
    edgecolor='grey',  # This "connects" the points with lines
    linewidth=0.3,
    alpha=0.5
)

# Original scatter — same style as before
ax1.scatter(log_a, log_b, log_i, marker='.', s=50, alpha=0.8, color='C0')

ax1.set_xlabel(fr'$\log_{{10}}({x_label})$')
ax1.set_ylabel(fr'$\log_{{10}}({y_label})$')
ax1.set_zlabel(r'$\log_{10}(I)$')
ax1.view_init(elev=30, azim=-150)

plt.tight_layout()
plt.savefig('images\\part_rate\\part_rate_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# --- VISUALIZATION OF DETAILED 3D HISTOGRAM ---

n_hist = 25
log_a_all = np.log10(df_res['a'].to_numpy())
log_b_all = np.log10(df_res['b'].to_numpy())

# Generate two-dimensional histogram data for density visualization
counts, x_edges, y_edges = np.histogram2d(log_a_all, log_b_all, bins=n_hist)

# Prepare grid coordinates for the three-dimensional bar plot
# Use 'xy' for standard Cartesian alignment
x_pos, y_pos = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing="xy")
x_pos, y_pos = x_pos.ravel(), y_pos.ravel()

# You may need to transpose the counts if they look "rotated"
dz = counts.T.ravel()

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

ax2.set_xlabel(rf'$\log_{{10}}({x_label})$')
ax2.set_ylabel(rf'$\log_{{10}}({y_label})$')
ax2.set_zlabel('Frequency')
ax2.view_init(elev=30, azim=-135)

plt.savefig('images\\part_rate\\part_rate_hist.png', bbox_inches='tight')
plt.show()

# --- VISUALIZATION OF HISTOGRAM PROJECTION ---

# Reconstruct the two-dimensional grid for the heatmap visualization
unique_x, unique_y = np.unique(x_pos), np.unique(y_pos)
dz_grid = dz.reshape(len(unique_y), len(unique_x))
X_grid, Y_grid = np.meshgrid(np.append(unique_x, unique_x[-1] + (x_edges[1]-x_edges[0])), 
                             np.append(unique_y, unique_y[-1] + (y_edges[1]-y_edges[0])))

fig3, ax3 = plt.subplots(figsize=(10, 8))
im = ax3.pcolormesh(X_grid, Y_grid, dz_grid, cmap='turbo', shading='auto')

ax3.set_xlabel(rf'$\log_{{10}}({x_label})$')
ax3.set_ylabel(rf'$\log_{{10}}({y_label})$')
fig3.colorbar(im, ax=ax3, label='Frequency')

plt.savefig('images\\part_rate\\part_rate_hist_2d.png', dpi=300, bbox_inches='tight')
plt.close()

# --- REGRESSION PREPARATION AND FILTERING ---

# Filter individual bins based on regime limits and statistical significance
max_samples = grouped['sample_count'].max()
mask_significance = (grouped['sample_count'] >= (max_samples * SAMPLE_THRESHOLD_PCT))

grouped_fit = grouped[mask_significance].copy()

# Prepare necessary variables for the curve fitting process
a_f, b_f, i_f = grouped_fit['a_mean'].values, grouped_fit['b_mean'].values, grouped_fit['impact_mean'].values
log_a_f, log_b_f, log_i_f = np.log10(a_f), np.log10(b_f), np.log10(i_f)
zerr_f = zerr[mask_significance]
xerr_f = xerr[mask_significance]
yerr_f = yerr[mask_significance]

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
# 1. Use the actual filtered data limits for the surface
a_range = np.linspace(log_a_f.min(), log_a_f.max(), 10)
b_range = np.linspace(log_b_f.min(), log_b_f.max(), 10)
a_s, b_s = np.meshgrid(a_range, b_range)

# 2. Re-calculate Z based on your mesh
z_s = log_Y_fit + d_fit * a_s + al_fit * b_s

fig4 = plt.figure(figsize=(12, 10))
ax4 = fig4.add_subplot(111, projection='3d')

# Plot data points - use a darker color so they pop through the surface
ax4.scatter(log_a_f, log_b_f, log_i_f, marker='o', s=30, alpha=0.8, color='#1f77b4', linewidth=0.5)

# IMPROVED SURFACE
ax4.plot_surface(a_s, b_s, z_s, 
                 color='gray',       # Use a neutral color like gray or lightblue
                 alpha=0.25,         # Very faint
                 linewidth=0,        # CRITICAL: removes the blocky grid lines
                 antialiased=True, 
                 shade=False,        # Keeps the transparency uniform
                 zorder=0)           # Attempts to push the surface behind points

# 3. Clean up the "Box" appearance
ax4.set_xlabel(rf'$\log_{{10}}({x_label})$')
ax4.set_ylabel(rf'$\log_{{10}}({y_label})$')
ax4.set_zlabel(r'$\log_{10}(I)$')

# Make panes transparent for a modern look
ax4.xaxis.pane.fill = False
ax4.yaxis.pane.fill = False
ax4.zaxis.pane.fill = False

ax4.view_init(elev=15, azim=20)
plt.savefig('images\\part_rate\\part_rate_fit.png', dpi=300, bbox_inches='tight')
plt.show()

"""
# --- FIT AND PLOT FOR DATA AT THE BORDER ---

# --- FIT AND PLOT FOR DATA AT THE BORDER ---
# The high participation rate regime (log_a > LOG_A_MAX) covers a very narrow
# range of a, so the original coarse 31-bin grid yields too few populated bins.
# We rebin the raw data directly within this regime using a denser grid.
 
# 1. Isolate the raw rows that belong to the high-a regime
df_high = df_res[np.log10(df_res['a']) > LOG_A_MAX].copy()
 
# 2. Build a denser log-spaced grid tailored to this narrow a-range
N_BINS_HIGH_A = 20   # finer in the participation-rate direction
N_BINS_HIGH_B = 100   # keep reasonable resolution in the volume direction
 
bins_a_h = np.logspace(np.log10(df_high['a'].min()), np.log10(df_high['a'].max()), N_BINS_HIGH_A + 1)
bins_b_h = np.logspace(np.log10(df_high['b'].min()), np.log10(df_high['b'].max()), N_BINS_HIGH_B + 1)
 
df_high['bin_a'] = pd.cut(df_high['a'], bins=bins_a_h, include_lowest=True)
df_high['bin_b'] = pd.cut(df_high['b'], bins=bins_b_h, include_lowest=True)
 
grouped_high = df_high.groupby(['bin_a', 'bin_b'], observed=True).agg({
    'a': ['mean', 'std'],
    'b': ['mean', 'std'],
    'MetaImpact': ['mean', 'std', 'count']
}).dropna()
grouped_high.columns = ['a_mean', 'a_std', 'b_mean', 'b_std', 'impact_mean', 'impact_std', 'sample_count']
 
# 3. Apply the same significance filter as before (relative to this sub-dataset)
max_samples_h = grouped_high['sample_count'].max()
mask_sig_h    = grouped_high['sample_count'] >= (max_samples_h * SAMPLE_THRESHOLD_PCT)
grouped_high  = grouped_high[mask_sig_h].copy()
 
# 4. Compute errors for the dense bins
grouped_high['err']   = safe_err(grouped_high['impact_std'], grouped_high['sample_count'], grouped_high['impact_mean'])
grouped_high['a_err'] = safe_err(grouped_high['a_std'],      grouped_high['sample_count'], grouped_high['a_mean'])
grouped_high['b_err'] = safe_err(grouped_high['b_std'],      grouped_high['sample_count'], grouped_high['b_mean'])
 
# 5. Prepare log-space arrays and propagate errors
a_h, b_h, i_h = grouped_high['a_mean'].values, grouped_high['b_mean'].values, grouped_high['impact_mean'].values
log_a_h = np.log10(a_h)
log_b_h = np.log10(b_h)
log_i_h = np.log10(i_h)
 
zerr_h = grouped_high['err'].values   / (i_h * np.log(10))
xerr_h = grouped_high['a_err'].values / (a_h  * np.log(10))
yerr_h = grouped_high['b_err'].values / (b_h  * np.log(10))
 
# 6. Iterative Weighted Regression on the dense bins
popt_h = [np.log10(np.median(i_h)), 0.5, 0.5]
for _ in range(10):
    _, d_h, al_h = popt_h
    sig_eff_h = np.sqrt(zerr_h**2 + (d_h**2 * xerr_h**2) + (al_h**2 * yerr_h**2))
    popt_h, _ = curve_fit(log_function_3_ab, (log_a_h, log_b_h), log_i_h,
                          p0=popt_h, sigma=sig_eff_h, absolute_sigma=True)
 
log_Y_h, d_h, al_h = popt_h
print(f"High Participation Fit (log_a > {LOG_A_MAX}, dense rebin {N_BINS_HIGH_A}x{N_BINS_HIGH_B}): "
      f"Y={10**log_Y_h:.4g}, delta={d_h:.4f}, alpha={al_h:.4f}")
 
# --- PLOTTING ---
fig6 = plt.figure(figsize=(12, 10))
ax6  = fig6.add_subplot(111, projection='3d')
 
# Scatter the densely-rebinned data points with error bars
ax6.scatter(log_a_h, log_b_h, log_i_h, marker='o', s=30, alpha=0.8, color='#1f77b4')
for xi, yi, zi, xe, ye, ze in zip(log_a_h, log_b_h, log_i_h, xerr_h, yerr_h, zerr_h):
    ax6.plot([xi-xe, xi+xe], [yi, yi],     [zi, zi],     lw=0.6, alpha=0.5, color='steelblue')
    ax6.plot([xi, xi],       [yi-ye, yi+ye],[zi, zi],     lw=0.6, alpha=0.5, color='steelblue')
    ax6.plot([xi, xi],       [yi, yi],     [zi-ze, zi+ze],lw=0.6, alpha=0.5, color='steelblue')
 
# Generate fitted surface over the dense data extent
a_range_h = np.linspace(log_a_h.min(), log_a_h.max(), 25)
b_range_h = np.linspace(log_b_h.min(), log_b_h.max(), 25)
A_h, B_h  = np.meshgrid(a_range_h, b_range_h)
Z_h       = log_Y_h + d_h * A_h + al_h * B_h
 
ax6.plot_surface(A_h, B_h, Z_h,
                 color='gray', alpha=0.20,
                 linewidth=0, antialiased=True, shade=False, zorder=0)
 
ax6.set_xlabel(r'$\log_{10}(V/V_P)$')
ax6.set_ylabel(r'$\log_{10}(V_P/V_D)$')
ax6.set_zlabel(r'$\log_{10}(I)$')
ax6.xaxis.pane.fill = False
ax6.yaxis.pane.fill = False
ax6.zaxis.pane.fill = False
ax6.view_init(elev=30, azim=-110)
 
plt.tight_layout()
plt.savefig('images\\part_rate\\part_rate_fit_2.png', dpi=300, bbox_inches='tight')
plt.show()
"""