import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import methods
import os
from os import listdir
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp

# --- Style Configuration (Inherited from your first script) ---
plt.rcParams.update({
    'font.size': 12,          # Base text size
    'axes.titlesize': 20,     # Title size
    'axes.labelsize': 16,     # Axis labels size
    'xtick.labelsize': 12,    # X-axis tick labels
    'ytick.labelsize': 12,    # Y-axis tick labels
    'legend.fontsize': 14     # Legend size
})

# --- Fit Models ---
def power_law_dist(x, C, alpha):
    """Standard power law distribution model."""
    return C * np.power(x, -alpha)

# --- Automatic search for x_min/x_max ---
def find_optimal_xmin(x, y, model_func, p0):
    """Finds the start of the tail that minimizes the KS distance for a given model."""
    start_idx = max(1, int(len(x) * 0.05))
    end_idx = int(len(x) * 0.7)
    best_xmin = x[start_idx]
    min_ks = np.inf
    
    for i in range(start_idx, end_idx):
        xmin_test = x[i]
        mask = x >= xmin_test
        if len(x[mask]) < 8: continue 
        try:
            popt, _ = curve_fit(model_func, x[mask], y[mask], p0=p0)
            y_model = model_func(x[mask], *popt)
            # Normalize to compare distributions via KS test
            stat, _ = ks_2samp(y[mask]/np.sum(y[mask]), y_model/np.sum(y_model))
            if stat < min_ks:
                min_ks = stat
                best_xmin = xmin_test
        except:
            continue
    return best_xmin

def find_optimal_xmax(x, y, model_func):
    """Finds the end of the 'head' that minimizes the KS distance."""
    start_idx = int(1)
    end_idx = int(len(x) * 0.95)
    
    best_xmax = x[end_idx]
    min_ks = np.inf
    
    for i in range(start_idx, end_idx):
        xmax_test = x[i]
        mask = x <= xmax_test
        try:
            popt, _ = curve_fit(model_func, x[mask], y[mask])
            y_model = model_func(x[mask], *popt)
            stat, _ = ks_2samp(y[mask]/np.sum(y[mask]), y_model/np.sum(y_model))
            
            if stat < min_ks:
                min_ks = stat
                best_xmax = xmax_test
        except:
            continue
    return best_xmax

# --- Configuration and Directory ---
data_dir = 'database\\data'
iterations = 100
nb_traders = 20
kind = 'power'
exponent = 2.0
csv_name = f'database\\dist_meta_child_{iterations}_{nb_traders}_{kind}_{exponent}.csv'

# --- 1. DATA GENERATION OR LOADING ---
if os.path.exists(csv_name):
    print(f"File '{csv_name}' found. Loading data...")
    df_results = pd.read_csv(csv_name)
    bin_centers = df_results['NbChild'].values
    mean_density = df_results['Mean_Density'].values
    std_density = df_results['Std_Dev'].values
else:
    print(f"File '{csv_name}' not found. Starting simulations...")
    all_densities = []
    paths = np.array(listdir(data_dir))

    for i in range(iterations):
        print(f'Iteration: {i + 1}/{iterations}')
        meta_tot = pd.DataFrame()
        for path in paths:
            l = len(meta_tot)
            meta, _ = methods.generate(path, nb_traders, kind, exponent, l, data_dir)
            meta_tot = pd.concat([meta_tot, meta])
        
        data = meta_tot['NbChild']
        if i == 0:
            min_v, max_v = int(data.min()), int(data.max())
            bins = np.arange(min_v - 0.5, max_v + 1.5, 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2

        counts, _ = np.histogram(data, bins=bins, density=True)
        all_densities.append(counts)

    all_densities = np.array(all_densities)
    mean_density = np.mean(all_densities, axis=0)
    std_density = np.std(all_densities, axis=0)

    df_results = pd.DataFrame({
        'NbChild': bin_centers,
        'Mean_Density': mean_density,
        'Std_Dev': std_density
    })
    df_results.to_csv(csv_name, index=False)

# --- 2. AUTOMATIC TAIL ANALYSIS ---
# Power Law Tail Fit
xmin_tail = find_optimal_xmin(bin_centers, mean_density, power_law_dist, [mean_density[0], 2.0])
mask_tail = bin_centers >= xmin_tail
popt_tail, pcov_tail = curve_fit(power_law_dist, bin_centers[mask_tail], mean_density[mask_tail], sigma=std_density[mask_tail], absolute_sigma=True)

# Power Law Head Fit
xmax_head = find_optimal_xmax(bin_centers, mean_density, power_law_dist)
mask_head = bin_centers <= xmax_head
popt_head, pcov_head = curve_fit(power_law_dist, bin_centers[mask_head], mean_density[mask_head], sigma=std_density[mask_head], absolute_sigma=True)

# --- 3. PLOTTING ---
fig = plt.figure(figsize=(15, 12))
grid = plt.GridSpec(5, 2, hspace=0.45, wspace=0.3)

# Main Plot (Log-Log)
ax_main = fig.add_subplot(grid[:3, :])

# Uncertainty Band
ax_main.fill_between(bin_centers, mean_density - std_density, mean_density + std_density, color='gray', alpha=0.15, label=r'Uncertainty ($\sigma$)')

# Scatter Plot with the specified style (White edges for visibility)
ax_main.errorbar(
    bin_centers, mean_density, yerr=std_density, 
    fmt='o', color='black', ecolor='gray', elinewidth=1, capsize=0, 
    markersize=7, markeredgecolor='white', alpha=0.8, label='Observed Data'
)

# Fit lines (Power Law Tail)
x_fine_tail = np.geomspace(xmin_tail, bin_centers.max(), 200)
ax_main.plot(x_fine_tail, power_law_dist(x_fine_tail, *popt_tail), color='tab:red', lw=3, label=r'Power-Law Tail Fit ($\alpha \approx $' + f'{popt_tail[1]:.2f})')

# Fit lines (Power Law Head)
x_fine_head = np.geomspace(bin_centers.min(), xmax_head, 200)
ax_main.plot(x_fine_head, power_law_dist(x_fine_head, *popt_head), color='tab:blue', lw=3, label=r'Power-Law Head Fit ($\alpha \approx $' + f'{popt_head[1]:.2f})')

# Aesthetics for Main Plot
ax_main.set_xscale('log')
ax_main.set_yscale('log')
ax_main.set_xlabel(r'$N_{child}$')
ax_main.set_ylabel(r'Density $P(N_{child})$')
ax_main.legend(loc='best', frameon=True)
ax_main.grid(True, which="both", ls="-", alpha=0.2)

# --- Residuals Plots ---

# Tail Residuals
ax_res_tail = fig.add_subplot(grid[3:, 1])
res_tail = (mean_density[mask_tail] - power_law_dist(bin_centers[mask_tail], *popt_tail)) / std_density[mask_tail]
ax_res_tail.scatter(bin_centers[mask_tail], res_tail, color='tab:red', s=40, alpha=0.7, edgecolor='white')
ax_res_tail.axhline(0, color='black', lw=1.5, ls='--')
ax_res_tail.set_title('Tail Residuals', fontsize=14)
ax_res_tail.set_xlabel(r'$N_{child}$')
ax_res_tail.set_ylabel('Std. Dev.')
ax_res_tail.grid(True, alpha=0.2)

# Head Residuals
ax_res_head = fig.add_subplot(grid[3:, 0])
res_head = (mean_density[mask_head] - power_law_dist(bin_centers[mask_head], *popt_head)) / std_density[mask_head]
ax_res_head.scatter(bin_centers[mask_head], res_head, color='tab:blue', s=40, alpha=0.7, edgecolor='white')
ax_res_head.axhline(0, color='black', lw=1.5, ls='--')
ax_res_head.set_title('Head Residuals', fontsize=14)
ax_res_head.set_xlabel(r'$N_{child}$')
ax_res_head.set_ylabel('Std. Dev.')
ax_res_head.grid(True, alpha=0.2)

# Save and Show
if not os.path.exists('images'): os.makedirs('images')
plt.tight_layout()
plt.savefig('images\\dist_meta_child.png', dpi=300, bbox_inches='tight')
plt.show()

# Print Fit Results
print(f"Tail fit: C = {popt_tail[0]:.3e} +- {np.sqrt(pcov_tail[0][0]):.3e}, Alpha = {popt_tail[1]:.3f} +- {np.sqrt(pcov_tail[1][1]):.3f}")
print(f"Head fit: C = {popt_head[0]:.3e} +- {np.sqrt(pcov_head[0][0]):.3e}, Alpha = {popt_head[1]:.3f} +- {np.sqrt(pcov_head[1][1]):.3f}")