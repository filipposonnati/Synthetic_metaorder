import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
import os
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 1. Models for Fitting
def power_law(x, A, alpha):
    return A * (x**-alpha)

def der_power_law(x, A, alpha):
    return -A * alpha * (x**(-alpha - 1))

os.makedirs('images', exist_ok=True)
input_dir = 'database/meta'
paths = sorted([f for f in listdir(input_dir) if f.endswith('.csv')])

# Setup Figures
fig2, ax2 = plt.subplots(figsize=(10, 7)) # Figure 2: Main Analysis + Border Zoom

# ENLARGED INSET (50% size) pushed to the lower-left border
# bbox_to_anchor coordinates (0.06, 0.06) move it closer to the axes edges
ax_ins = inset_axes(ax2, width="50%", height="50%", loc='lower left', 
                    bbox_to_anchor=(0.06, 0.06, 1, 1), bbox_transform=ax2.transAxes)

cmap = plt.get_cmap('tab10')
num_colors = len(cmap.colors)
table_results = []

for i, path in enumerate(paths):
    meta = pd.read_csv(os.path.join(input_dir, path))
    meta = meta[meta['NbChild'] > 1]
    volumes = meta['MetaVolume'].to_numpy()
    volumes = volumes[volumes > 0]
    
    # 2. Binning
    log_bins = np.logspace(np.log10(volumes.min()), np.log10(volumes.max()), 35)
    bin_indices = np.digitize(volumes, log_bins)
    
    x_m, x_e, y_d, y_e = [], [], [], []
    total_n = len(volumes)
    
    for b in range(1, len(log_bins)):
        bin_data = volumes[bin_indices == b]
        if len(bin_data) > 3:
            xm = np.mean(bin_data)
            xe = np.std(bin_data) / np.sqrt(len(bin_data))
            bw = log_bins[b] - log_bins[b-1]
            yd = len(bin_data) / (total_n * bw)
            ye = np.sqrt(len(bin_data)) / (total_n * bw)
            x_m.append(xm); x_e.append(xe); y_d.append(yd); y_e.append(ye)

    x_m, x_e, y_d, y_e = map(np.array, [x_m, x_e, y_d, y_e])
    label_name = path.replace('meta', '').replace('.csv', '').replace('_', ' ').strip()
    color = cmap(i % num_colors)

    # FITTING (Effective Variance Method)
    tail_threshold = np.quantile(volumes, 0.85)
    mask = x_m > tail_threshold
    xm_t, xe_t, yd_t, ye_t = x_m[mask], x_e[mask], y_d[mask], y_e[mask]
    
    alpha_val, alpha_err = np.nan, np.nan
    if len(xm_t) > 3:
        coeffs = np.polyfit(np.log(xm_t), np.log(yd_t), 1)
        A_curr, alpha_curr = np.exp(coeffs[1]), -coeffs[0]
        for _ in range(5):
            sig_eff = np.sqrt(ye_t**2 + (xe_t * der_power_law(xm_t, A_curr, alpha_curr))**2)
            try:
                popt, pcov = curve_fit(power_law, xm_t, yd_t, p0=[A_curr, alpha_curr], 
                                      sigma=sig_eff, absolute_sigma=True)
                A_curr, alpha_curr = popt
            except: break
        alpha_val, alpha_err = alpha_curr, np.sqrt(pcov[1,1])
        
        # Plot fits (Linewidth 2.2 for prominence)
        x_fit = np.logspace(np.log10(xm_t.min()), np.log10(xm_t.max()*1.15), 100)
        y_fit = power_law(x_fit, *popt)
        ax2.plot(x_fit, y_fit, color=color, linestyle='--', alpha=1.0, linewidth=1.0)
        ax_ins.plot(x_fit, y_fit, color=color, linestyle='--', alpha=1.0, linewidth=1.0)

    table_results.append({'Dataset': label_name, 'alpha': alpha_val, 'err': alpha_err})

    # FIGURE 2: Errorbars Main + Border Zoom (Solid colors)
    ax2.errorbar(x_m, y_d, xerr=x_e, yerr=y_e, fmt='o', color=color, markersize=4, 
                 alpha=1.0, label=label_name, capsize=0, elinewidth=1.2)
    
    if len(xm_t) > 0:
        ax_ins.errorbar(xm_t, yd_t, xerr=xe_t, yerr=ye_t, fmt='o', color=color, markersize=5, 
                        alpha=1.0, capsize=0, elinewidth=1.2)

# Axis Formatting
for ax in [ax2, ax_ins]:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle=':', alpha=0.5)

ax2.set_xlabel(r'Volume $Q$', fontsize=14); ax2.set_ylabel(r'Density $P(Q)$', fontsize=14)
ax2.legend(title="Datasets", loc='upper right', frameon=True)

ax_ins.tick_params(labelsize=9)

fig2.tight_layout()
fig2.savefig('images/meta_volume_dist.png', dpi=300)
#plt.show()

df_results = pd.DataFrame(table_results)

print(df_results.round(4).to_string(index=False))