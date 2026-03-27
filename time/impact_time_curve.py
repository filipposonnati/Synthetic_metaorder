import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

# --- Configurazione Grafica ---
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

def square_root(x, Y):
    return Y * np.sqrt(x)

def power_law(x, A, B):
    return A * np.power(x, B)

function = '20_power_2.0'
dir = '..\\database\\trades_' + function
paths = np.array(listdir(dir))

trades_total = None

for path in paths:
    trades = pd.read_csv(dir + f'\\{path}', sep=',')
    cols = ['Ratio_pre', 'BeginMid', 'MetaDuration', 'ElapsedTime', 'NbChild']
    daily_trades = trades[cols].copy()
    trades_total = pd.concat([trades_total, daily_trades])

# Rename columns for clarity
trades_total.rename(columns={'Ratio_pre': 'Ratio'}, inplace=True)

trades_total = trades_total[trades_total['NbChild'] > 1]
trades_total = trades_total[trades_total['ElapsedTime'] > 0.0]
#trades_total = trades_total[trades_total['PartialImpact'] != 0.0]

trades_total['NormalizedTime'] = trades_total['ElapsedTime'] / trades_total['MetaDuration']

geo_part  = np.geomspace(np.min(trades_total['NormalizedTime']), 0.125, 9)
mid_part  = np.linspace(0.125, 0.5, 4)
tail_part = np.linspace(0.5, 1.0, 6)
time_bins = np.unique(np.concatenate([geo_part, mid_part, tail_part]))

n_bins        = len(time_bins) - 1   # 20
n_plots       = 4
bins_per_plot = n_bins // n_plots    # 5


# ---------------------------------------------------------------
# DIAGNOSTICA: PartialImpact == 0
# ---------------------------------------------------------------
n_total      = len(trades_total)
n_zero = (trades_total['Ratio'] == 0.0).sum()
pct          = n_zero / n_total * 100

print('\nDiagnostica PartialImpact:')
print(f'  Trade totali              : {n_total:,}')
print(f'  PartialImpact == 0 esatto : {n_zero:,}  ({pct:.2f}%)')

trades_total['time_bin_idx'] = pd.cut(
    trades_total['NormalizedTime'],
    bins=time_bins,
    labels=False,
    include_lowest=True
)

# Breakdown zeri per bin (dopo pd.cut)
trades_total['is_zero'] = trades_total['Ratio'] == 0.0
zero_by_bin = trades_total.groupby('time_bin_idx', observed=True)['is_zero'].mean() * 100
print('\n  % di Ratio == 0 per bin (primi 10 bin):')
print(zero_by_bin.head(10).to_string())

x_min = trades_total['Ratio'].quantile(0.05)
x_max = trades_total['Ratio'].quantile(0.95)
x_kde = np.linspace(x_min, x_max, 500)

# 7 colori distinti per subplot
palette = [plt.get_cmap("tab10")(i) for i in range(5)]

# # ---------------------------------------------------------------
# # FIGURA 1: KDE
# # ---------------------------------------------------------------
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# axes = axes.flatten()
# 
# for plot_idx in range(n_plots):
#     ax        = axes[plot_idx]
#     start_bin = plot_idx * bins_per_plot
#     end_bin   = start_bin + bins_per_plot
# 
#     for j, i in enumerate(range(start_bin, end_bin)):
#         subset = trades_total.loc[trades_total['time_bin_idx'] == i, 'Ratio'].dropna()
#         if len(subset) < 10:
#             continue
# 
#         kde    = gaussian_kde(subset, bw_method='scott')
#         mean   = subset.mean()
#         t_low  = time_bins[i]
#         t_high = time_bins[i + 1]
#         color  = palette[j]
# 
#         ax.plot(x_kde, kde(x_kde), color=color, linewidth=1.8,
#                 label=f'[{t_low:.2e}, {t_high:.2e}]')
#         ax.axvline(mean, color=color, linewidth=1.2, linestyle='--')
# 
#     # xlim adattivo sul subplot: [5%, 95%] dei dati del gruppo di bin
#     subplot_data = trades_total.loc[
#         trades_total['time_bin_idx'].between(start_bin, end_bin - 1), 'Ratio'
#     ].dropna()
#     ax_xmin = subplot_data.quantile(0.01)
#     ax_xmax = subplot_data.quantile(0.99)
# 
#     t_start = time_bins[start_bin]
#     t_end   = time_bins[end_bin]
#     ax.set_title(f'$t/T$ ∈ [{t_start:.2e}, {t_end:.2e}]')
#     ax.set_xlabel(r'$I_i / \sqrt{Q}$')
#     ax.set_ylabel('Densità KDE')
#     ax.legend(title=r'$t/T$')
#     ax.set_xlim(ax_xmin, ax_xmax)
#     ax.grid(True, alpha=0.3)
# 
# plt.tight_layout()
# plt.savefig(os.path.join('images', f'{function}_kde.png'), dpi=150, bbox_inches='tight')
# plt.show()

# ---------------------------------------------------------------
# FIGURA 2: ISTOGRAMMI
# ---------------------------------------------------------------
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
axes2 = axes2.flatten()

for plot_idx in range(n_plots):
    ax        = axes2[plot_idx]
    start_bin = plot_idx * bins_per_plot
    end_bin   = start_bin + bins_per_plot

    for j, i in enumerate(range(start_bin, end_bin)):
        subset = trades_total.loc[trades_total['time_bin_idx'] == i, 'Ratio'].dropna()
        if len(subset) < 10:
            continue

        mean   = subset.mean()
        t_low  = time_bins[i]
        t_high = time_bins[i + 1]
        color  = palette[j]

        ax.hist(subset, bins=200, density=True, histtype='step',
                linewidth=1.5, color=color,
                label=f'[{t_low:.2e}, {t_high:.2e}]')
        ax.axvline(mean, color=color, linewidth=1.2, linestyle='--')

    subplot_data = trades_total.loc[
        trades_total['time_bin_idx'].between(start_bin, end_bin - 1), 'Ratio'
    ].dropna()
    ax_xmin = subplot_data.quantile(0.01)
    ax_xmax = subplot_data.quantile(0.99)

    t_start = time_bins[start_bin]
    t_end   = time_bins[end_bin]
    ax.set_title(f'$t/T$ ∈ [{t_start:.2e}, {t_end:.2e}]')
    ax.set_xlabel(r'$I_i / \sqrt{Q}$')
    ax.set_ylabel('Densità')
    ax.legend(title=r'$t/T$')
    ax.set_xlim(ax_xmin, ax_xmax)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join('..\\images', 'impact_time_curve', f'{function}_hist.png'), dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------
# DISTRIBUZIONE DI Ratio PER NbChild — bin fisso
# ---------------------------------------------------------------

TARGET_BIN = 15   # cambia questo valore per esplorare altri bin

nbchild_groups = {
    '2':   lambda x: x == 2,
    '3':   lambda x: x == 3,
    '4':   lambda x: x == 4,
    '5':   lambda x: x == 5,
    '6+':  lambda x: x >= 6,
}
group_colors = [plt.get_cmap("tab10")(i) for i in range(len(nbchild_groups))]

bin_data = trades_total[trades_total['time_bin_idx'] == TARGET_BIN].copy()

print(f"\nBin {TARGET_BIN} - distribuzione NbChild:")
print(bin_data['NbChild'].value_counts().sort_index().head(10).to_string())

t_low_label  = time_bins[TARGET_BIN]
t_high_label = time_bins[TARGET_BIN + 1]

xb_min = bin_data['Ratio'].quantile(0.01)
xb_max = bin_data['Ratio'].quantile(0.99)

fig_nb, ax_hist = plt.subplots(1, 1, figsize=(8, 5))
fig_nb.suptitle(
    f"Distribuzione di " + r"$I_i \sqrt{{Q}}$" + f" per il bin {TARGET_BIN}",
    fontsize=15
)


for (label, condition), color in zip(nbchild_groups.items(), group_colors):
    subset = bin_data.loc[condition(bin_data['NbChild']), 'Ratio'].dropna()
    if len(subset) < 10:
        continue
    mean = subset.mean()
    n    = len(subset)

    ax_hist.hist(subset, bins=100, density=True, histtype='step',
                 linewidth=1.5, color=color,
                 label=f'NbChild={label} (n={n:,})')
    ax_hist.axvline(mean, color=color, linewidth=1.2, linestyle='--')

ax_hist.set_xlim(xb_min, xb_max)
ax_hist.set_xlabel(r'$I_i / \sqrt{Q}$')
ax_hist.set_ylabel('Densità')
ax_hist.set_title('Istogramma')
ax_hist.legend(fontsize=9)
ax_hist.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join('..\\images', 'impact_time_curve', f'{function}_nchild_dist.png'), dpi=150, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------
# PIPELINE ORIGINALE: fit con gli stessi bin custom
# ---------------------------------------------------------------

trades_total['bin'] = pd.cut(
    trades_total['NormalizedTime'],
    bins=time_bins,
    include_lowest=True
)

grouped = trades_total.groupby('bin', observed=True).agg({
    'NormalizedTime': ['mean', 'std'],
    'Ratio':   ['mean', 'std', 'count'],
}).dropna()

grouped.columns = ['Time_mean', 'Time_std', 'Ratio_mean', 'Ratio_std', 'count']

x     = grouped['Time_mean'].to_numpy()
y     = grouped['Ratio_mean'].to_numpy()
x_err = grouped['Time_std'].to_numpy()  / np.sqrt(grouped['count'].to_numpy())
y_err = grouped['Ratio_std'].to_numpy() / np.sqrt(grouped['count'].to_numpy())

# Fit square root
popt, pcov = curve_fit(square_root, x, y, sigma=y_err, absolute_sigma=True)
Y     = popt[0]
Y_err = np.sqrt(pcov[0][0])

for i in range(5):
    err_eff = np.sqrt(y_err**2 + (Y * 0.5 * x**(-0.5) * x_err)**2)
    popt, pcov = curve_fit(square_root, x, y, sigma=err_eff, absolute_sigma=True)
    Y     = popt[0]
    Y_err = np.sqrt(pcov[0][0])

print(f'Fit Temporale: Y = {Y} +- {Y_err}')

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.errorbar(x, y, yerr=y_err, xerr=x_err, linestyle="", marker=".", color="tab:blue", label='Binned Data', zorder=3, capsize=2)
#ax3.plot(0, 0, marker='.', markersize=6, color='tab:blue', zorder=4)

x_plot = np.linspace(x.min() * 0.5, 1.0, 200)
ax3.plot(x_plot, Y * np.sqrt(x_plot), label=r'$y = Y\sqrt{t/T}$',
         linestyle=':', color="black", zorder=2)

# Fit power law
popt_power, pcov_power = curve_fit(power_law, x, y, sigma=y_err, absolute_sigma=True)
A, B = popt_power
ax3.plot(x_plot, power_law(x_plot, A, B), linestyle='--', color="red", zorder=1, label=f'Fit Power Law: A={A:.2f}, B={B:.2f}')

print(f'Fit Power Law: A = {A} +- {np.sqrt(pcov_power[0][0])}, B = {B} +- {np.sqrt(pcov_power[1][1])}')

ax3.set_xlabel(r'$t/T$')
ax3.set_ylabel(r'$I_i / \sqrt{Q}$')
ax3.legend()
ax3.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join('..\\images', 'impact_time_curve', f'{function}_impact_time_curve.png'))
plt.show()