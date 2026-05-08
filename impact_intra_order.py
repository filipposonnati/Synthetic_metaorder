import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from scipy.optimize import curve_fit

plt.rcParams.update({
    'font.size': 12,          # Dimensione base per tutto il testo
    'axes.titlesize': 20,     # Titolo
    'axes.labelsize': 16,     # Etichette assi
    'xtick.labelsize': 12,    # Numeri asse X
    'ytick.labelsize': 12,    # Numeri asse Y
    'legend.fontsize': 14     # Legenda
})

def power_law(x, Y, delta):
    return Y * x**delta

def square_root(x, Y):
    return Y * np.sqrt(x)

def constant(x, Y):
    return Y

function = '20_power_2.0'

model = 'var_1000'
#model = ''

print(function + ' ' + model)

dir = f'database\\{f"trades_{model}_" if model else "trades_"}' + function
paths = np.array(listdir(dir))

print(dir)

trades_total = None

for path in paths:
    trades = pd.read_csv(
        dir + f'\\{path}', 
        sep=','
    ) 

    daily_trades = trades[['PartialVolume', 'NbChild', 'PartialImpact', 'MetaVolume']].copy()
    #daily_trades.drop(columns=['NbChild'], inplace=True)

    trades_total = pd.concat([trades_total, daily_trades])

trades_total = trades_total[trades_total['NbChild'] > 1]

print('Cumulative fit')

# 1. Definizione dei bin (uso geomspace per chiarezza, o mantieni il tuo logspace)
bins = np.logspace(np.log10(trades_total['PartialVolume'].min()), 
                   np.log10(trades_total['PartialVolume'].max()), 51)

# 2. Assegnazione e Raggruppamento per la curva d'impatto
trades_total['bin'] = pd.cut(trades_total['PartialVolume'], bins=bins, include_lowest=True)
grouped = trades_total.groupby('bin', observed=True).agg({
    'PartialVolume': ['mean', 'std', 'count'],
    'PartialImpact': ['mean', 'std']
}).dropna()

grouped.columns = ['PartialVolume_mean', 'PartialVolume_std', 'count', 'PartialImpact_mean', 'PartialImpact_std']

max_samples = grouped['count'].max()
core_mask = grouped['count'] > (0.5 * max_samples)

x = grouped['PartialVolume_mean'].to_numpy()
y = grouped['PartialImpact_mean'].to_numpy()
x_err = grouped['PartialVolume_std'].to_numpy() / np.sqrt(grouped['count'].to_numpy())
y_err = grouped['PartialImpact_std'].to_numpy() / np.sqrt(grouped['count'].to_numpy())

x_core = x[core_mask]
y_core = y[core_mask]
x_err_core = x_err[core_mask]
y_err_core = y_err[core_mask]

popt, pcov = curve_fit(power_law, x_core, y_core)

Y = popt[0]
delta = popt[1]
Y_err = np.sqrt(pcov[0][0])
delta_err = np.sqrt(pcov[1][1])

print(f'Fit (no err): Y = {Y} +- {Y_err}, delta = {delta} +- {delta_err}')

popt, pcov = curve_fit(power_law, x_core, y_core, sigma=y_err_core, absolute_sigma=True)

Y = popt[0]
delta = popt[1]
Y_err = np.sqrt(pcov[0][0])
delta_err = np.sqrt(pcov[1][1])

print(f'Fit (y err): Y = {Y} +- {Y_err}, delta = {delta} +- {delta_err}')

for i in range(10):
    err_core = np.sqrt(y_err_core**2 + (Y * delta * x_core**(delta - 1) * x_err_core)**2)
    popt, pcov = curve_fit(power_law, x_core, y_core, sigma=err_core, absolute_sigma=True)

    Y = popt[0]
    delta = popt[1]
    Y_err = np.sqrt(pcov[0][0])
    delta_err = np.sqrt(pcov[1][1])

print(f'Fit (eff err): Y = {Y} +- {Y_err}, delta = {delta} +- {delta_err}')
Y_cum, delta_cum, Y_cum_err, delta_cum_err = Y, delta, Y_err, delta_err  # save for LaTeX table

# 3. Creazione del Plot
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(x, y, linestyle="", marker="o", label=f'Binned data', zorder=3)

x_theoretical = np.logspace(np.log10(1e-7), np.log10(2e-3), 100)
ax1.plot(x_theoretical, Y * x_theoretical**delta, label=r'$y = Yx^{\delta}$', linestyle=':', color="black", zorder=2)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r'$\Sigma q_i$')
ax1.set_ylabel(r'$I_i$')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.hist(trades_total['PartialVolume'], bins=bins, alpha=0.2, color='gray', zorder=1)
ax2.set_ylabel('Frequency')
ax2.tick_params(axis='y')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

ax1.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()

plt.savefig(f'images\\impact_intra_order\\{model + "_" if model else ""}{function}_cumulate.png')










print('Square root fit')

trades_total['Ratio'] = trades_total['PartialImpact'] / np.sqrt(trades_total['MetaVolume'])
trades_total['NormalizedVolume'] = trades_total['PartialVolume'] / trades_total['MetaVolume']

bins = np.linspace(0, 1, 21)**3

# 3. Assegnazione dei metaordini ai Bin
# 'include_lowest=True' assicura che il valore minimo sia incluso.
trades_total['bin'] = pd.cut(trades_total['NormalizedVolume'], bins=bins, include_lowest=True)

# 2. Assegnazione e Raggruppamento per la curva d'impatto
trades_total['bin'] = pd.cut(trades_total['NormalizedVolume'], bins=bins, include_lowest=True)
grouped = trades_total.groupby('bin', observed=True).agg({
    'NormalizedVolume': ['mean', 'std'],
    'Ratio': ['mean', 'std', 'count'],
}).dropna()

grouped.columns = ['NormalizedVolume_mean', 'NormalizedVolume_std', 'Ratio_mean', 'Ratio_std', 'count']

x = grouped['NormalizedVolume_mean'].to_numpy()
y = grouped['Ratio_mean'].to_numpy()
#y = grouped['PartialImpact'].to_numpy() / np.sqrt(grouped['MetaVolume'].to_numpy()) # Ratio computed on the means

x_err = grouped['NormalizedVolume_std'].to_numpy() / np.sqrt(grouped['count'].to_numpy())
y_err = grouped['Ratio_std'].to_numpy() / np.sqrt(grouped['count'].to_numpy())

popt, pcov = curve_fit(square_root, x, y)

Y = popt[0]
Y_err = np.sqrt(pcov[0][0])

print(f'Fit (no err): Y = {Y} +- {Y_err}')

popt, pcov = curve_fit(square_root, x, y, sigma=y_err, absolute_sigma=True)

Y = popt[0]
Y_err = np.sqrt(pcov[0][0])

print(f'Fit (y err): Y = {Y} +- {Y_err}')

for i in range(10):
    err = np.sqrt(y_err**2 + (Y * 0.5 * x**(-0.5) * x_err)**2)
    popt, pcov = curve_fit(square_root, x, y, sigma=err, absolute_sigma=True)

    Y = popt[0]
    Y_err = np.sqrt(pcov[0][0])

print(f'Fit (eff err): Y = {Y} +- {Y_err}')
Y_sqrt, Y_sqrt_err = Y, Y_err  # save for LaTeX table

# 3. Creazione del Plot
fig = plt.subplots(figsize=(8, 6))

plt.plot(x, y, linestyle="", marker="o", label=f'Binned data', zorder=3)

x_theoretical = np.linspace(0.0, 1.0, 50)
plt.plot(x_theoretical, Y * np.sqrt(x_theoretical), label=r'$y = Y\sqrt{x}$', linestyle=':', color="black", zorder=2)

plt.xlabel(r'$\Sigma q_i / Q$')
plt.ylabel(r'$I_i / \sqrt{Q}$')
plt.tick_params(axis='y')
plt.legend()

plt.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()

plt.savefig(f'images\\impact_intra_order\\{model + "_" if model else ""}{function}_ratio.png')










print('Flat fit')

trades_total['Ratio'] = trades_total['PartialImpact'] / np.sqrt(trades_total['PartialVolume'])
trades_total['NormalizedVolume'] = trades_total['PartialVolume'] / trades_total['MetaVolume']

bins = np.linspace(0, 1, 26)

# 3. Assegnazione dei metaordini ai Bin
# 'include_lowest=True' assicura che il valore minimo sia incluso.
trades_total['bin'] = pd.cut(trades_total['NormalizedVolume'], bins=bins, include_lowest=True)

# 2. Assegnazione e Raggruppamento per la curva d'impatto
trades_total['bin'] = pd.cut(trades_total['NormalizedVolume'], bins=bins, include_lowest=True)
grouped = trades_total.groupby('bin', observed=True).agg({
    'NormalizedVolume': ['mean', 'std'],
    'Ratio': ['mean', 'std', 'count'],
}).dropna()

grouped.columns = ['NormalizedVolume_mean', 'NormalizedVolume_std', 'Ratio_mean', 'Ratio_std', 'count']

x = grouped['NormalizedVolume_mean'].to_numpy()
y = grouped['Ratio_mean'].to_numpy()

x_err = grouped['NormalizedVolume_std'].to_numpy() / np.sqrt(grouped['count'].to_numpy())
y_err = grouped['Ratio_std'].to_numpy() / np.sqrt(grouped['count'].to_numpy())

popt, pcov = curve_fit(constant, x, y)

Y = popt[0]
Y_err = np.sqrt(pcov[0][0])

print(f'Fit (no err): Y = {Y} +- {Y_err}')

popt, pcov = curve_fit(constant, x, y, sigma = y_err, absolute_sigma=True)

Y = popt[0]
Y_err = np.sqrt(pcov[0][0])

print(f'Fit (y err): Y = {Y} +- {Y_err}')
Y_flat, Y_flat_err = Y, Y_err  # save for LaTeX table

# 3. Creazione del Plot
fig = plt.subplots(figsize=(8, 6))

plt.plot(x, y, linestyle="", marker="o", label=f'Binned data', zorder=3)

x_theoretical = np.linspace(0.0, 1.0, 100)
plt.plot(x_theoretical, np.full(len(x_theoretical), Y), label=r'$y = Y$', linestyle=':', color="black", zorder=2)

plt.xlabel(r'$\Sigma q_i / Q$')
plt.ylabel(r'$I_i / \sqrt{\Sigma q_i}$')
plt.tick_params(axis='y')
plt.legend()

plt.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()

plt.savefig(f'images\\impact_intra_order\\{model + "_" if model else ""}{function}_flat.png')





print('NbChild Analysis')

child_range = range(2, 12)
fig, ax = plt.subplots(figsize=(11, 7))

# Tab10 gives 10 maximally distinct colors
colors = plt.cm.tab10(np.linspace(0, 1, 10))

# Cycle through linestyles to add a second visual channel
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
linewidths  = [1.5] * 10  # optionally scale: [1.2 + 0.15*i for i in range(10)]

for idx, n_child in enumerate(child_range):
    subset = trades_total[trades_total['NbChild'] == n_child].copy()

    if len(subset) < 30:
        print(f"Skipping NbChild={n_child}: insufficient data ({len(subset)} samples)")
        continue

    subset['Ratio'] = subset['PartialImpact'] / np.sqrt(subset['MetaVolume'])
    subset['NormalizedVolume'] = subset['PartialVolume'] / subset['MetaVolume']

    bins_x = np.linspace(0, 1, 31)**2
    subset['bin'] = pd.cut(subset['NormalizedVolume'], bins=bins_x, include_lowest=True)

    grp = subset.groupby('bin', observed=True).agg({
        'NormalizedVolume': 'mean',
        'Ratio': ['mean', 'std', 'count']
    }).dropna()

    grp.columns = ['x', 'y', 'y_std', 'count']
    x_val = grp['x'].to_numpy()
    y_val = grp['y'].to_numpy()

    line, = ax.plot(
        x_val, y_val,
        color=colors[idx],
        linestyle=linestyles[idx],
        linewidth=linewidths[idx],
        alpha=0.85,
        label=f'n={n_child}'
    )

    # Direct end-of-line label (avoids legend hunting)
    """
    ax.annotate(
        f'n={n_child}',
        xy=(x_val[-1], y_val[-1]),
        xytext=(4, 0),
        textcoords='offset points',
        color=colors[idx],
        fontsize=9,
        va='center'
    )
    """

ax.set_xlabel(r'$\Sigma q_i / Q$')
ax.set_ylabel(r'$I_i / \sqrt{Q}$')
# Keep legend as secondary reference, smaller and compact
ax.legend(title="n children", fontsize=8, title_fontsize=9,
          bbox_to_anchor=(1, 1), loc='upper left')
ax.grid(True, ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig(f'images\\impact_intra_order\\{model + "_" if model else ""}{function}_ratio_child.png', dpi=150, bbox_inches='tight')