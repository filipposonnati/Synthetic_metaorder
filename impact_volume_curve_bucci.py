import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from scipy.optimize import curve_fit
import re

def power_law(x, a, delta):
    return a * x**delta

plt.rcParams.update({
    'font.size': 12,          # Dimensione base per tutto il testo
    'axes.titlesize': 20,     # Titolo
    'axes.labelsize': 16,     # Etichette assi
    'xtick.labelsize': 12,    # Numeri asse X
    'ytick.labelsize': 12,    # Numeri asse Y
    'legend.fontsize': 14     # Legenda
})

model = ""

dir = 'database\\meta'
if model != "":
    dir = dir + "_" + model

paths = np.array(listdir(dir))

fig, ax_main = plt.subplots(figsize=(8, 6))
ax_hist = ax_main.twinx()  # Right axis for histogram

image_name = 'impact_ratio_participation_curve'
if model != "":
    image_name = image_name + "_" + model

pattern = r"meta_(?:(?P<num_one>1)|(?P<num_others>\d+)_(?P<kind>\w+?)(?:_(?P<exp>[\d.]+))?)\.csv"

# Use the default matplotlib color cycle
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, path in enumerate(paths):
    if path == 'meta_1.csv':
        continue
    match = re.search(pattern, path)
    num_traders = match.group('num_one') or match.group('num_others')
    kind = match.group('kind') if match.group('kind') else ""
    exp = match.group('exp') if match.group('exp') else ""

    parts = [str(num_traders), kind, exp]
    label = " ".join(filter(None, parts))

    color = color_cycle[i % len(color_cycle)]

    synthetic_meta = pd.read_csv(
        f'{dir}\\{path}',
        sep=',',
        parse_dates=['BeginTime', 'EndTime']
    )

    # 1. Preparazione del DataFrame
    synthetic_meta['NbChild'] = pd.to_numeric(synthetic_meta['NbChild'], errors='coerce')

    df_res = synthetic_meta[['MetaVolume', 'MetaImpact', 'TradedVolume', 'NbChild']].copy()
    df_res = df_res[df_res['NbChild'] > 1]

    df_res.drop(columns=['NbChild'], inplace=True)

    # 2. Calcolo delle nuove variabili
    # Ratio: impact / sqrt(MetaVolume)
    df_res['Ratio'] = df_res['MetaImpact'] / np.sqrt(df_res['MetaVolume'])

    # Participation rate: MetaVolume / TradedVolume (volume traded during execution)
    df_res['ParticipationRate'] = df_res['MetaVolume'] / df_res['TradedVolume']

    # Drop rows with invalid values (zero TradedVolume, NaN, inf)
    df_res = df_res.replace([np.inf, -np.inf], np.nan).dropna(subset=['Ratio', 'ParticipationRate'])
    df_res = df_res[df_res['ParticipationRate'] > 0]

    pr = df_res['ParticipationRate'].to_numpy()

    # 3. Histogram for this path (right axis), same color as scatter, no label (already in legend)
    hist_bins = np.logspace(np.log10(pr.min()), np.log10(pr.max()), 51)
    counts, edges = np.histogram(pr, bins=hist_bins, density=False)
    widths = np.diff(edges)
    ax_hist.step(edges[:-1], counts, where='post', color=color, alpha=1.0, linewidth=1.5)

    # 4. Creazione dei Bin Logaritmici sull'asse X (participation rate)
    bins = np.logspace(np.log10(pr.min()), np.log10(pr.max()), 101)

    # 5. Assegnazione ai Bin
    df_res['bin'] = pd.cut(df_res['ParticipationRate'], bins=bins, include_lowest=True)

    # 6. Raggruppamento e Calcolo delle Medie
    grouped = df_res.groupby('bin', observed=True).agg({
        'ParticipationRate': ['mean', 'std'],
        'Ratio': ['mean', 'std', 'count']
    }).dropna()

    grouped.columns = ['PR_mean', 'PR_std', 'Ratio_mean', 'Ratio_std', 'sample_count']

    # 7. Estrazione degli Array di Risultato
    x = grouped['PR_mean'].to_numpy()
    y = grouped['Ratio_mean'].to_numpy()

    ax_main.plot(x, y, linestyle="", marker=".", color=color, label=f"{label}")

ax_hist.set_xscale("log")
ax_hist.set_ylabel('Density', fontsize=16)
ax_hist.tick_params(axis='y', labelsize=12)

# Keep histogram bars behind the main plot lines
ax_hist.set_zorder(ax_main.get_zorder() - 1)
ax_main.set_facecolor('none')  # Make main axis background transparent

# --- Main axis formatting ---
ax_main.set_xscale("log")
ax_main.set_yscale("log")
ax_main.set_xlabel(r'$\phi = Q / Q_P$', fontsize=16)
ax_main.set_ylabel(r'$I(Q) / \sqrt{Q}$', fontsize=16)
ax_main.grid(True, which="both", ls="-")
ax_main.legend(loc='lower left')

plt.tight_layout()
plt.savefig(f'images\\part_rate\\{image_name}.png')
plt.show()