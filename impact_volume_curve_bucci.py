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

# Two stacked subplots sharing the X axis: top = points, bottom = distributions
fig, (ax_main, ax_hist) = plt.subplots(
    2, 1,
    figsize=(8, 10),
    sharex=True,
    gridspec_kw={'hspace': 0.08}   # tight vertical gap; x-tick labels only on bottom
)

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

    # 3. Histogram on the bottom subplot
    hist_bins = np.logspace(np.log10(pr.min()), np.log10(pr.max()), 51)
    counts, edges = np.histogram(pr, bins=hist_bins, density=False)
    ax_hist.step(edges[:-1], counts, where='post', color=color, alpha=1.0, linewidth=1.5, label=label)

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

    # Top subplot: binned mean points
    ax_main.plot(x, y, linestyle="", marker=".", color=color, label=label)

# --- Top subplot (points) formatting ---
ax_main.set_xscale("log")
ax_main.set_yscale("log")
ax_main.set_ylabel(r'$I(Q) / \sqrt{Q}$')
ax_main.grid(True, which="both", ls="-")
# Single legend placed to the right of the top subplot
ax_main.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
# Hide x-tick labels on top plot (shared axis, labels shown on bottom)
plt.setp(ax_main.get_xticklabels(), visible=False)

# --- Bottom subplot (distributions) formatting ---
ax_hist.set_xscale("log")
ax_hist.set_xlabel(r'$\phi = Q / Q_P$')
ax_hist.set_ylabel('Count')
ax_hist.grid(True, which="both", ls="-")
# Extend x-axis to 1 so the full participation rate range is always visible.
# Note: the histogram naturally stops where your data stops (pr.max() per file),
# which is typically well below 1. This only affects the axis limit, not the data.
current_xlim = ax_hist.get_xlim()
ax_hist.set_xlim(right=max(current_xlim[1], 1.0))

plt.tight_layout()
plt.savefig(f'images\\part_rate\\{image_name}.png', bbox_inches='tight', dpi=300)
plt.show()