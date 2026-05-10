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

model = "noise"

dir = 'database\\meta'
if model != "":
    dir = dir + "_" + model

paths = np.array(listdir(dir))

fig = plt.figure(figsize=(8, 6))

image_name = 'impact_volume_curve'
if model != "":
    image_name = image_name + "_" + model

# Pattern Breakdown:
# meta_             : Matches literal prefix
# (                 : Start alternation
#   (?P<num_one>1)  : Case A: The number is exactly 1 (and nothing follows)
#   |               : OR
#   (?P<num_others>\d+)_(?P<kind>\w+?)(?:_(?P<exp>[\d.]+))? : Case B: Other numbers + kind + opt. exp
# )                 : End alternation
# \.csv             : Matches file extension
pattern = r"meta_(?:(?P<num_one>1)|(?P<num_others>\d+)_(?P<kind>\w+?)(?:_(?P<exp>[\d.]+))?)\.csv"

for path in paths:
    match = re.search(pattern, path)
    num_traders = match.group('num_one') or match.group('num_others')
    kind = match.group('kind') if match.group('kind') else ""
    exp = match.group('exp') if match.group('exp') else ""

    parts = [str(num_traders), kind, exp]
    label = " ".join(filter(None, parts))

    synthetic_meta = pd.read_csv(
        f'{dir}\\{path}', 
        sep=',',  # Usa il tabulatore (o cambia in ',' se il tuo file è un CSV standard)
        parse_dates=['BeginTime', 'EndTime'] # Carica queste colonne come datetime
    )

    # 1. Preparazione del DataFrame
    synthetic_meta['NbChild'] = pd.to_numeric(synthetic_meta['NbChild'], errors='coerce')

    df_res = synthetic_meta[['MetaVolume', 'MetaImpact', 'NbChild']].copy()
    df_res = df_res[df_res['NbChild'] > 1]

    df_res.drop(columns=['NbChild'], inplace=True)

    # 2. Creazione dei Bin Logaritmici
    # Determiniamo il range
    min_vol = df_res['MetaVolume'].min()
    max_vol = df_res['MetaVolume'].max()

    # Creiamo 51 punti (per 50 bin) equidistanti nello spazio logaritmico
    bins = np.logspace(np.log10(min_vol), np.log10(max_vol), 101)

    # 3. Assegnazione dei metaordini ai Bin
    # 'include_lowest=True' assicura che il valore minimo sia incluso.
    df_res['bin'] = pd.cut(df_res['MetaVolume'], bins=bins, include_lowest=True)

    # 4. Raggruppamento e Calcolo delle Medie
    # 'observed=True' è consigliato per le versioni recenti di Pandas quando si raggruppa con pd.cut
    grouped = df_res.groupby('bin', observed=True).agg({
        'MetaVolume': ['mean', 'std'], # Volume medio per rappresentare il centro del bin
        'MetaImpact': ['mean', 'std', 'count']  # Impatto medio sul prezzo
    }).dropna() # Rimuove i bin che non contengono dati

    grouped.columns = ['MetaVolume_mean', 'MetaVolume_std', 'MetaImpact_mean', 'MetaImpact_std', 'sample_count']

    # 5. Estrazione degli Array di Risultato
    x = grouped['MetaVolume_mean'].to_numpy()
    y = grouped['MetaImpact_mean'].to_numpy()

    x_err = grouped['MetaVolume_std'].to_numpy() / np.sqrt(grouped['sample_count'].to_numpy())
    y_err = grouped['MetaImpact_std'].to_numpy() / np.sqrt(grouped['sample_count'].to_numpy())

    #print(len(x))
    #print(len(y))

    #print(f'path: {path}')

    plt.plot(x, y, linestyle="", marker=".", label = f"{label}")

x_theoretical = np.linspace(1e-6, 1e-3, 2)
#plt.plot(x_theoretical, np.sqrt(x_theoretical), label=r'$y = \sqrt{x}$', linestyle=':', color = "black")
#plt.plot(x_theoretical, x_theoretical, label=r'$y = x$', linestyle='-.', color = "black")

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r'$Q$')
plt.ylabel(r'$I(Q)$')
plt.legend()
plt.grid(True, which="both", ls="-")

plt.savefig(f'images\\{image_name}.png')
plt.show()