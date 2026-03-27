import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir

plt.rcParams.update({
    'font.size': 12,          # Dimensione base per tutto il testo
    'axes.titlesize': 20,     # Titolo
    'axes.labelsize': 16,     # Etichette assi
    'xtick.labelsize': 12,    # Numeri asse X
    'ytick.labelsize': 12,    # Numeri asse Y
    'legend.fontsize': 14     # Legenda
})

path = 'meta_20_power_2.0.csv'

print(path)

dir = '..\\database\\meta'

print(dir)

image_name = 'impact_time_dist'

synthetic_meta = pd.read_csv(
    f'{dir}\\{path}', 
    sep=',',  # Usa il tabulatore (o cambia in ',' se il tuo file è un CSV standard)
    parse_dates=['BeginTime', 'EndTime'] # Carica queste colonne come datetime
)

# 1. Preparazione del DataFrame
# Selezioniamo e rinominiamo le colonne per coerenza con la logica fornita.
# Inoltre, filtriamo i volumi non positivi, necessari per la scala logaritmica.
df_res = pd.DataFrame({
    'MetaDuration': (synthetic_meta['EndTime'] - synthetic_meta['BeginTime']).dt.total_seconds(),
    'MetaImpact': synthetic_meta['MetaImpact'],
    'MetaVolume': synthetic_meta['MetaVolume'],
    'NbChild': synthetic_meta['NbChild'],
    'Ratio': synthetic_meta['MetaImpact'] / np.sqrt(synthetic_meta['MetaVolume'])
})

df_res = df_res[df_res['MetaDuration'] > 0]
df_res = df_res[df_res['NbChild'] > 1]

# 2. Creazione dei Bin Logaritmici
# Determiniamo il range
min = df_res['MetaDuration'].min()
max = df_res['MetaDuration'].max()

bins = np.logspace(np.log10(min), np.log10(max), 51)

# 3. Assegnazione dei metaordini ai Bin
# 'include_lowest=True' assicura che il valore minimo sia incluso.
df_res['bin'] = pd.cut(df_res['MetaDuration'], bins=bins, include_lowest=True)

# 4. Raggruppamento e Calcolo delle Medie
# 'observed=True' è consigliato per le versioni recenti di Pandas quando si raggruppa con pd.cut
grouped = df_res.groupby('bin', observed=True).agg({
    'MetaDuration': 'mean',
    'MetaImpact': 'mean',
    'MetaVolume': 'mean',
    'Ratio': 'mean'
}).dropna()

# 5. Estrazione degli Array di Risultato
x = grouped['MetaDuration'].to_numpy()
y = grouped['Ratio'].to_numpy()

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

ax1.set_xscale("log")
ax1.set_xlabel('Duration [s]')
ax1.set_ylabel(r'$I / \sqrt{Q}$')
ax1.set_ylim(0.0, 2.0)
ax1.set_xlim(1e-6, 1e4)
ax1.grid(True, which="major", ls="-", alpha=0.5)

ax2.set_ylabel('Frequency')

ax2.hist(df_res['MetaDuration'], bins=bins, color='lightgrey', alpha=0.6, label='Density', zorder=1)

if(path[8:13] == "power"):
    ax1.scatter(x, y, linestyle="", marker="o", label = f"{path[5:17]}")
elif not(path[6:7] == "_"):
    ax1.scatter(x, y, linestyle="", marker="o", label = f"{path[5:15]}")
else:
    ax1.scatter(x, y, linestyle="", marker="o", label = f"{path[5:14]}")

ax1.legend()

plt.tight_layout()

plt.savefig(f'..\\images\\{image_name}.png')
plt.show()