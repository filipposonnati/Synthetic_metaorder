import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir

from scipy.optimize import curve_fit
import scipy.stats as stats

def power_law(x, Y, delta):
    return Y * x**delta

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

dir = 'database\\meta'

print(dir)

image_name = 'impact_volume_dist_curve'

synthetic_meta = pd.read_csv(
    f'{dir}\\{path}', 
    sep=',',  # Usa il tabulatore (o cambia in ',' se il tuo file è un CSV standard)
    parse_dates=['BeginTime', 'EndTime'] # Carica queste colonne come datetime
)

# 1. Preparazione del DataFrame
# Selezioniamo e rinominiamo le colonne per coerenza con la logica fornita.
# Inoltre, filtriamo i volumi non positivi, necessari per la scala logaritmica.
df_res = synthetic_meta[['MetaVolume', 'MetaImpact', 'NbChild']].copy()
df_res = df_res[df_res['NbChild'] > 1]

# 2. Creazione dei Bin Logaritmici
# Determiniamo il range
min = df_res['MetaVolume'].min()
max = df_res['MetaVolume'].max()

bins = np.logspace(np.log10(min), np.log10(max), 51)

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

# 1. Trova il "core" ad altissima densità per il fit preliminare
# Usiamo i bin che hanno almeno il 10% del numero massimo di sample
max_samples = grouped['sample_count'].max()
core_mask = grouped['sample_count'] > (0.5 * max_samples)

x_core = x[core_mask]
y_core = y[core_mask]
x_err_core = x_err[core_mask]
y_err_core = y_err[core_mask]

# Fit preliminare sul core robusto
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

fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

ax1.set_xscale("log")
ax1.set_yscale("log")

x_theoretical = np.linspace(1e-7, 0.1, 2)

ax1.set_xlabel(r'$Q$')
ax1.set_ylabel(r'$I(Q)$')

ax1.grid(True, which="major", ls="-", alpha=0.5)

ax2.set_ylabel('Frequency')

ax2.hist(df_res['MetaVolume'], bins=bins, color='lightgrey', alpha=0.6)

ax1.errorbar(x, y, yerr=y_err, xerr=x_err, marker='o', linestyle="", color='C0', label="Binned data")

ax1.plot(x_theoretical, power_law(x_theoretical, Y, delta), label=r'Fitted curve', linestyle='--', color="black")

ax1.legend()

plt.tight_layout()

plt.savefig(f'images\\{image_name}.png')
#plt.show()