import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir

from scipy.optimize import curve_fit
import scipy.stats as stats

def function_3_ab(x, Y, delta, alpha):
    a, b = x
    return Y * a**delta * b**alpha

def der_1_function_3_ab(x, Y, delta, alpha):
    a, b = x
    return Y * delta * a**(delta - 1) * b**alpha

def der_2_function_3_ab(x, Y, delta, alpha):
    a, b = x
    return Y * a**delta * alpha * b**(alpha - 1)

path = 'meta_20_power_2.0.csv'

print(path)

dir = 'database\\meta'

print(dir)

synthetic_meta = pd.read_csv(
    f'{dir}\\{path}', 
    sep=',',  # Usa il tabulatore (o cambia in ',' se il tuo file è un CSV standard)
    parse_dates=['BeginTime', 'EndTime'] # Carica queste colonne come datetime
)

# 1. Preparazione e calcolo delle variabili
df_res = synthetic_meta[['MetaVolume', 'DailyVolume', 'TradedVolume', 'MetaImpact']].copy()

# a = Q / participation_rate 
# b = participation rate / daily_volume
df_res['a'] = df_res['MetaVolume'] * df_res['DailyVolume'] / df_res['TradedVolume']
df_res['b'] = df_res['TradedVolume'] / df_res['DailyVolume']

# 2. Binning 2D globale (senza loop su NbChild)
# Usiamo 11 bin logaritmici come nel codice originale
bins_a = np.logspace(np.log10(df_res['a'].min()), np.log10(df_res['a'].max()), 51)
bins_b = np.logspace(np.log10(df_res['b'].min()), np.log10(df_res['b'].max()), 51)

df_res['bin_a'] = pd.cut(df_res['a'], bins=bins_a, include_lowest=True)
df_res['bin_b'] = pd.cut(df_res['b'], bins=bins_b, include_lowest=True)

# Raggruppiamo per le "Aree" definite dalla combinazione dei due bin
grouped = df_res.groupby(['bin_a', 'bin_b'], observed=True).agg({
    'a': ['mean', 'std'],
    'b': ['mean', 'std'],
    'MetaImpact': ['mean', 'std', 'count']
}).dropna()

grouped.columns = ['a_mean', 'a_std', 'b_mean', 'b_std', 'MetaImpact_mean', 'MetaImpact_std', 'sample_count']

# Filtro per robustezza statistica
# Nota: filtrando per > 0.5 * max su tutto il dataset potresti scartare molti bin se i dati sono concentrati.
# Se il fit fallisce, valuta di abbassare questa soglia (es. 0.1 o un numero fisso come > 20).
#grouped = grouped[grouped['sample_count'] > 1]

#print(grouped['sample_count'])

# Calcolo degli errori
grouped['err'] = grouped['MetaImpact_std'] / np.sqrt(grouped['sample_count'])
grouped['a_err'] = grouped['a_std'] / np.sqrt(grouped['sample_count'])
grouped['b_err'] = grouped['b_std'] / np.sqrt(grouped['sample_count'])

# 3. Preparazione dei dati per il fit
# Input per function_3_ab: (a, b)
x_fit_ab = (
    grouped['a_mean'].to_numpy(),
    grouped['b_mean'].to_numpy()
)
y_fit = grouped['MetaImpact_mean'].to_numpy()

print('\nFit con 3 params: grouping with 50x50 bins')

# --- FIT 2: Errore solo su Y ---
#popt, pcov = curve_fit(function_3_ab, x_fit_ab, y_fit, sigma=grouped['err'].to_numpy(), absolute_sigma=True)
#print(f"Fit (y err): {popt} +- {np.sqrt(np.diag(pcov))}")

popt = [1.0, 0.5, 0.5]

# --- FIT 3: Errore effettivo (Iterativo) ---
for i in range(10):
    # Propagazione dell'errore: err_tot^2 = err_y^2 + (err_a * dI/da)^2 + (err_b * dI/db)^2
    err_eff = np.sqrt(
        (grouped['err'].to_numpy())**2 + 
        (grouped['a_err'].to_numpy() * der_1_function_3_ab(x_fit_ab, *popt))**2 + 
        (grouped['b_err'].to_numpy() * der_2_function_3_ab(x_fit_ab, *popt))**2
    )
    
    popt, pcov = curve_fit(
        function_3_ab, 
        x_fit_ab, 
        y_fit, 
        sigma=err_eff,
        absolute_sigma=True
    )

print(f"Fit (eff err): {popt} +- {np.sqrt(np.diag(pcov))}")

fig = plt.figure(figsize=(12, 10))
# Creiamo un asse 3D
ax = fig.add_subplot(111, projection='3d')

# Estraiamo i dati effettivi
a_data = grouped['a_mean'].to_numpy()
b_data = grouped['b_mean'].to_numpy()
impact_data = grouped['MetaImpact_mean'].to_numpy()

# Plottiamo i dati binnati come scatter plot (usiamo log10 per stabilità visiva)
ax.scatter(np.log10(a_data), np.log10(b_data), np.log10(impact_data), 
           color='red', marker='.', s=40, label='Binned Data', depthshade=True, zorder=5)

# Creiamo una griglia densa per a e b per tracciare la superficie teorica
# Usiamo i minimi e massimi dei dati per definire i limiti della griglia
a_grid = np.logspace(np.log10(a_data.min()), np.log10(a_data.max()), 2)
b_grid = np.logspace(np.log10(b_data.min()), np.log10(b_data.max()), 2)
A, B = np.meshgrid(a_grid, b_grid)

# Calcoliamo la Z (Impact) teorica usando i parametri fittati nell'ultimo step (popt)
Z_theoretical = function_3_ab((A, B), *popt)

# Plottiamo la superficie (anche qui con log10)
surf = ax.plot_surface(np.log10(A), np.log10(B), np.log10(Z_theoretical), alpha=0.5, edgecolor='none', zorder=1)

# Estetica degli assi
ax.set_xlabel(r'$\log_{10}(\frac{Q}{V_P})$')
ax.set_ylabel(r'$\log_{10}(\frac{V_P}{V_D})$')
ax.set_zlabel(r'$\log_{10}(I)$')

# Angolazione iniziale della camera (elevazione, azimut) - puoi modificarli per ruotare il grafico
ax.view_init(elev=30, azim=-150)

plt.legend(loc='upper left')
plt.tight_layout()

# Salva l'immagine
plt.savefig('images\\impact_law_participation_rate.png', dpi=300, bbox_inches='tight')
#plt.show()