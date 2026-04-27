import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random, datetime
from scipy.stats import powerlaw
from scipy.optimize import curve_fit

import os
from os import listdir
import shutil
from pathlib import Path

# Code to verify the square root impact law

def function_4(x, Y, delta, alpha, mu):
    a, b, N = x
    return Y * a**delta * b**alpha * N**mu

def der_1_function_4(x, Y, delta, alpha, mu):
    a, b, N = x
    return Y * delta * a**(delta - 1) * b**alpha * N**mu

def der_2_function_4(x, Y, delta, alpha, mu):
    a, b, N = x
    return Y * a**delta * alpha * b**(alpha - 1) * N**mu

def function_3(x, Y, delta, mu):
    a, N = x
    return Y * a**delta * N**mu

def der_function_3(x, Y, delta, mu):
    a, N = x
    return Y * delta * a**(delta - 1) * N**mu

plt.rcParams.update({
    'font.size': 12,          # Dimensione base per tutto il testo
    'axes.titlesize': 20,     # Titolo
    'axes.labelsize': 16,     # Etichette assi
    'xtick.labelsize': 12,    # Numeri asse X
    'ytick.labelsize': 12,    # Numeri asse Y
    'legend.fontsize': 14     # Legenda
})

meta_tot = pd.DataFrame()
dir = 'database\\meta'

nb_traders = 20
kind = 'power'
exponent = 2.0

print(nb_traders, kind, exponent)

if kind == 'uniform':
    path = 'meta_' + str(nb_traders) + '_' + kind + '.csv'
else:
    path = 'meta_' + str(nb_traders) + '_' + kind + '_' + str(exponent) + '.csv'

synthetic_meta = pd.read_csv(
    f'{dir}\\{path}', 
    sep=',',  # Usa il tabulatore (o cambia in ',' se il tuo file è un CSV standard)
    parse_dates=['BeginTime', 'EndTime'] # Carica queste colonne come datetime
)

# 1. Preparazione e Filtraggio
df_res = synthetic_meta[['MetaVolume', 'DailyVolume', 'TradedVolume', 'NbChild', 'MetaImpact']].copy()

df_res = df_res[df_res['NbChild'] > 1]

maximum = np.max(df_res['NbChild'].unique())

binned_data_list = []

# Iteriamo su ogni valore unico di NbChild
for nb_child_val in sorted(df_res['NbChild'].unique()):
    subset = df_res[df_res['NbChild'] == nb_child_val].copy()
    
    # Se abbiamo abbastanza dati per questo NbChild, creiamo i bin
    if len(subset) > 50:  # Soglia minima di campioni per NbChild
        min_vol = subset['MetaVolume'].min()
        max_vol = subset['MetaVolume'].max()
        
        bins = np.logspace(np.log10(min_vol), np.log10(max_vol), 31)
        subset['bin'] = pd.cut(subset['MetaVolume'], bins=bins, include_lowest=True)
            
        grouped = subset.groupby('bin', observed=True).agg({
            'MetaVolume': ['mean', 'std'],
            'MetaImpact': ['mean', 'std', 'count'] # Calcoliamo media, deviazione e conteggio
        }).dropna()

        # Appiattiamo le colonne rinominandole per comodità
        grouped.columns = ['MetaVolume_mean', 'MetaVolume_std', 'MetaImpact_mean', 'MetaImpact_std', 'sample_count']

        max_samples = grouped['sample_count'].max()

        grouped = grouped[grouped['sample_count'] > 0.5 * max_samples]
        #grouped = grouped[grouped['sample_count'] > 10]

        # Aggiungiamo NbChild per il fit
        grouped['NbChild'] = nb_child_val
        binned_data_list.append(grouped)

# Consolidamento dati
final_binned_df = pd.concat(binned_data_list)

final_binned_df['y_err'] = final_binned_df['MetaImpact_std'] / np.sqrt(final_binned_df['sample_count'])
final_binned_df['x_err'] = final_binned_df['MetaVolume_std'] / np.sqrt(final_binned_df['sample_count'])

x_fit_3 = (final_binned_df['MetaVolume_mean'].to_numpy(), final_binned_df['NbChild'].to_numpy())
y_fit_3 = final_binned_df['MetaImpact_mean'].to_numpy()

print('Fit con n e 3 params: count > 50, grouping with 31 bins, conditions > 50% max')

popt, pcov = curve_fit(
    function_3, 
    x_fit_3, 
    y_fit_3
)

Y, delta, mu = popt
Y_err, delta_err, mu_err = np.sqrt(np.diag(pcov))

print(f'Fit (no err): Y = {Y} +- {Y_err}, delta = {delta} +- {delta_err}, mu = {mu} +- {mu_err}')

popt, pcov = curve_fit(
    function_3, 
    x_fit_3, 
    y_fit_3, 
    sigma=final_binned_df['y_err'],
    absolute_sigma=True 
)

Y, delta, mu = popt
Y_err, delta_err, mu_err = np.sqrt(np.diag(pcov))

print(f'Fit (y err): Y = {Y} +- {Y_err}, delta = {delta} +- {delta_err}, mu = {mu} +- {mu_err}')

for i in range(10):
    err = np.sqrt(final_binned_df['y_err']**2 + (final_binned_df['x_err'] * der_function_3(x_fit_3, Y, delta, mu))**2)
    popt, pcov = curve_fit(
        function_3, 
        x_fit_3, 
        y_fit_3, 
        sigma=err,
        absolute_sigma=True 
    )

    Y, delta, mu = popt
    Y_err, delta_err, mu_err = np.sqrt(np.diag(pcov))

print(f'Fit (eff err): Y = {Y} +- {Y_err}, delta = {delta} +- {delta_err}, mu = {mu} +- {mu_err}')

plt.figure(figsize=(12, 8))

unique_nb = np.arange(2, maximum)
colors = plt.get_cmap('tab20')

for i, nb_val in enumerate(unique_nb):
    subset_plot = final_binned_df[final_binned_df['NbChild'] == nb_val]
    
    if not subset_plot.empty:
        current_color = colors(i)
        
        plt.errorbar(
            subset_plot['MetaVolume_mean'], 
            subset_plot['MetaImpact_mean'],
            yerr=subset_plot['y_err'],
            xerr=subset_plot['x_err'],
            fmt='o',         
            capsize=0, # Rimuove le "cap" per un look più moderno
            markersize=6,
            label=f'{nb_val}',
            color=current_color,
            ecolor=current_color,
            markeredgecolor='white', # Aggiunge un bordo bianco ai punti per staccarli
        )
        
        # 2. Plot del fit con linea continua più spessa
        x_range = np.logspace(np.log10(subset_plot['MetaVolume_mean'].min()), np.log10(subset_plot['MetaVolume_mean'].max()), 100)
        y_theoretical = function_3((x_range, nb_val), *popt)
        
        plt.plot(x_range, y_theoretical, linestyle='-', color=current_color, lw=2)

# Raffinatezze estetiche
plt.xscale('log')
plt.yscale('log')

# LaTeX per le etichette degli assi
plt.xlabel(r'$Q/V_D$')
plt.ylabel(r'$I$')

# Legenda posizionata fuori o in un angolo pulito
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()

# Salvataggio in alta risoluzione
#plt.savefig('images\\square_root_verify.png', dpi=300, bbox_inches='tight')
plt.show()










# 1. Preparazione e calcolo delle nuove variabili
df_res = synthetic_meta[['MetaVolume', 'DailyVolume', 'TradedVolume', 'NbChild', 'MetaImpact']].copy()

# Definiamo le variabili per function_4
df_res['a'] = df_res['MetaVolume'] * df_res['DailyVolume'] / df_res['TradedVolume']
df_res['b'] = df_res['TradedVolume'] / df_res['DailyVolume']

df_res = df_res[df_res['NbChild'] > 1]

binned_data_list = []

# 2. Raggruppamento per NbChild e Binning 2D
for nb_child_val in sorted(df_res['NbChild'].unique()):
    subset = df_res[df_res['NbChild'] == nb_child_val].copy()
    
    if len(subset) > 20: # Alziamo la soglia perché il binning 2D richiede più dati
        bins_a = np.logspace(np.log10(subset['a'].min()), np.log10(subset['a'].max()), 11)
        bins_b = np.logspace(np.log10(subset['b'].min()), np.log10(subset['b'].max()), 11)
        
        subset['bin_a'] = pd.cut(subset['a'], bins=bins_a, include_lowest=True)
        subset['bin_b'] = pd.cut(subset['b'], bins=bins_b, include_lowest=True)
        
        # Raggruppiamo per le "Aree" definite dalla combinazione dei due bin
        grouped = subset.groupby(['bin_a', 'bin_b'], observed=True).agg({
            'a': ['mean', 'std'],
            'b': ['mean',  'std'],
            'MetaImpact': ['mean', 'std', 'count']
        }).dropna()

        grouped.columns = ['a_mean', 'a_std', 'b_mean', 'b_std', 'MetaImpact_mean', 'MetaImpact_std', 'sample_count']
        
        grouped = grouped[grouped['sample_count'] > np.max(grouped['sample_count']) * 0.5]
        #grouped = grouped[grouped['sample_count'] > 10]
        
        grouped['NbChild'] = nb_child_val
        binned_data_list.append(grouped)

final_binned_df = pd.concat(binned_data_list)
final_binned_df['err'] = final_binned_df['MetaImpact_std'] / np.sqrt(final_binned_df['sample_count'])
final_binned_df['a_err'] = final_binned_df['a_std'] / np.sqrt(final_binned_df['sample_count'])
final_binned_df['b_err'] = final_binned_df['b_std'] / np.sqrt(final_binned_df['sample_count'])

# Input per function_4: (a, b, N)
x_fit_4 = (
    final_binned_df['a_mean'].to_numpy(),
    final_binned_df['b_mean'].to_numpy(),
    final_binned_df['NbChild'].to_numpy()
)
y_fit = final_binned_df['MetaImpact_mean'].to_numpy()

print('\nFit con n e 4 params: count > 20, grouping with 11 bins, conditions > 50% max')

popt, pcov = curve_fit(
    function_4, 
    x_fit_4, 
    y_fit
)

print(f"Fit (no err): {popt} +- {np.sqrt(np.diag(pcov))}")

popt, pcov = curve_fit(
    function_4, 
    x_fit_4, 
    y_fit, 
    sigma=final_binned_df['err'].to_numpy(),
    absolute_sigma=True
)

print(f"Fit (y err): {popt} +- {np.sqrt(np.diag(pcov))}")

for i in range(10):
    err = np.sqrt((final_binned_df['err'].to_numpy())**2 + (final_binned_df['a_err'].to_numpy() * der_1_function_4(x_fit_4, *popt))**2 + (final_binned_df['b_err'].to_numpy() * der_2_function_4(x_fit_4, *popt))**2)
    popt, pcov = curve_fit(
        function_4, 
        x_fit_4, 
        y_fit, 
        sigma=err,
        absolute_sigma=True
    )

print(f"Fit (eff err): {popt} +- {np.sqrt(np.diag(pcov))}")