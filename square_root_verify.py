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

def function_3(x, Y, delta, mu):
    a, N = x
    return Y * a**delta * N**mu

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

maximum = 15

# Filtro iniziale (manteniamo quello del tuo script originale)
mask = (df_res['MetaVolume'] > 2e-5) & (df_res['MetaVolume'] < 1e-3)
df_res = df_res[mask & (df_res['NbChild'] > 1) & (df_res['NbChild'] < maximum)]

binned_data_list = []

# Iteriamo su ogni valore unico di NbChild
for nb_child_val in sorted(df_res['NbChild'].unique()):
    subset = df_res[df_res['NbChild'] == nb_child_val].copy()
    
    # Se abbiamo abbastanza dati per questo NbChild, creiamo i bin
    if len(subset) > 20:  # Soglia minima di campioni per NbChild
        min_vol = subset['MetaVolume'].min()
        max_vol = subset['MetaVolume'].max()
        
        # Evitiamo errori se min == max
        if min_vol < max_vol:
            bins = np.logspace(np.log10(min_vol), np.log10(max_vol), 21)
            subset['bin'] = pd.cut(subset['MetaVolume'], bins=bins, include_lowest=True)
            
        grouped = subset.groupby('bin', observed=True).agg({
            'MetaVolume': 'mean',
            'MetaImpact': ['mean', 'std', 'count'] # Calcoliamo media, deviazione e conteggio
        }).dropna()

        # Appiattiamo le colonne rinominandole per comodità
        grouped.columns = ['MetaVolume', 'MetaImpact_mean', 'MetaImpact_std', 'sample_count']

        # SCARTO: Teniamo solo i bin che hanno più di 1 punto
        grouped = grouped[grouped['sample_count'] > 1]

        # Aggiungiamo NbChild per il fit
        grouped['NbChild'] = nb_child_val
        binned_data_list.append(grouped)

# 3. Consolidamento dei dati binnati
if not binned_data_list:
    print(f"Dati insufficienti per il fit in {path}")
    exit()
    
# Consolidamento dati
final_binned_df = pd.concat(binned_data_list)

final_binned_df['sem'] = final_binned_df['MetaImpact_std'] / np.sqrt(final_binned_df['sample_count'])

x_fit_3 = (final_binned_df['MetaVolume'].to_numpy(), final_binned_df['NbChild'].to_numpy())
y_fit = final_binned_df['MetaImpact_mean'].to_numpy()

popt, pcov = curve_fit(
    function_3, 
    x_fit_3, 
    y_fit, 
    sigma=final_binned_df['sem'].to_numpy(), # Pesa di più i punti con SEM minore
    absolute_sigma=True 
)

print(popt)
print(np.sqrt(np.diag(pcov)))

#plt.style.use('seaborn-v0_8-whitegrid') # Stile pulito
plt.figure(figsize=(12, 8))

# Definiamo una palette di colori basata sul numero di NbChild
# 'viridis' o 'plasma' sono ottime per dati scientifici
unique_nb = np.arange(2, min(9, maximum))
colors = plt.cm.plasma(np.linspace(0, 0.8, len(unique_nb))) 

Y_fit, delta_fit, mu_fit = popt

for i, nb_val in enumerate(unique_nb):
    subset_plot = final_binned_df[final_binned_df['NbChild'] == nb_val]
    
    if not subset_plot.empty:
        current_color = colors[i]
        
        # 1. Plot dei dati con barre di errore sottili e colore coordinato
        plt.errorbar(
            subset_plot['MetaVolume'], 
            subset_plot['MetaImpact_mean'], 
            yerr=subset_plot['sem'], 
            fmt='o',            
            capsize=0,          # Rimuove le "cap" per un look più moderno
            markersize=6, 
            label=f'{nb_val}',
            color=current_color,
            ecolor=current_color,
            alpha=0.5,
            markeredgecolor='white', # Aggiunge un bordo bianco ai punti per staccarli
            markeredgewidth=0.5
        )
        
        # 2. Plot del fit con linea continua più spessa
        x_range = np.logspace(np.log10(subset_plot['MetaVolume'].min()), 
                            np.log10(subset_plot['MetaVolume'].max()), 100)
        y_theoretical = function_3((x_range, nb_val), *popt)
        
        plt.plot(x_range, y_theoretical, linestyle='-', color=current_color, lw=2, alpha=0.8)

# Raffinatezze estetiche
plt.xscale('log')
plt.yscale('log')

# LaTeX per le etichette degli assi
plt.xlabel(r'MetaVolume normalized: $Q/V_D$', fontsize=13)
plt.ylabel(r'Market Impact: $\mathcal{I}$', fontsize=13)

plt.title(f'Square Root Law Fit: $\mathcal{{I}} \sim (Q/V_D)^{{{delta_fit:.2f}}} \cdot N^{{{mu_fit:.2f}}}$', 
        fontsize=14, pad=20)

# Legenda posizionata fuori o in un angolo pulito
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()

# Salvataggio in alta risoluzione
plt.savefig('images\\square_root_verify.png', dpi=300, bbox_inches='tight')
#plt.show()

# 1. Preparazione e calcolo delle nuove variabili
df_res = synthetic_meta[['MetaVolume', 'DailyVolume', 'TradedVolume', 'NbChild', 'MetaImpact']].copy()

# Definiamo le variabili per function_4
df_res['a'] = df_res['MetaVolume'] * df_res['DailyVolume'] / df_res['TradedVolume']
df_res['b'] = df_res['TradedVolume'] / df_res['DailyVolume']

maximum = 6
mask = (df_res['MetaVolume'] > 2e-5) & (df_res['MetaVolume'] < 1e-3)
df_res = df_res[mask & (df_res['NbChild'] > 1) & (df_res['NbChild'] < maximum)]

binned_data_list = []

# 2. Raggruppamento per NbChild e Binning 2D (Aree)
for nb_child_val in sorted(df_res['NbChild'].unique()):
    subset = df_res[df_res['NbChild'] == nb_child_val].copy()
    
    if len(subset) > 20: # Alziamo la soglia perché il binning 2D richiede più dati
        # Creiamo i bin logaritmici per 'a' e 'b'
        bins_a = np.logspace(np.log10(subset['a'].min()), np.log10(subset['a'].max()), 11)
        bins_b = np.logspace(np.log10(subset['b'].min()), np.log10(subset['b'].max()), 11)
        
        subset['bin_a'] = pd.cut(subset['a'], bins=bins_a, include_lowest=True)
        subset['bin_b'] = pd.cut(subset['b'], bins=bins_b, include_lowest=True)
        
        # Raggruppiamo per le "Aree" definite dalla combinazione dei due bin
        grouped = subset.groupby(['bin_a', 'bin_b'], observed=True).agg({
            'a': 'mean',
            'b': 'mean',
            'MetaImpact': ['mean', 'std', 'count']
        }).dropna()

        grouped.columns = ['a_mean', 'b_mean', 'MetaImpact_mean', 'MetaImpact_std', 'sample_count']
        
        # Scartiamo bin con un solo punto (std non calcolabile)
        grouped = grouped[grouped['sample_count'] > 1]
        
        grouped['NbChild'] = nb_child_val
        binned_data_list.append(grouped)

# 3. Consolidamento e Fit a 4 parametri
if binned_data_list:
    final_binned_df = pd.concat(binned_data_list)
    final_binned_df['sem'] = final_binned_df['MetaImpact_std'] / np.sqrt(final_binned_df['sample_count'])

    # Input per function_4: (a, b, N)
    x_fit_4 = (
        final_binned_df['a_mean'].to_numpy(),
        final_binned_df['b_mean'].to_numpy(),
        final_binned_df['NbChild'].to_numpy()
    )
    y_fit = final_binned_df['MetaImpact_mean'].to_numpy()

    try:
        popt4, pcov4 = curve_fit(
            function_4, 
            x_fit_4, 
            y_fit, 
            sigma=final_binned_df['sem'].to_numpy(),
            absolute_sigma=True
        )
        
        print(f"--- Fit 4 Parametri (Aree) per {path} ---")
        print(f"Parametri [Y, delta, alpha, mu]:\n{popt4}")
        print(f"Errori (std dev):\n{np.sqrt(np.diag(pcov4))}")
        
    except Exception as e:
        print(f"Errore nel fit a 4 parametri: {e}")