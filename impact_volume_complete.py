import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
from os import listdir
import shutil
from pathlib import Path

# Code to verify the square root impact law

plt.rcParams.update({
    'font.size': 12,          # Dimensione base per tutto il testo
    'axes.titlesize': 20,     # Titolo
    'axes.labelsize': 16,     # Etichette assi
    'xtick.labelsize': 12,    # Numeri asse X
    'ytick.labelsize': 12,    # Numeri asse Y
    'legend.fontsize': 14     # Legenda
})

# ==========================================
# CARTELLA DI OUTPUT PER LE IMMAGINI
# ==========================================
output_dir = os.path.join('images', 'impact_volume_complete')
os.makedirs(output_dir, exist_ok=True)

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
    sep=',',
    parse_dates=['BeginTime', 'EndTime']
)


def run_analysis(synthetic_meta, min_nb_child, max_nb_child=20, comparison_operator='>'):
    """
    Esegue l'intera analisi (filtraggio, binning stratificato, binning globale,
    plot) per una data soglia minima su NbChild.

    comparison_operator: '>' oppure '>=' per decidere come applicare min_nb_child
    """

    # 1. Preparazione e Filtraggio
    df_res = synthetic_meta[['MetaVolume', 'DailyVolume', 'TradedVolume', 'NbChild', 'MetaImpact']].copy()

    if comparison_operator == '>':
        df_res = df_res[df_res['NbChild'] > min_nb_child]
    else:
        df_res = df_res[df_res['NbChild'] >= min_nb_child]

    df_res = df_res[df_res['NbChild'] <= max_nb_child]

    if df_res.empty:
        print(f"Nessun dato disponibile per la soglia NbChild {comparison_operator} {min_nb_child}. Analisi saltata.")
        return

    maximum = np.max(df_res['NbChild'].unique())
    minimum = np.min(df_res['NbChild'].unique())

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
                'MetaImpact': ['mean', 'std', 'count']
            }).dropna()

            # Appiattiamo le colonne rinominandole per comodità
            grouped.columns = ['MetaVolume_mean', 'MetaVolume_std', 'MetaImpact_mean', 'MetaImpact_std', 'sample_count']

            max_samples = grouped['sample_count'].max()
            grouped = grouped[grouped['sample_count'] > 0.5 * max_samples]

            # Aggiungiamo NbChild per il fit
            grouped['NbChild'] = nb_child_val
            binned_data_list.append(grouped)

    if not binned_data_list:
        print(f"Dati insufficienti per creare i bin con soglia NbChild {comparison_operator} {min_nb_child}. Analisi saltata.")
        return

    # Consolidamento dati stratificati per NbChild
    final_binned_df = pd.concat(binned_data_list)
    final_binned_df['y_err'] = final_binned_df['MetaImpact_std'] / np.sqrt(final_binned_df['sample_count'])
    final_binned_df['x_err'] = final_binned_df['MetaVolume_std'] / np.sqrt(final_binned_df['sample_count'])

    # ==========================================
    # BINNING DEI DATI COMPLESSIVI
    # ==========================================
    min_vol_global = df_res['MetaVolume'].min()
    max_vol_global = df_res['MetaVolume'].max()
    bins_global = np.logspace(np.log10(min_vol_global), np.log10(max_vol_global), 31)

    df_global = df_res.copy()
    df_global['bin'] = pd.cut(df_global['MetaVolume'], bins=bins_global, include_lowest=True)

    global_grouped = df_global.groupby('bin', observed=True).agg({
        'MetaVolume': ['mean', 'std'],
        'MetaImpact': ['mean', 'std', 'count']
    }).dropna()

    global_grouped.columns = ['MetaVolume_mean', 'MetaVolume_std', 'MetaImpact_mean', 'MetaImpact_std', 'sample_count']
    max_samples_global = global_grouped['sample_count'].max()
    global_grouped = global_grouped[global_grouped['sample_count'] > 0.1 * max_samples_global]
    # ==========================================

    plt.figure(figsize=(12, 8))

    unique_nb = np.arange(minimum, maximum + 1)
    colors = plt.get_cmap('tab20')

    for i, nb_val in enumerate(unique_nb):
        subset_plot = final_binned_df[final_binned_df['NbChild'] == nb_val]

        if not subset_plot.empty:
            current_color = colors(i % 20)

            plt.plot(
                subset_plot['MetaVolume_mean'],
                subset_plot['MetaImpact_mean'],
                marker='',
                markersize=6,
                label=f'{nb_val}',
                color=current_color
            )

    # PLOT DEL BINNING COMPLESSIVO (Linee nere tratteggiate con marker quadrati)
    plt.plot(
        global_grouped['MetaVolume_mean'],
        global_grouped['MetaImpact_mean'],
        linestyle='--',
        marker='s',
        markersize=5,
        color='black',
        linewidth=2,
        label='Total'
    )

    # 2. Plot del fit con linea continua più spessa
    x_range = np.logspace(-5, -3, 100)
    y_theoretical = x_range**0.5

    plt.plot(x_range, y_theoretical, linestyle='-', color='b', lw=2, label='$Q^{0.5}$')

    # Raffinatezze estetiche
    plt.xscale('log')
    plt.yscale('log')

    # LaTeX per le etichette degli assi
    plt.xlabel(r'$Q$')
    plt.ylabel(r'$I$')

    plt.title(f'NbChild {comparison_operator} {min_nb_child}')

    # Legenda posizionata fuori o in un angolo pulito
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    filename = f'impact_volume_complete_nbchild_{comparison_operator.replace(">", "gt").replace("=", "e")}_{min_nb_child}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f'Salvata immagine: {filepath}')

    plt.show()
    plt.close()


# ==========================================
# ESECUZIONE DELL'ANALISI PER LE VARIE SOGLIE
# ==========================================

# Analisi originale: NbChild > 1
run_analysis(synthetic_meta, min_nb_child=1, comparison_operator='>')

# Analisi con NbChild >= 5
run_analysis(synthetic_meta, min_nb_child=5, comparison_operator='>=')

# Analisi con NbChild >= 10
run_analysis(synthetic_meta, min_nb_child=10, comparison_operator='>=')