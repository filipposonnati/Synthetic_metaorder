import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
import os
from scipy.optimize import curve_fit
import re

def post_impact(x, a, beta):
    return a * (x**(1 - beta) - (x - 1)**(1 - beta))

def der_post_impact(x, a, beta):
    return a * (1 - beta) * (x**(-beta) - (x - 1)**(-beta))

plt.rcParams.update({
    'font.size': 12,          # Dimensione base per tutto il testo
    'axes.titlesize': 20,     # Titolo
    'axes.labelsize': 16,     # Etichette assi
    'xtick.labelsize': 12,    # Numeri asse X
    'ytick.labelsize': 12,    # Numeri asse Y
    'legend.fontsize': 14     # Legenda
})

def save(index_time):
    paths = np.array(listdir('..\\database\\trades_20_power_2.0'))

    for path in paths:
        print(path)
        trades = pd.read_csv(
            f'..\\database\\trades_20_power_2.0\\{path}', 
            sep=',',
            parse_dates=['BeginTime']
        )

        meta = pd.read_csv(
            f'..\\database\\meta_20_power_2.0\\meta{path[6:]}', 
            sep=',',
            parse_dates=['BeginTime', 'EndTime']
        )

        meta = meta[meta['NbChild'] >= 5]

        for index, meta_row in meta.iterrows():
            meta_id = index # Usiamo l'indice come ID del metaordine per semplicità
            t_begin_meta = meta_row['BeginTime']
            t_end_meta = meta_row['EndTime']
            meta_duration = t_end_meta - t_begin_meta
            meta_sign = meta_row['sign']
            meta_impact = meta_row['MetaImpact']
            meta_volume = meta_row['MetaVolume']
            
            # 3.1. Calcola l'intervallo di tempo post-metaordine (T_End_Meta, T_End_Meta + MetaDuration)
            end_window = t_end_meta + meta_duration * index_time
            t_start_window = t_end_meta + meta_duration * (index_time - 1)
            
            # 3.2. Filtra i trade nell'intervallo post-metaordine: 
            post_trades = trades[
                (trades['BeginTime'] >= t_start_window) & 
                (trades['BeginTime'] < end_window)
            ].copy()
            
            # Calcolo dell'Impatto Cumulativo Normalizzato
            begin_meta_price = meta_row['BeginMid']
            #post_impact = (post_trades['EndMid'] - begin_meta_price) * meta_sign
            post_impact = (post_trades['BeginMid'] - begin_meta_price) * meta_sign
            
            # Calcolo del Time Lag Normalizzato per OGNI trade nella finestra
            # t_exec_trade - t_end_meta (in secondi)
            time_lag_seconds = (post_trades['BeginTime'] - t_begin_meta).dt.total_seconds()
            
            normalized_time_lag = time_lag_seconds / meta_duration.total_seconds()

            post_trades['NormalizedTime'] = normalized_time_lag
            post_trades['Ratio'] = post_impact / np.sqrt(meta_volume) 

            file_exists = os.path.isfile(f'..\\database\\post_trades\\20_power_2.0_{index_time}.csv')
            post_trades[['NormalizedTime', 'Ratio']].to_csv(f'database\\post_trades\\20_power_2.0_{index_time}.csv', mode='a', index=False, header=not file_exists)

length = 2

for index_time in range(length):
    if not os.path.exists(f'..\\database\\post_trades\\20_power_2.0_{index_time}.csv'):
        save(index_time)

bins_analysis = np.array([])
times_analysis = np.array([])

bins_analysis_err = np.array([])
times_analysis_err = np.array([])

paths = np.array(listdir('..\\database\\post_trades'))

for path in paths[:length]:
    print(path)
    post_trades = pd.read_csv(
        f'..\\database\\post_trades\\{path}', 
        sep=','
    )

    match = re.search(r"_(\d+)\.csv$", path)

    index = int(match.group(1))

    if index == 0:
        # Bin con distribuzione a potenza (esponente 2): fitti vicino a 0, radi verso 1
        n_bins = 30
        bins = np.linspace(0.0, 1.0, n_bins + 1) ** 3  # bordi in [0, 1]
    else:
        bins = np.linspace(0.0 + index, 1.0 + index, 11)

    post_trades['bin'] = pd.cut(post_trades['NormalizedTime'], bins=bins, include_lowest=True) 

    grouped = post_trades.groupby('bin', observed=True).agg({
        'NormalizedTime': ['mean', 'std'],
        'Ratio': ['mean', 'std', 'count']
    })

    grouped.columns = ['NormalizedTime_mean', 'NormalizedTime_std', 'Ratio_mean', 'Ratio_std', 'count']

    x = grouped['NormalizedTime_mean'].to_numpy()
    y = grouped['Ratio_mean'].to_numpy()

    bins_analysis = np.concatenate((bins_analysis, y))
    times_analysis = np.concatenate((times_analysis, x))

    x_err = grouped['NormalizedTime_std'].to_numpy() / np.sqrt(grouped['count'].to_numpy())
    y_err = grouped['Ratio_std'].to_numpy() / np.sqrt(grouped['count'].to_numpy())

    bins_analysis_err = np.concatenate((bins_analysis_err, y_err))
    times_analysis_err = np.concatenate((times_analysis_err, x_err))

mask = times_analysis > 1.0

popt, pcov = curve_fit(post_impact, times_analysis[mask], bins_analysis[mask])

Y = popt[0]
beta = popt[1]

Y_err = np.sqrt(pcov[0][0])
beta_err = np.sqrt(pcov[1][1])

print(f'Fit (no err): Y = {Y} +- {Y_err}, beta = {beta} +- {beta_err}')

popt, pcov = curve_fit(post_impact, times_analysis[mask], bins_analysis[mask], sigma=bins_analysis_err[mask], absolute_sigma=True)

Y = popt[0]
beta = popt[1]

Y_err = np.sqrt(pcov[0][0])
beta_err = np.sqrt(pcov[1][1])

print(f'Fit (y err): Y = {Y} +- {Y_err}, beta = {beta} +- {beta_err}')

for i in range(10):
    err = np.sqrt(bins_analysis_err**2 + (times_analysis_err * der_post_impact(times_analysis, Y, beta))**2)

    popt, pcov = curve_fit(post_impact, times_analysis[mask], bins_analysis[mask], sigma=err[mask], absolute_sigma=True)

    Y = popt[0]
    beta = popt[1]

    Y_err = np.sqrt(pcov[0][0])
    beta_err = np.sqrt(pcov[1][1])

print(f'Fit (eff err): Y = {Y} +- {Y_err}, beta = {beta} +- {beta_err}')

x_theoretical = np.linspace(1.0, length, 100)
plt.plot(x_theoretical, post_impact(x_theoretical, Y, beta), linestyle=':', color = "black")

#plt.errorbar(x, y, yerr=y_err, xerr=x_err, linestyle="", marker=".", label = f"20_power_2.0")
plt.plot(times_analysis, bins_analysis, linestyle="", marker="o", color='C0')

plt.xlabel(r'$t / T$')
plt.ylabel(r'$I(t) / \sqrt{Q}$')
#plt.legend()
plt.grid(True, which="both", ls="-")

plt.savefig('..\\images\\impact_time_curve_ext.png')
plt.show()

# --- Distribuzione dei Ratio nel primo bin (t/T più vicino a 0) ---
post_trades_0 = pd.read_csv(
    f'..\\database\\post_trades\\20_power_2.0_0.csv',
    sep=','
)

n_bins_0 = 30
bins_0 = np.linspace(0.0, 1.0, n_bins_0 + 1) ** 3
first_bin_mask = (post_trades_0['NormalizedTime'] >= bins_0[0]) & (post_trades_0['NormalizedTime'] <= bins_0[1])
first_bin_data = post_trades_0.loc[first_bin_mask, 'Ratio']

print(f"\nPrimo bin: [{bins_0[0]:.5f}, {bins_0[1]:.5f}]")
print(f"  Count : {len(first_bin_data)}")
print(f"  Mean  : {first_bin_data.mean():.4f}")
print(f"  Median: {first_bin_data.median():.4f}")
print(f"  Std   : {first_bin_data.std():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Istogramma
axes[0].hist(first_bin_data, bins=50, color='C0', edgecolor='white', linewidth=0.4)
axes[0].axvline(first_bin_data.mean(),   color='red',    linestyle='--', label=f'Mean = {first_bin_data.mean():.3f}')
axes[0].axvline(first_bin_data.median(), color='orange', linestyle='--', label=f'Median = {first_bin_data.median():.3f}')
axes[0].set_xlabel(r'$I(t) / \sqrt{Q}$')
axes[0].set_ylabel('Count')
axes[0].set_title(f'First bin  $[{bins_0[0]:.4f},\\ {bins_0[1]:.4f}]$')
axes[0].legend()
axes[0].grid(True, ls='-')

# Log-scale per vedere le code
axes[1].hist(first_bin_data, bins=50, color='C0', edgecolor='white', linewidth=0.4, log=True)
axes[1].axvline(first_bin_data.mean(),   color='red',    linestyle='--', label=f'Mean = {first_bin_data.mean():.3f}')
axes[1].axvline(first_bin_data.median(), color='orange', linestyle='--', label=f'Median = {first_bin_data.median():.3f}')
axes[1].set_xlabel(r'$I(t) / \sqrt{Q}$')
axes[1].set_ylabel('Count (log scale)')
axes[1].set_title(f'First bin  $[{bins_0[0]:.4f},\\ {bins_0[1]:.4f}]$  — log scale')
axes[1].legend()
axes[1].grid(True, ls='-')

plt.tight_layout()
plt.savefig('..\\images\\first_bin_distribution.png')
plt.show()