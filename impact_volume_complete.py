import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit

# ==========================================
# PLOT STYLING CONFIGURATION
# ==========================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11
})

# Modello lineare per la scala logaritmica
def linear_model(log_x, slope, intercept):
    return slope * log_x + intercept

# ==========================================
# OUTPUT & DATA LOADING
# ==========================================
output_dir = os.path.join('images', 'impact_volume_complete')
os.makedirs(output_dir, exist_ok=True)

model = "lmf_1.5_0.3_real_tim_lin"
#model = 'real_reg'

dir = 'meta'
if model != "":
    dir = dir + "_" + model

dir_path = os.path.join('database', dir)
nb_traders = 20
kind = 'power'
exponent = 2.0

if kind == 'uniform':
    conf = f'{nb_traders}_{kind}'
else:
    conf = f'{nb_traders}_{kind}_{exponent}'

path = 'meta_' + conf

# ==========================================
# DEFINE RANGES & PLOT SETUP
# ==========================================
ranges_config = [
    {'min_val': 1,  'op': '>', 'label': r'$n > 1$',    'marker': 'o', 'color': 'tab:blue'},
    {'min_val': 5,  'op': '>=', 'label': r'$n \geq 5$',  'marker': 's', 'color': 'tab:orange'},
    {'min_val': 10, 'op': '>=', 'label': r'$n \geq 10$', 'marker': '^', 'color': 'tab:green'}
]

plt.figure(figsize=(10, 7))

all_x_min = []
all_x_max = []

# Loop over each range, bin the data, compute log-log fit, and plot
for cfg in ranges_config:
    if os.path.exists(os.path.join(dir_path, path + '_' + str(cfg['min_val']) + '.csv')):
        data_path = os.path.join(dir_path, path + '_' + str(cfg['min_val']) + '.csv')
    else:
        data_path = os.path.join(dir_path, path + '.csv')
    synthetic_meta = pd.read_csv(
        data_path,
        sep=',',
        parse_dates=['BeginTime', 'EndTime']
    )

    synthetic_meta = synthetic_meta[synthetic_meta['MetaVolume'] > 0]

    df_res = synthetic_meta[['MetaVolume', 'DailyVolume', 'TradedVolume', 'NbChild', 'MetaImpact']].copy()

    if cfg['op'] == '>':
        df_res = df_res[df_res['NbChild'] > cfg['min_val']]
    else:
        df_res = df_res[df_res['NbChild'] >= cfg['min_val']]

    #df_res = df_res[df_res['NbChild'] <= cfg['max_val']]

    if df_res.empty:
        continue

    # Global logarithmic binning for the range
    min_vol = df_res['MetaVolume'].min()
    max_vol = df_res['MetaVolume'].max()
    bins = np.logspace(np.log10(min_vol), np.log10(max_vol), 51)

    df_res['bin'] = pd.cut(df_res['MetaVolume'], bins=bins, include_lowest=True)

    # 1. Aggregate ALL bins
    grouped_all = df_res.groupby('bin', observed=True).agg({
        'MetaVolume': ['mean', 'std', 'count'],
        'MetaImpact': ['mean', 'std']
    }).dropna()

    grouped_all.columns = [
        'MetaVolume_mean', 'MetaVolume_std', 'sample_count',
        'MetaImpact_mean', 'MetaImpact_std'
    ]

    grouped_all = grouped_all[
        (grouped_all['MetaVolume_mean'] > 0) & 
        (grouped_all['MetaImpact_mean'] > 0)
    ]

    if grouped_all.empty:
        continue

    # Extract all binned points for plotting
    x_data_all = grouped_all['MetaVolume_mean'].values
    y_data_all = grouped_all['MetaImpact_mean'].values

    all_x_min.append(np.min(x_data_all))
    all_x_max.append(np.max(x_data_all))

    # Plot ALL binned data points
    plt.plot(
        x_data_all,
        y_data_all,
        linestyle='',
        marker=cfg['marker'],
        markersize=6,
        color=cfg['color'],
        label=f"Data {cfg['label']}"
    )

    # 2. Filter for HIGH-FREQUENCY bins only (used exclusively for fitting)
    max_samples = grouped_all['sample_count'].max()
    grouped_high_freq = grouped_all[grouped_all['sample_count'] > 0.5 * max_samples].copy()

    if not grouped_high_freq.empty and len(grouped_high_freq) > 2:
        x_fit_data = grouped_high_freq['MetaVolume_mean'].values
        y_fit_data = grouped_high_freq['MetaImpact_mean'].values

        x_std = grouped_high_freq['MetaVolume_std'].values
        y_std = grouped_high_freq['MetaImpact_std'].values
        counts = grouped_high_freq['sample_count'].values

        # Incertezza della media (Standard Error of Mean = std / sqrt(N))
        x_sem = np.where(counts > 1, x_std / np.sqrt(counts), 1e-8)
        y_sem = np.where(counts > 1, y_std / np.sqrt(counts), 1e-8)

        # Trasformazione nello spazio logaritmico
        log_x = np.log10(x_fit_data)
        log_y = np.log10(y_fit_data)

        # Incertezza propagata in log10: d(log10(z)) = dz / (z * ln(10))
        sigma_log_x = x_sem / (x_fit_data * np.log(10))
        sigma_log_y = y_sem / (y_fit_data * np.log(10))

        # Evita errori zero o negativi
        sigma_log_x = np.maximum(sigma_log_x, 1e-6)
        sigma_log_y = np.maximum(sigma_log_y, 1e-6)

        # Passaggio 1: Fit preliminare per stimare il coefficiente angolare (slope)
        p0_fit, _ = curve_fit(linear_model, log_x, log_y, sigma=sigma_log_y, absolute_sigma=True)
        slope_approx = p0_fit[0]

        # Passaggio 2: Calcolo Errore Efficace nello spazio logaritmico
        sigma_eff_log = np.sqrt(sigma_log_y**2 + (slope_approx * sigma_log_x)**2)

        # Passaggio 3: Fit finale con curve_fit usando gli errori efficaci
        popt, pcov = curve_fit(
            linear_model,
            log_x,
            log_y,
            sigma=sigma_eff_log,
            absolute_sigma=True
        )

        slope, intercept = popt
        # Errore sui parametri di best fit (deviazione standard = radice dei valori diagonali della matrice di covarianza)
        perr = np.sqrt(np.diag(pcov))
        slope_err, intercept_err = perr[0], perr[1]

        # Stampa a schermo degli errori sui parametri
        print(f"[{cfg['label']}] Slope: {slope:.4f} ± {slope_err:.4f} | Intercept: {intercept:.4f} ± {intercept_err:.4f}")

        # Grid per tracciare il fit
        x_fit = np.logspace(np.log10(x_fit_data.min()), np.log10(x_fit_data.max()), 100)
        y_fit = (10**intercept) * (x_fit**slope)

        # Plot linea di fit
        plt.plot(
            x_fit,
            y_fit,
            linestyle='--',
            linewidth=1.8,
            color=cfg['color'],
            label=f"Fit {cfg['label']}: ${slope:.3f} \pm {slope_err:.3f}$"
        )

# ==========================================
# THEORETICAL CURVE (Q^0.5)
# ==========================================
if all_x_min and all_x_max:
    global_x_min = min(all_x_min) / 2
    global_x_max = max(all_x_max) * 2
    x_ref = np.logspace(np.log10(global_x_min), np.log10(global_x_max), 100)
    y_ref = x_ref**0.5

    plt.plot(
        x_ref,
        y_ref,
        linestyle='-',
        color='black',
        linewidth=1.2,
        label=r'$\sqrt{Q}$'
    )

# Formatting
#plt.xlim([10**-5, 5 * 10**-3])
#plt.ylim([10**-3, 10**-1])

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$Q$')
plt.ylabel(r'$I$')
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
if model == '':
    filepath = os.path.join(output_dir, conf + '.png')
else:
    filepath = os.path.join(output_dir, model + '_' + conf + '.png')
plt.savefig(filepath, bbox_inches='tight')
print(f'Saved single fitted figure to: {filepath}')
plt.close()