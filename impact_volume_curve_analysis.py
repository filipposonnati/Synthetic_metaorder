import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.odr import ODR, Model, RealData
import os

# Global Plot Configuration
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14
})

def linear_func(p, x):
    """Modello lineare per ODR (spazio log-log): p = [slope, intercept]."""
    return p[0] * x + p[1]

def robust_power_law_fit(x_fit_data, y_fit_data, x_std, y_std, counts):
    """
    Fit ODR in spazio log-log, identico a impact_volume_curve.py:
    errori su X e Y (SEM propagati in log10), beta0=[1, 0].
    """
    if len(x_fit_data) <= 2:
        return None

    try:
        # Incertezza della media (Standard Error of Mean = std / sqrt(N))
        x_sem = np.where(counts > 1, x_std / np.sqrt(counts), 1e-8)
        y_sem = np.where(counts > 1, y_std / np.sqrt(counts), 1e-8)

        log_x = np.log10(x_fit_data)
        log_y = np.log10(y_fit_data)

        # Incertezza propagata in log10: d(log10(z)) = dz / (z * ln(10))
        sigma_log_x = x_sem / (x_fit_data * np.log(10))
        sigma_log_y = y_sem / (y_fit_data * np.log(10))

        sigma_log_x = np.maximum(sigma_log_x, 1e-6)
        sigma_log_y = np.maximum(sigma_log_y, 1e-6)

        model_odr = Model(linear_func)
        data = RealData(log_x, log_y, sx=sigma_log_x, sy=sigma_log_y)
        out = ODR(data, model_odr, beta0=[1.0, 0.0]).run()

        slope, intercept = out.beta
        slope_err, intercept_err = out.sd_beta

        delta = slope
        delta_err = slope_err

        Y = 10**intercept
        Y_err = np.log(10) * Y * intercept_err

        return Y, delta, Y_err, delta_err
    except Exception:
        return None

def bin_data(df, n_bins=51):
    """Common logic to bin MetaVolume and calculate stats (senza copiare il dataframe)."""
    v_min, v_max = df['MetaVolume'].min(), df['MetaVolume'].max()
    if v_min <= 0 or v_max <= 0 or v_min == v_max:
        return None

    bins = np.logspace(np.log10(v_min), np.log10(v_max), n_bins)

    # I bin vengono calcolati su una Series: nessun df.copy() con colonna aggiuntiva
    bin_idx = pd.cut(df['MetaVolume'], bins=bins, include_lowest=True)

    grouped = df.groupby(bin_idx, observed=True).agg(
        MetaVolume_mean=('MetaVolume', 'mean'),
        MetaVolume_std=('MetaVolume', 'std'),
        sample_count=('MetaVolume', 'count'),
        MetaImpact_mean=('MetaImpact', 'mean'),
        MetaImpact_std=('MetaImpact', 'std'),
    ).dropna()

    # Come in impact_volume_curve.py: solo bin con medie positive (necessario per il log)
    grouped = grouped[(grouped['MetaVolume_mean'] > 0) & (grouped['MetaImpact_mean'] > 0)]

    return grouped, bins

def load_model_data(model, file_name_con_estensione, min_child=2):
    """Load and clean data for a given model ('' means real data)."""
    folder_prefix = f"meta_{model}" if model else "meta"
    if os.path.exists(os.path.join("database", folder_prefix, f"meta_{file_name_con_estensione}_{min_child}")):
        path = os.path.join("database", folder_prefix, f"meta_{file_name_con_estensione}_{min_child}")
    else:
        path = os.path.join("database", folder_prefix, f"meta_{file_name_con_estensione}")
    # Legge solo le colonne necessarie
    data = pd.read_csv(path, usecols=['NbChild', 'MetaVolume', 'MetaImpact'])
    data = data[data['MetaVolume'] > 0]   # come in impact_volume_curve.py
    return data[data['NbChild'] >= min_child]

def model_display_name(model):
    """Human-readable label for a model string."""
    return "Real data" if model == '' else model

def process_model(model, file_name_con_estensione, min_child, n_bins):
    """
    Carica, binna e fitta UNA configurazione. Restituisce solo il record leggero
    (grouped, grouped_fit, fit) oppure None. Il dataframe grezzo vive solo qui
    dentro e viene rilasciato all'uscita dalla funzione.
    """
    label_base = model_display_name(model)
    try:
        df = load_model_data(model, file_name_con_estensione, min_child)
    except FileNotFoundError:
        print(f"[WARNING] File not found for model='{model}', skipping.")
        return None
    except Exception as e:
        print(f"[WARNING] Could not load model='{model}': {e}, skipping.")
        return None

    res = bin_data(df, n_bins)

    # Libera subito il dataframe grezzo: serve solo il risultato del binning
    del df
    gc.collect()

    if res is None:
        print(f"[WARNING] Binning failed for model='{model}', skipping.")
        return None
    grouped, _ = res

    max_samples = grouped['sample_count'].max()
    grouped_high_freq = grouped[grouped['sample_count'] > 0.5 * max_samples].copy()

    if grouped_high_freq.empty or len(grouped_high_freq) <= 2:
        print(f"[{label_base}] Not enough high frequency bins, skipping.")
        return None

    fit = robust_power_law_fit(
        grouped_high_freq['MetaVolume_mean'].values,
        grouped_high_freq['MetaImpact_mean'].values,
        grouped_high_freq['MetaVolume_std'].values,
        grouped_high_freq['MetaImpact_std'].values,
        grouped_high_freq['sample_count'].values
    )

    if fit is None:
        print(f"[{label_base}] Fit failed, skipping.")
        return None

    return dict(model=model, label=label_base, grouped=grouped, grouped_fit=grouped_high_freq, fit=fit)


def plot_aggregate_comparison(file_name_con_estensione, models, image_name,
                              n_bins=51, vertical_shift=10.0, min_child=2):
    """Comparison plot with datasets shifted vertically."""
    records = []

    for model in models:
        rec = process_model(model, file_name_con_estensione, min_child, n_bins)
        if rec is not None:
            records.append(rec)

    if not records:
        print("[ERROR] No models could be fitted; aborting comparison plot.")
        return

    # Sort ascending by delta
    records.sort(key=lambda r: r['fit'][1])
    n = len(records)
    palette = cm.tab10(np.linspace(0, 1, max(n, 1)))

    fig, ax1 = plt.subplots(figsize=(10, 12))
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r'$Q$')
    if vertical_shift != 1:
        ax1.set_ylabel(r'$I(Q)$ (shifted)')
    else:
        ax1.set_ylabel(r'$I(Q)$')
    ax1.set_xlim([1e-5, 1e-3])
    ax1.set_ylim([2e-3, 2e-2])

    print(f"\n{'Rank':>5}  {'Model':<20}  {'delta':>7}  {'shift':>8}")
    print("-" * 50)

    for rank, (rec, color) in enumerate(zip(records, palette)):
        shift_exp = n - 1 - rank
        shift = vertical_shift ** shift_exp
        grouped = rec['grouped']
        grouped_fit = rec['grouped_fit']
        Y, delta, Y_err, delta_err = rec['fit']
        label_base = rec['label']

        print(f"{rank:>5}  {label_base:<20}  {delta:>7.3f}  ×{shift:>7.2f}")

        x_line = np.logspace(
            np.log10(grouped_fit['MetaVolume_mean'].min()),
            np.log10(grouped_fit['MetaVolume_mean'].max()), 200
        )
        shift_str = rf" ($\times {vertical_shift:.0f}^{{{shift_exp}}}$)" if shift_exp != 0 and vertical_shift != 1.0 else ""
        fit_label = rf"{label_base.replace('_', ' ').capitalize()}: $\delta={delta:.2f} \pm {delta_err:.2f}$" + shift_str

        ax1.plot(
            grouped['MetaVolume_mean'], grouped['MetaImpact_mean'] * shift,
            marker='o', linestyle='', color=color, alpha=1.0, markersize=5
        )

        # E fai lo stesso per la linea di fit teorica
        ax1.plot(
            x_line, (Y * shift) * (x_line**(delta)),
            linestyle='--', linewidth=2.0, color=color,
            label=fit_label, solid_capstyle='butt'
        )

    print("-" * 50)

    ax1.legend(
        loc='lower left',
        bbox_to_anchor=(0.0, 1.02),
        ncol=1
    )
    ax1.grid(True, which='both', linewidth=1.0, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join('images', f'{image_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {os.path.join('images', f'{image_name}.png')}")

    return records

def plot_delta_comparison(records, image_name):
    """Crea un grafico a barre d'errore per confrontare direttamente i delta."""
    labels = [r['label'].replace('_', ' ').capitalize() for r in records]
    deltas = [r['fit'][1] for r in records]
    delta_errs = [r['fit'][3] for r in records]

    fig, ax = plt.subplots()
    x_pos = np.arange(len(labels))
    
    # Disegna i punti con le barre d'errore
    ax.errorbar(x_pos, deltas, yerr=delta_errs, fmt='.', elinewidth=1.5, capsize=4, color='black', ecolor='black')

    # Linea di riferimento basata sui dati reali (il primo record)
    ax.axhline(y=0.5, color='blue', linestyle='--', alpha=1.0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel(r'$\delta$')
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)
    #ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('images', f'{image_name}_delta_comparison.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    file_name_con_estensione = '20_power_2.0.csv'
    function_clean = file_name_con_estensione.replace('.csv', '')

    models = ['', 'tim_1.0_1000', 'tim_0.5_1000', 'tim_0.0_1000']
    #models = ['', 'tim_param_1.0_1000', 'tim_param_0.5_1000', 'tim_param_0.3_1000', 'tim_param_0.0_1000']

    target_dir = "impact_volume_curve_analysis"
    min_child = 2

    print("\n" + "="*70)
    print(f"AVVIO ANALISI: {target_dir} (n >= {min_child})")
    print("="*70)

    os.makedirs(os.path.join("images", target_dir), exist_ok=True)

    img_comparison = f"{target_dir}/tim_{function_clean}_comparison"
    print(f"Grafico di confronto complessivo: images/{img_comparison}.png")
    records = plot_aggregate_comparison(
        file_name_con_estensione,
        models=models,
        image_name=img_comparison,
        vertical_shift=1.0,
        min_child=min_child,
    )

    if records:
        plot_delta_comparison(records, img_comparison)