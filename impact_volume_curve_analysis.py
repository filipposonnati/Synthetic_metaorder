import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
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

def linear_model(log_x, slope, intercept):
    """Modello lineare per la scala logaritmica."""
    return slope * log_x + intercept

def robust_power_law_fit(x_fit_data, y_fit_data, x_std, y_std, counts):
    """
    Performs the exact 3-pass linear log-log fit with effective variance propagation
    as implemented in impact_volume_complete.py.
    """
    if len(x_fit_data) <= 2:
        return None

    try:
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
        perr = np.sqrt(np.diag(pcov))
        slope_err, intercept_err = perr[0], perr[1]

        # Back-transform parameters to physical power-law terms
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
    ax1.set_ylabel(r'$I(Q)$ (shifted)')
    ax1.set_xlim([1e-6, 1e-2])

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

        ax1.plot(
            grouped['MetaVolume_mean'], grouped['MetaImpact_mean'] * shift,
            marker='o', linestyle='', color=color, alpha=0.7, markersize=4
        )

        x_line = np.logspace(
            np.log10(grouped_fit['MetaVolume_mean'].min()),
            np.log10(grouped_fit['MetaVolume_mean'].max()), 200
        )
        shift_str = rf" ($\times {vertical_shift:.0f}^{{{shift_exp}}}$)" if shift_exp != 0 else ""
        fit_label = rf"{label_base}: $\delta={delta:.3f} \pm {delta_err:.3f}$" + shift_str
        ax1.plot(
            x_line, (Y * shift) * (x_line**delta),
            linestyle='--', linewidth=1.8, color=color,
            label=fit_label, solid_capstyle='butt'
        )

    print("-" * 50)

    ax1.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        fontsize=10,
        ncol=2  # Dispone le voci della legenda su 2 colonne
    )
    ax1.grid(True, which='both', linewidth=1.0, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join('images', f'{image_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {os.path.join('images', f'{image_name}.png')}")


if __name__ == "__main__":
    file_name_con_estensione = '20_power_2.0.csv'
    function_clean = file_name_con_estensione.replace('.csv', '')

    models = ['', 'nn', 'reg_delta_1.0_1000', 'reg_delta_0.5_1000', 'reg_delta_0.0_1000', 'lmf_1.5_0.3_real_tim_sqrt', 'lmf_1.5_0.3_real_tim_lin', 'real_reg_lin', 'real_reg_sqrt', 'real_tim_sqrt', 'real_tim_lin']

    target_dir = "impact_volume_curve_analysis"
    min_child = 2

    print("\n" + "="*70)
    print(f"AVVIO ANALISI: {target_dir} (n >= {min_child})")
    print("="*70)

    os.makedirs(os.path.join("images", target_dir), exist_ok=True)

    img_comparison = f"{target_dir}/{function_clean}_comparison"
    print(f"Grafico di confronto complessivo: images/{img_comparison}.png")
    plot_aggregate_comparison(
        file_name_con_estensione,
        models=models,
        image_name=img_comparison,
        vertical_shift=10.0,
        min_child=min_child,
    )