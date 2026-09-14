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
    """Common logic to bin MetaVolume and calculate stats."""
    v_min, v_max = df['MetaVolume'].min(), df['MetaVolume'].max()
    if v_min <= 0 or v_max <= 0 or v_min == v_max:
        return None

    bins = np.logspace(np.log10(v_min), np.log10(v_max), n_bins)
    df = df.copy()
    df['bin'] = pd.cut(df['MetaVolume'], bins=bins, include_lowest=True)

    grouped = df.groupby('bin', observed=True).agg({
        'MetaVolume': ['mean', 'std', 'count'],
        'MetaImpact': ['mean', 'std']
    }).dropna()

    grouped.columns = [
        'MetaVolume_mean', 'MetaVolume_std', 'sample_count',
        'MetaImpact_mean', 'MetaImpact_std'
    ]

    return grouped, bins

def load_model_data(model, file_name_con_estensione, min_child=2):
    """Load and clean data for a given model ('' means real data)."""
    folder_prefix = f"meta_{model}" if model else "meta"
    path = os.path.join("database", folder_prefix, f"meta_{file_name_con_estensione}")
    data = pd.read_csv(path)
    return data[data['NbChild'] >= min_child].copy()

def model_display_name(model):
    """Human-readable label for a model string."""
    return "Real data" if model == '' else model


def plot_aggregate_impact(df, image_name, n_bins=51):
    """Single aggregate plot with full fit params (Y and delta) in the legend."""
    res = bin_data(df, n_bins)
    if res is None:
        return
    grouped, bins = res

    # Filter high-frequency bins for fitting (exact condition from impact_volume_complete.py)
    max_samples = grouped['sample_count'].max()
    grouped_high_freq = grouped[grouped['sample_count'] > 0.5 * max_samples].copy()

    fit = None
    if not grouped_high_freq.empty and len(grouped_high_freq) > 2:
        fit = robust_power_law_fit(
            grouped_high_freq['MetaVolume_mean'].values,
            grouped_high_freq['MetaImpact_mean'].values,
            grouped_high_freq['MetaVolume_std'].values,
            grouped_high_freq['MetaImpact_std'].values,
            grouped_high_freq['sample_count'].values
        )

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r'$Q$')
    ax1.set_ylabel(r'$I(Q)$')
    ax2.set_ylabel('Frequency')

    ax2.hist(df['MetaVolume'], bins=bins, color='lightgrey', alpha=0.6)

    # Pre-calculate SEM for exact plotting of uncertainties
    x_sem = np.where(grouped['sample_count'] > 1, grouped['MetaVolume_std'] / np.sqrt(grouped['sample_count']), 1e-8)
    y_sem = np.where(grouped['sample_count'] > 1, grouped['MetaImpact_std'] / np.sqrt(grouped['sample_count']), 1e-8)

    ax1.errorbar(
        grouped['MetaVolume_mean'], grouped['MetaImpact_mean'],
        xerr=x_sem, yerr=y_sem,
        marker='o', linestyle='', color='C0', label='Binned data'
    )

    if fit:
        Y, delta, Y_err, delta_err = fit
        x_line = np.logspace(
            np.log10(grouped_high_freq['MetaVolume_mean'].min()),
            np.log10(grouped_high_freq['MetaVolume_mean'].max()), 100
        )
        fit_label = rf'$Y={Y:.2e} \pm {Y_err:.2e},\ \delta={delta:.3f} \pm {delta_err:.3f}$'
        ax1.plot(x_line, Y * (x_line**delta), 'k--', label=fit_label)
        print(f"Aggregate Fit: Y={Y:.4e} ± {Y_err:.4e}, delta={delta:.4f} ± {delta_err:.4f}")

    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, which='major', linewidth=1.0, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join('images', f'{image_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_comparison(file_name_con_estensione, models, image_name,
                              n_bins=51, vertical_shift=10.0):
    """Comparison plot with datasets shifted vertically."""
    records = []

    for model in models:
        label_base = model_display_name(model)
        try:
            df = load_model_data(model, file_name_con_estensione)
        except FileNotFoundError:
            print(f"[WARNING] File not found for model='{model}', skipping.")
            continue
        except Exception as e:
            print(f"[WARNING] Could not load model='{model}': {e}, skipping.")
            continue

        res = bin_data(df, n_bins)
        if res is None:
            print(f"[WARNING] Binning failed for model='{model}', skipping.")
            continue
        grouped, _ = res

        max_samples = grouped['sample_count'].max()
        grouped_high_freq = grouped[grouped['sample_count'] > 0.5 * max_samples].copy()

        if grouped_high_freq.empty or len(grouped_high_freq) <= 2:
            print(f"[{label_base}] Not enough high frequency bins, skipping.")
            continue

        fit = robust_power_law_fit(
            grouped_high_freq['MetaVolume_mean'].values,
            grouped_high_freq['MetaImpact_mean'].values,
            grouped_high_freq['MetaVolume_std'].values,
            grouped_high_freq['MetaImpact_std'].values,
            grouped_high_freq['sample_count'].values
        )

        if fit is None:
            print(f"[{label_base}] Fit failed, skipping.")
            continue

        records.append(dict(model=model, label=label_base, grouped=grouped, grouped_fit=grouped_high_freq, fit=fit))

    if not records:
        print("[ERROR] No models could be fitted; aborting comparison plot.")
        return

    # Sort ascending by delta
    records.sort(key=lambda r: r['fit'][1])
    n = len(records)
    palette = cm.tab10(np.linspace(0, 1, max(n, 1)))

    fig, ax1 = plt.subplots(figsize=(10, 7))
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

    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, which='both', linewidth=1.0, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join('images', f'{image_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {os.path.join('images', f'{image_name}.png')}")


if __name__ == "__main__":
    file_name_con_estensione = '20_power_2.0.csv'
    function_clean = file_name_con_estensione.replace('.csv', '')

    models = ['', 'ar_1000', 'var_1000', 'delta_0.5_1000', 'lmf_1.5_50_mem_tim_sqrt', 'lmf_1.5_50_mem_tim_lin']

    _orig_load_model_data = load_model_data

    target_dir = "impact_volume_curve_analysis_10"
    min_child = 10

    print("\n" + "="*70)
    print(f"AVVIO ANALISI: {target_dir.upper()} (n >= {min_child})")
    print("="*70)

    os.makedirs(os.path.join("images", target_dir), exist_ok=True)

    load_model_data = lambda m, f: _orig_load_model_data(m, f, min_child=min_child)

    img_comparison = f"{target_dir}/{function_clean}_comparison"
    print(f"\n[GENERAZIONE] Grafico di confronto complessivo: images/{img_comparison}.png")
    plot_aggregate_comparison(
        file_name_con_estensione,
        models=models,
        image_name=img_comparison,
        vertical_shift=10.0,
    )

    for model in models:
        label = model_display_name(model)
        img_prefix = f"{model + '_' if model else ''}"
        nome_img_aggregato = f"{target_dir}/{img_prefix}{function_clean}"

        try:
            df_clean = load_model_data(model, file_name_con_estensione)
            print(f" └─ [{label}] Generazione plot individuale: images/{nome_img_aggregato}.png")
            plot_aggregate_impact(df_clean, nome_img_aggregato)
        except Exception as e:
            print(f" └─ [{label}] ERRORE: {e}")

    load_model_data = _orig_load_model_data
    print("\n[FINISH] Tutte le analisi sono state completate con successo.")