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

def power_law(x, Y, delta):
    return Y * x**delta

def robust_power_law_fit(x, y, x_err, y_err, sample_count):
    """
    Performs an iterative power-law fit on the 'core' high-density data.
    """
    core_mask = sample_count > (0.5 * sample_count.max())
    xc, yc = x[core_mask], y[core_mask]
    xerr_c, yerr_c = x_err[core_mask], y_err[core_mask]

    if len(xc) < 2:
        return None

    try:
        popt, _ = curve_fit(power_law, xc, yc, maxfev=5000)
        Y, delta = popt

        for _ in range(10):
            eff_err = np.sqrt(yerr_c**2 + (Y * delta * xc**(delta - 1) * xerr_c)**2)
            popt, pcov = curve_fit(power_law, xc, yc, sigma=eff_err, absolute_sigma=True, maxfev=5000)
            Y, delta = popt

        return Y, delta, np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])
    except:
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
        'MetaVolume': ['mean', 'std'],
        'MetaImpact': ['mean', 'std', 'count']
    }).dropna()

    grouped.columns = ['x', 'x_std', 'y', 'y_std', 'count']
    grouped['x_err'] = grouped['x_std'] / np.sqrt(grouped['count'])
    grouped['y_err'] = grouped['y_std'] / np.sqrt(grouped['count'])

    return grouped, bins

def load_model_data(model, file_name_con_estensione):
    """Load and clean data for a given model ('' means real data)."""
    folder_prefix = f"meta_{model}" if model else "meta"
    path = f"database\\{folder_prefix}\\meta_{file_name_con_estensione}"
    data = pd.read_csv(path)
    return data[data['NbChild'] > 1].copy()

def model_display_name(model):
    """Human-readable label for a model string."""
    return "Real data" if model == '' else model


def plot_aggregate_impact(df, image_name, n_bins=51):
    """Single aggregate plot with full fit params (Y and delta) in the legend."""
    res = bin_data(df, n_bins)
    if res is None:
        return
    grouped, bins = res

    fit = robust_power_law_fit(
        grouped['x'], grouped['y'],
        grouped['x_err'], grouped['y_err'],
        grouped['count']
    )

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r'$Q$')
    ax1.set_ylabel(r'$I(Q)$')
    ax2.set_ylabel('Frequency')

    ax2.hist(df['MetaVolume'], bins=bins, color='lightgrey', alpha=0.6)
    ax1.errorbar(
        grouped['x'], grouped['y'],
        xerr=grouped['x_err'], yerr=grouped['y_err'],
        marker='o', linestyle='', color='C0', label='Binned data'
    )

    if fit:
        Y, delta, Y_err, delta_err = fit
        x_line = np.logspace(
            np.log10(grouped['x'].min()),
            np.log10(grouped['x'].max()), 100
        )
        fit_label = rf'$Y={Y:.2e} \pm {Y_err:.2e},\ \delta={delta:.3f} \pm {delta_err:.3f}$'
        ax1.plot(x_line, power_law(x_line, Y, delta), 'k--', label=fit_label)
        print(f"Aggregate Fit: Y={Y:.4e} ± {Y_err:.4e}, delta={delta:.4f} ± {delta_err:.4f}")

    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, which='major', linewidth=1.0, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'images\\{image_name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_comparison(file_name_con_estensione, models, image_name,
                              n_bins=51, vertical_shift=10.0):
    """
    Comparison plot: datasets sorted by fitted delta (ascending), then stacked
    vertically with a multiplicative shift so slopes can be compared without
    overlap. Top of the ladder = lowest delta, bottom = highest delta.

    Parameters
    ----------
    file_name_con_estensione : str
        CSV filename (e.g. '20_power_2.0.csv').
    models : list[str]
        Models to compare. Use '' for real data (no model prefix).
    image_name : str
        Output image path (without 'images\\' prefix or extension).
    n_bins : int
        Number of log-spaced bins.
    vertical_shift : float
        Multiplicative spacing between successive datasets in log space.
        Each dataset i is multiplied by vertical_shift^(n-1-i) so that
        rank 0 (lowest delta) sits highest on the plot.
    """
    # ── Pass 1: load, bin, fit every model ───────────────────────────────────
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

        fit = robust_power_law_fit(
            grouped['x'], grouped['y'],
            grouped['x_err'], grouped['y_err'],
            grouped['count']
        )
        if fit is None:
            print(f"[{label_base}] Fit failed, skipping.")
            continue

        records.append(dict(model=model, label=label_base, grouped=grouped, fit=fit))

    if not records:
        print("[ERROR] No models could be fitted; aborting comparison plot.")
        return

    # ── Sort ascending by delta ───────────────────────────────────────────────
    records.sort(key=lambda r: r['fit'][1])  # fit[1] == delta
    n = len(records)

    # ── Assign colours in sorted order (tab10) ───────────────────────────────
    palette = cm.tab10(np.linspace(0, 1, max(n, 1)))

    # ── Pass 2: plot ─────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r'$Q$')
    ax1.set_ylabel(r'$I(Q)$ (shifted)')
    ax1.set_xlim([1e-6, 1e-2])

    print(f"\n{'Rank':>5}  {'Model':<20}  {'delta':>7}  {'shift':>8}")
    print("-" * 50)

    for rank, (rec, color) in enumerate(zip(records, palette)):
        # rank 0 (lowest delta) → top of plot → largest shift exponent
        shift_exp = n - 1 - rank
        shift = vertical_shift ** shift_exp
        grouped = rec['grouped']
        Y, delta, Y_err, delta_err = rec['fit']
        label_base = rec['label']

        print(f"{rank:>5}  {label_base:<20}  {delta:>7.3f}  ×{shift:>7.2f}")

        # Shifted scatter
        ax1.plot(
            grouped['x'], grouped['y'] * shift,
            marker='o', linestyle='', color=color, alpha=0.7, markersize=4
        )

        # Shifted fit line clipped to model's own x-range
        x_line = np.logspace(
            np.log10(grouped['x'].min()),
            np.log10(grouped['x'].max()), 200
        )
        if shift_exp != 0:
            shift_str = rf" ($\times {vertical_shift:.0f}^{{{shift_exp}}}$)"
        else:
            shift_str = ""
        fit_label = rf"{label_base}: $\delta={delta:.3f} \pm {delta_err:.3f}$" + shift_str
        ax1.plot(
            x_line, power_law(x_line, Y * shift, delta),
            linestyle='--', linewidth=1.8, color=color,
            label=fit_label, solid_capstyle='butt'
        )

    print("-" * 50)

    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, which='both', linewidth=1.0, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'images\\{image_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: images\\{image_name}.png")

def load_model_data(model, file_name_con_estensione, min_child=2):
    """Load and clean data for a given model ('' means real data)."""
    folder_prefix = f"meta_{model}" if model else "meta"
    path = f"database\\{folder_prefix}\\meta_{file_name_con_estensione}"
    data = pd.read_csv(path)
    return data[data['NbChild'] >= min_child].copy()


if __name__ == "__main__":
    # ── Configurazione File ──────────────────────────────────────────────────
    file_name_con_estensione = '20_power_2.0.csv'
    function_clean = file_name_con_estensione.replace('.csv', '')

    # Modelli da analizzare
    models = ['', 'ar_1000', 'var_1000', 'delta_0.2_1000', 'delta_0.5_1000', 'delta_0.8_1000', 'lmf_tim_sqrt', 'lmf_tim_lin']

    # Conserviamo il riferimento originale della funzione prima di applicare modifiche dinamiche
    _orig_load_model_data = load_model_data

    target_dir = "impact_volume_curve_analysis_ge3"
    min_child = 3

    print("\n" + "="*70)
    print(f"AVVIO ANALISI: {target_dir.upper()} (NbChild >= {min_child})")
    print("="*70)

    # 1. Crea la cartella se non esiste
    os.makedirs(os.path.join("images", target_dir), exist_ok=True)

    # 2. Sovrascriviamo temporaneamente il comportamento di default della funzione di load
    load_model_data = lambda m, f: _orig_load_model_data(m, f, min_child=min_child)

    # 3. Generazione del grafico di confronto (Comparison Plot)
    img_comparison = f"{target_dir}/{function_clean}_comparison"
    print(f"\n[GENERAZIONE] Grafico di confronto complessivo: images/{img_comparison}.png")
    plot_aggregate_comparison(
        file_name_con_estensione,
        models=models,
        image_name=img_comparison,
        vertical_shift=10.0,
    )

    # 4. Generazione dei singoli grafici per modello
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

    # Ripristina la funzione originale al termine dello script per sicurezza
    load_model_data = _orig_load_model_data
    print("\n[FINISH] Tutte le analisi sono state completate con successo.")