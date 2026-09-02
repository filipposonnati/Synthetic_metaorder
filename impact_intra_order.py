import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from scipy.optimize import curve_fit

# ──────────────────────────────────────────────────────────────────────────────
# Plot defaults
# ──────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
})

# ──────────────────────────────────────────────────────────────────────────────
# Model functions
# ──────────────────────────────────────────────────────────────────────────────

def power_law(x, Y, delta):
    return Y * x**delta

def linear(x, Y, C):
    return Y * x + C

def constant(x, Y):
    return Y

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_trades(function: str, model: str) -> pd.DataFrame:
    """Load and concatenate all CSV trade files for the given function/model."""
    dir_path = f'database\\{f"trades_{model}_" if model else "trades_"}{function}'
    print(f'Loading data from: {dir_path}')

    trades_total = None
    for path in listdir(dir_path):
        trades = pd.read_csv(f'{dir_path}\\{path}', sep=',')
        daily = trades[['PartialVolume', 'NbChild', 'PartialImpact', 'MetaVolume']].copy()
        trades_total = pd.concat([trades_total, daily])

    trades_total = trades_total[trades_total['NbChild'] > 1]
    return trades_total

# ──────────────────────────────────────────────────────────────────────────────
# Fitting helpers
# ──────────────────────────────────────────────────────────────────────────────

def fit_iterative(func, x, y, y_err, x_err=None, n_iter=10, x_err_gradient=None):
    """
    Run an iterative curve_fit that propagates x-errors into an effective sigma.

    Parameters
    ----------
    func             : callable  – model function
    x, y             : arrays    – data
    y_err            : array     – uncertainty on y
    x_err            : array     – uncertainty on x (optional)
    n_iter           : int       – number of refinement iterations
    x_err_gradient   : callable  – function(popt, x) → |∂f/∂x| for error propagation
                                   Required when x_err is provided.

    Returns
    -------
    popt, pcov, popt_err
    """
    # Initial fit ignoring x errors
    popt, pcov = curve_fit(func, x, y, sigma=y_err, absolute_sigma=True)

    if x_err is None or x_err_gradient is None:
        return popt, pcov, np.sqrt(np.diag(pcov))

    for _ in range(n_iter):
        eff_err = np.sqrt(y_err**2 + (x_err_gradient(popt, x) * x_err)**2)
        popt, pcov = curve_fit(func, x, y, sigma=eff_err, absolute_sigma=True)

    return popt, pcov, np.sqrt(np.diag(pcov))


def report_fit(label: str, popt, popt_err, param_names):
    parts = ', '.join(f'{n} = {v:.6g} ± {e:.6g}'
                      for n, v, e in zip(param_names, popt, popt_err))
    print(f'Fit ({label}): {parts}')

# ──────────────────────────────────────────────────────────────────────────────
# Binning helpers
# ──────────────────────────────────────────────────────────────────────────────

def bin_log(series: pd.Series, n_bins: int = 51):
    return np.logspace(np.log10(series.min()), np.log10(series.max()), n_bins)

def bin_linear(n_bins: int = 26):
    return np.linspace(0, 1, n_bins)

def group_by_bins(df, col_x, col_y, bins):
    """Bin col_x and return grouped statistics for (col_x, col_y)."""
    df = df.copy()
    df['bin'] = pd.cut(df[col_x], bins=bins, include_lowest=True)
    grouped = df.groupby('bin', observed=True).agg({
        col_x: ['mean', 'std', 'count'],
        col_y: ['mean', 'std'],
    }).dropna()
    grouped.columns = ['x_mean', 'x_std', 'count', 'y_mean', 'y_std']
    n = grouped['count'].to_numpy()
    grouped['x_err'] = grouped['x_std'] / np.sqrt(n)
    grouped['y_err'] = grouped['y_std'] / np.sqrt(n)
    return grouped

# ──────────────────────────────────────────────────────────────────────────────
# Analysis sections
# ──────────────────────────────────────────────────────────────────────────────

def analyse_cumulative(trades: pd.DataFrame, function: str, model: str):
    """Fit I_i ~ Y * (Σq_i)^delta and save the cumulative impact plot."""
    print('\n── Cumulative fit ──')

    bins = bin_log(trades['PartialVolume'])
    grouped = group_by_bins(trades, 'PartialVolume', 'PartialImpact', bins)

    x      = grouped['x_mean'].to_numpy()
    y      = grouped['y_mean'].to_numpy()
    x_err  = grouped['x_err'].to_numpy()
    y_err  = grouped['y_err'].to_numpy()

    # Restrict fit to well-populated bins
    core_mask = grouped['count'] > 0.5 * grouped['count'].max()
    x_c, y_c, x_err_c, y_err_c = x[core_mask], y[core_mask], x_err[core_mask], y_err[core_mask]

    # No-error fit
    popt0, _, perr0 = fit_iterative(power_law, x_c, y_c, y_err_c)
    report_fit('no err', popt0, perr0, ['Y', 'delta'])

    # y-error only
    popt1, _, perr1 = fit_iterative(power_law, x_c, y_c, y_err_c)
    report_fit('y err', popt1, perr1, ['Y', 'delta'])

    # Effective (x+y) error – propagate ∂(power_law)/∂x = Y·δ·x^(δ−1)
    def grad(popt, x):
        return popt[0] * popt[1] * x**(popt[1] - 1)

    popt, _, perr = fit_iterative(power_law, x_c, y_c, y_err_c, x_err_c,
                                  x_err_gradient=grad)
    report_fit('eff err', popt, perr, ['Y', 'delta'])

    Y, delta, Y_err, delta_err = popt[0], popt[1], perr[0], perr[1]

    # ── Plot ──
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.errorbar(x, y, xerr=x_err, yerr=y_err,
                 linestyle='', marker='o', capsize=3, label='Binned data', zorder=3)

    x_th = np.logspace(np.log10(1e-7), np.log10(2e-3), 100)
    ax1.plot(x_th, Y * x_th**delta,
             label=r'$y = Yx^{\delta}$', linestyle=':', color='black', zorder=2)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\Sigma q_i$')
    ax1.set_ylabel(r'$I_i$')

    ax2 = ax1.twinx()
    ax2.hist(trades['PartialVolume'], bins=bins, alpha=0.2, color='gray', zorder=1)
    ax2.set_ylabel('Frequency')

    lines, labels   = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    ax1.grid(True, which='both', ls='-', alpha=0.3)

    plt.tight_layout()
    _save(f'{function}_cumulate', model)

    return Y, delta, Y_err, delta_err


def analyse_linear(trades: pd.DataFrame, function: str, model: str, exponent: float = 0.5):
    """Fit I_i/Q^exponent ~ Y·(Σq_i/Q) + C and save the ratio plot."""
    print('\n── Linear fit ──')

    df = trades.copy()
    df['Ratio']           = df['PartialImpact'] / df['MetaVolume']**exponent
    df['NormalizedVolume'] = df['PartialVolume']  / df['MetaVolume']

    bins    = bin_linear()
    grouped = group_by_bins(df, 'NormalizedVolume', 'Ratio', bins)

    x     = grouped['x_mean'].to_numpy()
    y     = grouped['y_mean'].to_numpy()
    x_err = grouped['x_err'].to_numpy()
    y_err = grouped['y_err'].to_numpy()

    # No-error fit
    popt0, _, perr0 = fit_iterative(linear, x, y, y_err)
    report_fit('no err', popt0, perr0, ['Y', 'C'])

    # y-error only
    popt1, _, perr1 = fit_iterative(linear, x, y, y_err)
    report_fit('y err', popt1, perr1, ['Y', 'C'])

    # Effective error – ∂(Yx+C)/∂x = Y
    def grad(popt, x):
        return np.full_like(x, popt[0])

    popt, _, perr = fit_iterative(linear, x, y, y_err, x_err,
                                  x_err_gradient=grad)
    report_fit('eff err', popt, perr, ['Y', 'C'])

    Y, C, Y_err, C_err = popt[0], popt[1], perr[0], perr[1]

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x, y, xerr=x_err, yerr=y_err,
                linestyle='', marker='o', capsize=3, label='Binned data', zorder=3)

    x_th = np.linspace(0.0, 1.0, 50)
    ax.plot(x_th, Y * x_th + C,
            label=r'$y = Yx + C$', linestyle=':', color='black', zorder=2)

    ax.set_xlabel(r'$\Sigma q_i / Q$')
    ax.set_ylabel(rf'$I_i / Q^{{{exponent:g}}}$')
    ax.legend()
    ax.grid(True, which='both', ls='-', alpha=0.3)

    plt.tight_layout()
    _save(f'{function}_ratio', model)

    return Y, C, Y_err, C_err


def analyse_flat(trades: pd.DataFrame, function: str, model: str, exponent: float = 0.5):
    """Fit I_i/(Σq_i)^exponent ~ Y (constant) and save the flat plot."""
    print('\n── Flat fit ──')

    df = trades.copy()
    df['Ratio']           = df['PartialImpact'] / df['PartialVolume']**exponent
    df['NormalizedVolume'] = df['PartialVolume']  / df['MetaVolume']

    bins    = bin_linear()
    grouped = group_by_bins(df, 'NormalizedVolume', 'Ratio', bins)

    x     = grouped['x_mean'].to_numpy()
    y     = grouped['y_mean'].to_numpy()
    x_err = grouped['x_err'].to_numpy()
    y_err = grouped['y_err'].to_numpy()

    popt0, _, perr0 = fit_iterative(constant, x, y, y_err)
    report_fit('no err', popt0, perr0, ['Y'])

    popt, _, perr = fit_iterative(constant, x, y, y_err)
    report_fit('y err', popt, perr, ['Y'])

    Y, Y_err = popt[0], perr[0]

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x, y, xerr=x_err, yerr=y_err,
                linestyle='', marker='o', capsize=3, label='Binned data', zorder=3)

    x_th = np.linspace(0.0, 1.0, 100)
    ax.plot(x_th, np.full(len(x_th), Y),
            label=r'$y = Y$', linestyle=':', color='black', zorder=2)

    ax.set_xlabel(r'$\Sigma q_i / Q$')
    ax.set_ylabel(rf'$I_i / (\Sigma q_i)^{{{exponent:g}}}$')
    ax.legend()
    ax.grid(True, which='both', ls='-', alpha=0.3)

    plt.tight_layout()
    _save(f'{function}_flat', model)

    return Y, Y_err


def analyse_by_nb_child(trades: pd.DataFrame, function: str, model: str,
                        child_range=range(2, 12), exponent: float = 0.5):
    """Plot I_i/Q^exponent vs Σq_i/Q for each NbChild value."""
    print('\n── NbChild Analysis ──')

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(11, 7))

    for idx, n_child in enumerate(child_range):
        subset = trades[trades['NbChild'] == n_child].copy()

        if len(subset) < 30:
            print(f'  Skipping NbChild={n_child}: insufficient data ({len(subset)} samples)')
            continue

        subset['Ratio']           = subset['PartialImpact'] / subset['MetaVolume']**exponent
        subset['NormalizedVolume'] = subset['PartialVolume']  / subset['MetaVolume']

        bins    = np.linspace(0, 1, 21)
        grouped = group_by_bins(subset, 'NormalizedVolume', 'Ratio', bins)

        ax.errorbar(
            grouped['x_mean'].to_numpy(),
            grouped['y_mean'].to_numpy(),
            xerr=grouped['x_err'].to_numpy(),
            yerr=grouped['y_err'].to_numpy(),
            color=colors[idx],
            linestyle='-',
            linewidth=1.5,
            alpha=0.85,
            capsize=2,
            label=f'n={n_child}',
        )

    ax.set_xlabel(r'$\Sigma q_i / Q$')
    ax.set_ylabel(rf'$I_i / Q^{{{exponent:g}}}$')
    ax.legend(title='n children', fontsize=8, title_fontsize=9,
              bbox_to_anchor=(1, 1), loc='upper left')
    ax.grid(True, ls='--', alpha=0.3)

    plt.tight_layout()
    _save(f'{function}_ratio_child', model, dpi=150, bbox_inches='tight')

def analyse_linear_min_child(trades: pd.DataFrame, function: str, model: str,
                             min_child: int = 5, exponent: float = 0.5):
    """
    Same as analyse_linear but restricted to metaorders with NbChild >= min_child.
    Fits I_i/Q^exponent ~ Y·(Σq_i/Q) + C and saves the filtered ratio plot.
    """
    print(f'\n── Linear fit (NbChild ≥ {min_child}) ──')

    df = trades[trades['NbChild'] >= min_child].copy()
    print(f'  Rows after filter: {len(df):,}  (dropped {len(trades) - len(df):,})')

    if len(df) < 30:
        print('  Insufficient data after filter – skipping.')
        return None, None

    df['Ratio']            = df['PartialImpact'] / df['MetaVolume']**exponent
    df['NormalizedVolume'] = df['PartialVolume']  / df['MetaVolume']

    bins    = bin_linear()
    grouped = group_by_bins(df, 'NormalizedVolume', 'Ratio', bins)

    x     = grouped['x_mean'].to_numpy()
    y     = grouped['y_mean'].to_numpy()
    x_err = grouped['x_err'].to_numpy()
    y_err = grouped['y_err'].to_numpy()

    # No-error fit
    popt0, _, perr0 = fit_iterative(linear, x, y, y_err)
    report_fit('no err', popt0, perr0, ['Y', 'C'])

    # y-error only
    popt1, _, perr1 = fit_iterative(linear, x, y, y_err)
    report_fit('y err', popt1, perr1, ['Y', 'C'])

    # Effective error – ∂(Yx+C)/∂x = Y
    def grad(popt, x):
        return np.full_like(x, popt[0])

    popt, _, perr = fit_iterative(linear, x, y, y_err, x_err,
                                  x_err_gradient=grad)
    report_fit('eff err', popt, perr, ['Y', 'C'])

    Y, C, Y_err, C_err = popt[0], popt[1], perr[0], perr[1]

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x, y, xerr=x_err, yerr=y_err,
                linestyle='', marker='o', capsize=3, label='Binned data', zorder=3)

    x_th = np.linspace(0.0, 1.0, 50)
    ax.plot(x_th, Y * x_th + C,
            label=r'$y = Yx + C$', linestyle=':', color='black', zorder=2)

    ax.set_xlabel(r'$\Sigma q_i / Q$')
    ax.set_ylabel(rf'$I_i / Q^{{{exponent:g}}}$')
    #ax.set_title(rf'NbChild $\geq$ {min_child}')
    ax.legend()
    ax.grid(True, which='both', ls='-', alpha=0.3)

    plt.tight_layout()
    _save(f'{function}_ratio_min{min_child}child', model)

    return Y, C, Y_err, C_err


# ──────────────────────────────────────────────────────────────────────────────
# I/O helper
# ──────────────────────────────────────────────────────────────────────────────

def _save(name: str, model: str, **kwargs):
    prefix = f'{model}_' if model else ''
    path   = f'images\\impact_intra_order\\{prefix}{name}.png'
    plt.savefig(path, **kwargs)
    plt.close()
    print(f'  Saved → {path}')

if __name__ == '__main__':
    function = '20_power_2.0'
    model    = ''
    exponent = 0.5  # set the exponent of Q used in I ~ Q^exponent (0.5 = square root)
    print(f'{function}  {model}  (exponent = {exponent})')

    trades = load_trades(function, model)

    Y_lin, C_lin, Y_lin_err, C_lin_err = analyse_linear(trades, function, model, exponent=exponent)

    analyse_by_nb_child(trades, function, model, exponent=exponent)

    Y_lin5, C_lin5, Y_lin5_err, C_lin5_err = analyse_linear_min_child(trades, function, model, min_child=5, exponent=exponent)
    Y_lin10, C_lin10, Y_lin10_err, C_lin10_err = analyse_linear_min_child(trades, function, model, min_child=10, exponent=exponent)

    # ── Summary (ready for a LaTeX table) ──
    print('\n── Parameter summary ──')
    #print(f'  Cumulative       : Y = {Y_cum:.6g} ± {Y_cum_err:.6g},  δ = {delta_cum:.6g} ± {delta_cum_err:.6g}')
    print(f'  Linear           : Y = {Y_lin:.6g} ± {Y_lin_err:.6g},  C = {C_lin:.6g} ± {C_lin_err:.6g}')
    print(f'  Linear ≥5ch      : Y = {Y_lin5:.6g} ± {Y_lin5_err:.6g},  C = {C_lin5:.6g} ± {C_lin5_err:.6g}')
    print(f'  Linear ≥10ch     : Y = {Y_lin10:.6g} ± {Y_lin10_err:.6g},  C = {C_lin10:.6g} ± {C_lin10_err:.6g}')
    #print(f'  Flat             : Y = {Y_flat:.6g} ± {Y_flat_err:.6g}')