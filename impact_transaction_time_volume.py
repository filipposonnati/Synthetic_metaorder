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

def linear(x, a, b):
    return a * x + b

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
        # Kept the original columns and loaded TradedVolume explicitly
        daily = trades[['PartialVolume', 'NbChild', 'PartialImpact', 'MetaVolume', 'TradedVolume']].copy()
        trades_total = pd.concat([trades_total, daily])

    trades_total = trades_total[trades_total['NbChild'] > 1]
    return trades_total

# ──────────────────────────────────────────────────────────────────────────────
# Fitting helpers
# ──────────────────────────────────────────────────────────────────────────────

def fit_iterative(func, x, y, y_err, x_err=None, n_iter=10, x_err_gradient=None):
    """Run an iterative curve_fit that propagates x-errors into an effective sigma."""
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
    """Fit I_i ~ Y * (TradedVolume)^delta and save the cumulative impact plot."""
    print('\n── Cumulative fit (X = TradedVolume) ──')

    # X-axis switched to TradedVolume
    bins = bin_log(trades['TradedVolume'])
    grouped = group_by_bins(trades, 'TradedVolume', 'PartialImpact', bins)

    x      = grouped['x_mean'].to_numpy()
    y      = grouped['y_mean'].to_numpy()
    x_err  = grouped['x_err'].to_numpy()
    y_err  = grouped['y_err'].to_numpy()

    core_mask = grouped['count'] > 0.5 * grouped['count'].max()
    x_c, y_c, x_err_c, y_err_c = x[core_mask], y[core_mask], x_err[core_mask], y_err[core_mask]

    popt0, _, perr0 = fit_iterative(power_law, x_c, y_c, y_err_c)
    report_fit('no err', popt0, perr0, ['Y', 'delta'])

    popt1, _, perr1 = fit_iterative(power_law, x_c, y_c, y_err_c)
    report_fit('y err', popt1, perr1, ['Y', 'delta'])

    def grad(popt, x):
        return popt[0] * popt[1] * x**(popt[1] - 1)

    popt, _, perr = fit_iterative(power_law, x_c, y_c, y_err_c, x_err_c, x_err_gradient=grad)
    report_fit('eff err', popt, perr, ['Y', 'delta'])

    Y, delta, Y_err, delta_err = popt[0], popt[1], perr[0], perr[1]

    # ── Plot ──
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.errorbar(x, y, xerr=x_err, yerr=y_err,
                 linestyle='', marker='o', capsize=3, label='Binned data', zorder=3)

    x_th = np.logspace(np.log10(x.min() if x.min() > 0 else 1e-5), np.log10(x.max()), 100)
    ax1.plot(x_th, Y * x_th**delta,
             label=r'$y = Yx^{\delta}$', linestyle=':', color='black', zorder=2)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('TradedVolume')
    ax1.set_ylabel(r'$I_i$')

    ax2 = ax1.twinx()
    ax2.hist(trades['TradedVolume'], bins=bins, alpha=0.2, color='gray', zorder=1)
    ax2.set_ylabel('Frequency')

    lines, labels   = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    ax1.grid(True, which='both', ls='-', alpha=0.3)

    plt.tight_layout()
    _save(f'{function}_cumulate', model)

    return Y, delta, Y_err, delta_err


def analyse_linear(trades: pd.DataFrame, function: str, model: str):
    """Fit I_i/√Q ~ a·TradedVolume + b and save the ratio plot."""
    print('\n── Linear fit (X = TradedVolume) ──')

    df = trades.copy()
    df['Ratio']            = df['PartialImpact'] / np.sqrt(df['MetaVolume'])
    df['NormalizedVolume'] = df['TradedVolume']  # TradedVolume is used directly here

    bins    = np.linspace(0, 1, 21)
    grouped = group_by_bins(df, 'NormalizedVolume', 'Ratio', bins)

    x     = grouped['x_mean'].to_numpy()
    y     = grouped['y_mean'].to_numpy()
    x_err = grouped['x_err'].to_numpy()
    y_err = grouped['y_err'].to_numpy()

    popt0, _, perr0 = fit_iterative(linear, x, y, y_err)
    report_fit('no err', popt0, perr0, ['a', 'b'])

    popt1, _, perr1 = fit_iterative(linear, x, y, y_err)
    report_fit('y err', popt1, perr1, ['a', 'b'])

    def grad(popt, x):
        return np.full_like(x, popt[0])

    popt, _, perr = fit_iterative(linear, x, y, y_err, x_err, x_err_gradient=grad)
    report_fit('eff err', popt, perr, ['a', 'b'])

    a, b, a_err, b_err = popt[0], popt[1], perr[0], perr[1]

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x, y, xerr=x_err, yerr=y_err,
                linestyle='', marker='o', capsize=3, label='Binned data', zorder=3)

    x_th = np.linspace(0.0, 1.0, 50)
    ax.plot(x_th, linear(x_th, a, b),
            label=r'$y = ax + b$', linestyle=':', color='black', zorder=2)

    ax.set_xlabel('TradedVolume')
    ax.set_ylabel(r'$I_i / \sqrt{Q}$')
    ax.legend()
    ax.grid(True, which='both', ls='-', alpha=0.3)

    plt.tight_layout()
    _save(f'{function}_ratio', model)

    return a, b, a_err, b_err


def analyse_flat(trades: pd.DataFrame, function: str, model: str):
    """Fit I_i/√(Σq_i) ~ Y (constant) plotted against TradedVolume."""
    print('\n── Flat fit ──')

    df = trades.copy()
    df['Ratio']            = df['PartialImpact'] / np.sqrt(df['PartialVolume'])
    df['NormalizedVolume'] = df['TradedVolume']  # TradedVolume is used directly here

    bins    = np.linspace(0, 1, 21)
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

    ax.set_xlabel('TradedVolume')
    ax.set_ylabel(r'$I_i / \sqrt{\Sigma q_i}$')
    ax.legend()
    ax.grid(True, which='both', ls='-', alpha=0.3)

    plt.tight_layout()
    _save(f'{function}_flat', model)

    return Y, Y_err


def analyse_by_nb_child(trades: pd.DataFrame, function: str, model: str,
                        child_range=range(2, 12)):
    """Plot I_i/√Q vs TradedVolume for each NbChild value."""
    print('\n── NbChild Analysis ──')

    colors     = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(11, 7))

    for idx, n_child in enumerate(child_range):
        subset = trades[trades['NbChild'] == n_child].copy()

        if len(subset) < 30:
            print(f'  Skipping NbChild={n_child}: insufficient data ({len(subset)} samples)')
            continue

        subset['Ratio']            = subset['PartialImpact'] / np.sqrt(subset['MetaVolume'])
        subset['NormalizedVolume'] = subset['TradedVolume']  # TradedVolume is used directly here

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

    ax.set_xlabel('TradedVolume')
    ax.set_ylabel(r'$I_i / \sqrt{Q}$')
    ax.legend(title='n children', fontsize=8, title_fontsize=9,
              bbox_to_anchor=(1, 1), loc='upper left')
    ax.grid(True, ls='--', alpha=0.3)

    plt.tight_layout()
    _save(f'{function}_ratio_child', model, dpi=150, bbox_inches='tight')


def analyse_intra_order_trend(trades: pd.DataFrame, function: str, model: str,
                              min_child_thresholds=None, plot_dotted_lines=None, exponent: float = 0.5):
    """
    Plot I_i/Q^exponent vs TradedVolume for multiple NbChild thresholds.
    
    Parameters
    ----------
    trades : pd.DataFrame
        Trade data
    function : str
        Function name for saving
    model : str
        Model name
    min_child_thresholds : list, optional
        List of minimum child thresholds (default: [2, 5, 10])
    plot_dotted_lines : list of bool, optional
        Boolean mask corresponding to min_child_thresholds.
        If True, plot the dotted fit line; if False, omit it.
    exponent : float
        Exponent for Q normalization
    """
    if min_child_thresholds is None:
        min_child_thresholds = [2, 5, 10]

    if plot_dotted_lines is None:
        plot_dotted_lines = [True] * len(min_child_thresholds)
    
    print('\n── Intra-Order Trend (Multiple Thresholds) ──')
    
    # Define colors for each threshold
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    if len(min_child_thresholds) > len(colors):
        colors = plt.cm.tab10(np.linspace(0, 1, len(min_child_thresholds)))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for threshold_idx, min_child in enumerate(min_child_thresholds):
        subset = trades[trades['NbChild'] >= min_child].copy()
        
        if len(subset) < 30:
            print(f'  Skipping threshold NbChild ≥ {min_child}: insufficient data ({len(subset)} samples)')
            continue
        
        print(f'  Threshold NbChild ≥ {min_child}: {len(subset):,} samples')
        
        subset['Ratio']           = subset['PartialImpact'] / subset['MetaVolume']**exponent
        subset['NormalizedVolume'] = subset['TradedVolume']
        
        bins    = np.linspace(0, 1, 21)
        grouped = group_by_bins(subset, 'NormalizedVolume', 'Ratio', bins)
        
        x     = grouped['x_mean'].to_numpy()
        y     = grouped['y_mean'].to_numpy()
        x_err = grouped['x_err'].to_numpy()
        y_err = grouped['y_err'].to_numpy()
        
        # Plot binned data with error bars connected by a solid line
        ax.errorbar(
            x, y,
            xerr=x_err, yerr=y_err,
            color=colors[threshold_idx],
            linestyle='-',
            marker='o',
            capsize=3,
            markersize=6,
            alpha=0.85,
            elinewidth=1.5,
            capthick=1.5,
            label=f'n ≥ {min_child}',
            zorder=3
        )
        
        # Fit linear model
        def grad(popt, x):
            return np.full_like(x, popt[0])
        
        popt, _, perr = fit_iterative(linear, x, y, y_err, x_err,
                                      x_err_gradient=grad)
        
        report_fit(f'n ≥ {min_child}', popt, perr, ['a', 'b'])
        
        # Determine whether to draw the dotted fit line
        should_plot_dotted = (
            plot_dotted_lines[threshold_idx] 
            if threshold_idx < len(plot_dotted_lines) 
            else True
        )

        # Plot fit line (dotted) in the same color if enabled
        if should_plot_dotted:
            x_th = np.linspace(0.0, 1.0, 100)
            ax.plot(x_th, linear(x_th, popt[0], popt[1]),
                    color=colors[threshold_idx],
                    linestyle=':',
                    linewidth=2.5,
                    alpha=0.9,
                    zorder=2)
    
    ax.set_xlabel('Traded Volume', fontsize=14)
    ax.set_ylabel(rf'$I_i / Q^{{{exponent:g}}}$', fontsize=14)
    ax.legend(loc='best', fontsize=12, framealpha=0.95)
    ax.grid(True, which='both', ls='-', alpha=0.3)
    
    plt.tight_layout()
    _save(f'{function}_impact_transaction_time_volume', model)


# ──────────────────────────────────────────────────────────────────────────────
# I/O helper
# ──────────────────────────────────────────────────────────────────────────────

def _save(name: str, model: str, **kwargs):
    prefix = f'{model}_' if model else ''
    # New directory path destination applied here:
    path   = f'images\\impact_transaction_time_volume\\{prefix}{name}.png'
    plt.savefig(path, **kwargs)
    plt.close()
    print(f'  Saved → {path}')


if __name__ == '__main__':
    function = '20_power_2.0'
    model    = ''
    exponent = 0.5  # set the exponent of Q used in I ~ Q^exponent (0.5 = square root)
    
    # Configure minimum child thresholds and their dotted line display mask
    min_child_thresholds = [2, 5, 10]
    plot_dotted_lines    = [False, False, True]  # Toggle dotted lines per threshold index

    print(f'{function}  {model}  (exponent = {exponent})')
    print(f'Min child thresholds: {min_child_thresholds}')
    print(f'Plot dotted lines:    {plot_dotted_lines}\n')

    trades = load_trades(function, model)

    # Multi-threshold intra-order trend
    analyse_intra_order_trend(trades, function, model, 
                              min_child_thresholds=min_child_thresholds,
                              plot_dotted_lines=plot_dotted_lines,
                              exponent=exponent)