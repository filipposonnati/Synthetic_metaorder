import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
})

# ---------------------------------------------------------------------------
# Fit functions
# ---------------------------------------------------------------------------

def power_law(x, Y, delta):
    return Y * x**delta

def square_root(x, Y):
    return Y * np.sqrt(x)

def constant(x, Y):
    return Y

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def build_directory(function: str, model: str) -> str:
    prefix = f"trades_{model}_" if model else "trades_"
    return f"database\\{prefix}{function}"


def load_trades(directory: str) -> pd.DataFrame:
    """Load and concatenate all CSV files found in *directory*."""
    paths = listdir(directory)
    frames = []
    for path in paths:
        df = pd.read_csv(
            f"{directory}\\{path}",
            sep=',',
            usecols=['PartialVolume', 'NbChild', 'PartialImpact', 'MetaVolume', 'TradedVolume'],
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def filter_and_engineer(trades: pd.DataFrame) -> pd.DataFrame:
    """Keep only meta-orders with more than one child and add derived columns."""
    df = trades[trades['NbChild'] > 1].copy()
    df['Ratio'] = df['PartialImpact'] / np.sqrt(df['MetaVolume'])
    df['NormalizedVolume'] = df['TradedVolume']
    return df


def bin_and_aggregate(df: pd.DataFrame, bins: np.ndarray) -> pd.DataFrame:
    """
    Bin *NormalizedVolume* and aggregate mean / std / count per bin.

    Points with NormalizedVolume == 1 are excluded from the binning process;
    their Ratio is averaged separately and appended as the final row.
    """
    df = df.copy()

    mask_at_one = df['NormalizedVolume'] == 1.0
    df_binned   = df[~mask_at_one]
    df_at_one   = df[mask_at_one]

    df_binned = df_binned.copy()
    df_binned['bin'] = pd.cut(df_binned['NormalizedVolume'], bins=bins, include_lowest=True)
    grouped = df_binned.groupby('bin', observed=True).agg(
        NormalizedVolume_mean=('NormalizedVolume', 'mean'),
        NormalizedVolume_std=('NormalizedVolume', 'std'),
        Ratio_mean=('Ratio', 'mean'),
        Ratio_std=('Ratio', 'std'),
        count=('Ratio', 'count'),
    ).dropna()

    if not df_at_one.empty:
        row_at_one = pd.DataFrame([{
            'NormalizedVolume_mean': 1.0,
            'NormalizedVolume_std':  df_at_one['NormalizedVolume'].std(),
            'Ratio_mean':            df_at_one['Ratio'].mean(),
            'Ratio_std':             df_at_one['Ratio'].std(),
            'count':                 len(df_at_one),
        }])
        grouped = pd.concat([grouped, row_at_one], ignore_index=True)

    return grouped

# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def fit_square_root(x: np.ndarray, y: np.ndarray,
                    y_err: np.ndarray, x_err: np.ndarray,
                    n_iter: int = 10) -> tuple[float, float]:
    """
    Fit  y = Y * sqrt(x)  with iteratively refined effective errors.

    Returns
    -------
    Y, Y_err
    """
    # No-error fit
    popt, pcov = curve_fit(square_root, x, y)
    Y = popt[0]
    print(f'Fit (no err):  Y = {Y:.6f} +- {np.sqrt(pcov[0, 0]):.6f}')

    # y-error only
    popt, pcov = curve_fit(square_root, x, y, sigma=y_err, absolute_sigma=True)
    Y = popt[0]
    print(f'Fit (y err):   Y = {Y:.6f} +- {np.sqrt(pcov[0, 0]):.6f}')

    # Effective error (propagated x uncertainty iterated until convergence)
    for _ in range(n_iter):
        eff_err = np.sqrt(y_err**2 + (Y * 0.5 * x**(-0.5) * x_err)**2)
        popt, pcov = curve_fit(square_root, x, y, sigma=eff_err, absolute_sigma=True)
        Y = popt[0]

    Y_err = np.sqrt(pcov[0, 0])
    print(f'Fit (eff err): Y = {Y:.6f} +- {Y_err:.6f}')
    return Y, Y_err

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_impact_vs_volume(x: np.ndarray, y: np.ndarray,
                          Y: float, function: str, model: str) -> None:
    """Plot binned impact ratio against normalised volume with a sqrt fit."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, y, linestyle='', marker='o', label='Binned data', zorder=3)

    x_th = np.linspace(0.0, 1.0, 50)
    ax.plot(x_th, Y * np.sqrt(x_th),
            label=r'$y = Y\sqrt{x}$', linestyle=':', color='black', zorder=2)

    ax.set_xlabel(r'$t_q$')
    ax.set_ylabel(r'$I_i / \sqrt{Q}$')
    ax.legend()
    ax.grid(True, which='both', ls='-', alpha=0.3)

    fig.tight_layout()
    out = f'images\\impact_time_volume\\{model + "_" if model else ""}{function}.png'
    fig.savefig(out)
    plt.close(fig)


def plot_impact_with_histogram(trades: pd.DataFrame, function: str, model: str,
                               bins: np.ndarray, n_child: int = 2,
                               min_samples: int = 30) -> None:
    """
    For a given NbChild value, plot the binned impact ratio curve (left y-axis)
    overlaid on a frequency histogram of NormalizedVolume (right y-axis, shaded).
    """
    subset = trades[trades['NbChild'] == n_child].copy()

    if len(subset) < min_samples:
        print(f"plot_impact_with_histogram: insufficient data for NbChild={n_child} ({len(subset)} samples)")
        return

    # Bin & aggregate impact ratio (x == 1 treated separately)
    mask_at_one  = subset['NormalizedVolume'] == 1.0
    sub_binned   = subset[~mask_at_one].copy()
    sub_at_one   = subset[mask_at_one]

    sub_binned['bin'] = pd.cut(sub_binned['NormalizedVolume'], bins=bins, include_lowest=True)
    grp = sub_binned.groupby('bin', observed=True).agg(
        x=('NormalizedVolume', 'mean'),
        y=('Ratio', 'mean'),
        y_std=('Ratio', 'std'),
        count=('Ratio', 'count'),
    ).dropna()

    if not sub_at_one.empty:
        row_at_one = pd.DataFrame([{
            'x':     1.0,
            'y':     sub_at_one['Ratio'].mean(),
            'y_std': sub_at_one['Ratio'].std(),
            'count': len(sub_at_one),
        }])
        grp = pd.concat([grp, row_at_one], ignore_index=True)

    x_val   = grp['x'].to_numpy()
    y_val   = grp['y'].to_numpy()
    y_err   = (grp['y_std'] / np.sqrt(grp['count'])).to_numpy()

    # Bin counts for the histogram (same binning scheme)
    counts, _ = np.histogram(sub_binned['NormalizedVolume'].dropna(), bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width   = bins[1] - bins[0]

    # ---- figure with twin axes ------------------------------------------------
    fig, ax_impact = plt.subplots(figsize=(10, 6))
    ax_hist = ax_impact.twinx()

    # Histogram on the background axis (draw first so impact sits on top)
    ax_hist.bar(
        bin_centers, counts,
        width=bin_width * 0.9,
        color='steelblue', alpha=0.25,
        label='Frequency',
        zorder=1,
    )
    ax_hist.set_ylabel('Frequency', color='steelblue')
    ax_hist.tick_params(axis='y', labelcolor='steelblue')
    ax_hist.yaxis.set_label_position('right')

    # Impact ratio curve with error bars on the foreground axis
    ax_impact.errorbar(
        x_val, y_val, yerr=y_err,
        linestyle='-', marker='o',
        color='tab:orange', capsize=3, linewidth=1.8,
        label=r'$I_i / \sqrt{Q}$  (n=%d)' % n_child,
        zorder=3,
    )
    ax_impact.set_xlabel(r'$t_q$')
    ax_impact.set_ylabel(r'$I_i / \sqrt{Q}$', color='tab:orange')
    ax_impact.tick_params(axis='y', labelcolor='tab:orange')
    ax_impact.set_xlim(bins[0], bins[-1])

    # Combined legend
    lines_impact, labels_impact = ax_impact.get_legend_handles_labels()
    lines_hist,   labels_hist   = ax_hist.get_legend_handles_labels()
    ax_impact.legend(lines_impact + lines_hist, labels_impact + labels_hist,
                     loc='upper left')

    ax_impact.grid(True, ls='--', alpha=0.3, zorder=0)
    ax_impact.set_title(f'Impact ratio & volume distribution  (NbChild = {n_child})')

    fig.tight_layout()
    out = (f'images\\impact_time_volume\\'
           f'{model + "_" if model else ""}{function}_impact_hist_n{n_child}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()


def plot_ratio_by_nb_child(trades: pd.DataFrame, function: str, model: str,
                            child_range: range, bins: np.ndarray,
                            min_samples: int = 30) -> None:
    """Plot the impact ratio curve for each value of NbChild."""
    colors     = plt.cm.tab10(np.linspace(0, 1, 10))
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

    fig, ax = plt.subplots(figsize=(11, 7))

    for idx, n_child in enumerate(child_range):
        subset = trades[trades['NbChild'] == n_child].copy()

        if len(subset) < min_samples:
            print(f"Skipping NbChild={n_child}: insufficient data ({len(subset)} samples)")
            continue

        subset['Ratio']            = subset['PartialImpact'] / np.sqrt(subset['MetaVolume'])
        subset['NormalizedVolume']  = subset['TradedVolume']

        mask_at_one   = subset['NormalizedVolume'] == 1.0
        sub_binned    = subset[~mask_at_one].copy()
        sub_at_one    = subset[mask_at_one]

        sub_binned['bin'] = pd.cut(sub_binned['NormalizedVolume'], bins=bins, include_lowest=True)
        grp = sub_binned.groupby('bin', observed=True).agg(
            x=('NormalizedVolume', 'mean'),
            y=('Ratio', 'mean'),
            y_std=('Ratio', 'std'),
            count=('Ratio', 'count'),
        ).dropna()

        if not sub_at_one.empty:
            row_at_one = pd.DataFrame([{
                'x':     1.0,
                'y':     sub_at_one['Ratio'].mean(),
                'y_std': sub_at_one['Ratio'].std(),
                'count': len(sub_at_one),
            }])
            grp = pd.concat([grp, row_at_one], ignore_index=True)

        ax.plot(
            grp['x'], grp['y'],
            color=colors[idx],
            linestyle=linestyles[idx],
            linewidth=1.5,
            alpha=0.85,
            label=f'n={n_child}',
        )

    ax.set_xlabel(r'$t_q$')
    ax.set_ylabel(r'$I_i / \sqrt{Q}$')
    ax.legend(title='n children', fontsize=8, title_fontsize=9,
              bbox_to_anchor=(1, 1), loc='upper left')
    ax.grid(True, ls='--', alpha=0.3)

    fig.tight_layout()
    out = (f'images\\impact_time_volume\\'
           f'{model + "_" if model else ""}{function}_ratio_child.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    function = '20_power_2.0'
    model    = 'var_1000'

    print(f"{function} {model}")

    # Load data
    directory = build_directory(function, model)
    print(directory)
    trades_raw = load_trades(directory)

    # Prepare
    trades = filter_and_engineer(trades_raw)

    # Bin & aggregate
    bins    = np.linspace(0, 1, 21)
    grouped = bin_and_aggregate(trades, bins)

    x     = grouped['NormalizedVolume_mean'].to_numpy()
    y     = grouped['Ratio_mean'].to_numpy()
    x_err = grouped['NormalizedVolume_std'].to_numpy() / np.sqrt(grouped['count'].to_numpy())
    y_err = grouped['Ratio_std'].to_numpy()             / np.sqrt(grouped['count'].to_numpy())

    # Fit
    Y, Y_err = fit_square_root(x, y, y_err, x_err)

    # Plots
    plot_impact_vs_volume(x, y, Y, function, model)

    print('NbChild Analysis')
    plot_ratio_by_nb_child(trades, function, model,
                           child_range=range(2, 12), bins=bins)

    print('Impact + Histogram (n=2)')
    plot_impact_with_histogram(trades, function, model, bins=bins, n_child=2)