import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from os import listdir
import re

DIR = 'database\\meta'
PATH = 'meta_20_power_2.0.csv'
SAMPLE_THRESHOLD_PCT = 0.1

X_LABEL = 'V/V_D'
Y_LABEL = 'V/V_P'

def log_function_3_ab(log_x, log_Y, delta, alpha):
    """
    Linearized Power Law: log10(I) = log10(Y) + delta*log10(a) + alpha*log10(b)
    Equivalent to: I = Y * a^delta * b^alpha
    """
    log_a, log_b = log_x
    return log_Y + delta * log_a + alpha * log_b


def safe_err(std_col, count_col, mean_col, fallback_rel=1.0):
    """
    Standard error = std / sqrt(n).
    Assigns a fallback relative error for bins with zero or one sample
    to prevent division by zero during fitting.
    """
    err = std_col / np.sqrt(count_col)
    bad = (err == 0) | np.isnan(err)
    err[bad] = fallback_rel * mean_col[bad]
    return err

def compute_binned_stats(df, n_bins=31, sample_threshold_pct=SAMPLE_THRESHOLD_PCT):
    """
    Perform 2D logarithmic binning on columns 'a' and 'b', then compute
    per-bin means, standard errors (in linear and log space), and a
    significance mask.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'a', 'b', and 'MetaImpact'.
    n_bins : int
        Number of log-spaced bin edges along each axis.
    sample_threshold_pct : float
        Bins whose sample count falls below this fraction of the maximum
        bin count are flagged as insignificant.

    Returns
    -------
    grouped : pd.DataFrame
        Binned statistics with columns:
        a_mean, a_std, a_err, b_mean, b_std, b_err,
        impact_mean, impact_std, sample_count, err
    log_vals : dict
        Log10-transformed arrays:  log_a, log_b, log_i, zerr, xerr, yerr
        (computed over ALL significant bins).
    mask_significance : pd.Series (bool)
        True for bins that pass the sample-count threshold.
    """
    bins_a = np.logspace(np.log10(df['a'].min()), np.log10(df['a'].max()), n_bins)
    bins_b = np.logspace(np.log10(df['b'].min()), np.log10(df['b'].max()), n_bins)

    df = df.copy()
    df['bin_a'] = pd.cut(df['a'], bins=bins_a, include_lowest=True)
    df['bin_b'] = pd.cut(df['b'], bins=bins_b, include_lowest=True)

    grouped = df.groupby(['bin_a', 'bin_b'], observed=True).agg(
        a_mean=('a', 'mean'), a_std=('a', 'std'),
        b_mean=('b', 'mean'), b_std=('b', 'std'),
        impact_mean=('MetaImpact', 'mean'),
        impact_std=('MetaImpact', 'std'),
        sample_count=('MetaImpact', 'count'),
    ).dropna()

    # Linear-space standard errors
    grouped['err']   = safe_err(grouped['impact_std'], grouped['sample_count'], grouped['impact_mean'])
    grouped['a_err'] = safe_err(grouped['a_std'],      grouped['sample_count'], grouped['a_mean'])
    grouped['b_err'] = safe_err(grouped['b_std'],      grouped['sample_count'], grouped['b_mean'])

    # Keep only positive means (required for log transform)
    grouped = grouped[
        (grouped['a_mean'] > 0) &
        (grouped['b_mean'] > 0) &
        (grouped['impact_mean'] > 0)
    ].copy()

    # Significance mask
    mask_significance = grouped['sample_count'] >= (grouped['sample_count'].max() * sample_threshold_pct)

    # Log10 transforms and log-space error propagation (derivative method)
    log_a = np.log10(grouped['a_mean'])
    log_b = np.log10(grouped['b_mean'])
    log_i = np.log10(grouped['impact_mean'])

    zerr = grouped['err'].values   / (grouped['impact_mean'].values * np.log(10))
    xerr = grouped['a_err'].values / (grouped['a_mean'].values      * np.log(10))
    yerr = grouped['b_err'].values / (grouped['b_mean'].values      * np.log(10))

    log_vals = dict(log_a=log_a, log_b=log_b, log_i=log_i,
                    zerr=zerr, xerr=xerr, yerr=yerr)

    return grouped, log_vals, mask_significance


def iterative_weighted_regression(log_a, log_b, log_i, zerr, xerr, yerr, n_iter=10):
    """
    Fit log_function_3_ab using the Effective Variance Method, iterating
    the sigma estimate until the exponent estimates converge.

    Returns
    -------
    log_Y_fit, d_fit, al_fit : float
        Fitted parameters (Y in log10 space, delta, alpha).
    """
    popt = [np.log10(np.median(10 ** log_i)), 0.5, 0.5]
    for _ in range(n_iter):
        _, d, al = popt
        sig_eff = np.sqrt(zerr**2 + (d**2 * xerr**2) + (al**2 * yerr**2))
        popt, pcov = curve_fit(
            log_function_3_ab,
            (log_a, log_b), log_i,
            p0=popt, sigma=sig_eff, absolute_sigma=True,
        )
    return popt, pcov  # [log_Y_fit, d_fit, al_fit]

# =============================================================================
# PLOT FUNCTIONS
# =============================================================================

def plot_scatter_3d(log_a, log_b, log_i, x_label=X_LABEL, y_label=Y_LABEL,
                    save_path='images\\part_rate\\part_rate_scatter.png'):
    """
    3-D trisurf + scatter of the binned log-space data.

    Parameters
    ----------
    log_a, log_b, log_i : array-like
        Log10-transformed bin means for the two predictors and impact.
    x_label, y_label : str
        Axis label strings (inserted into LaTeX math mode).
    save_path : str
        Output file path.
    """
    fig = plt.figure(figsize=(12, 10))
    ax  = fig.add_subplot(111, projection='3d')
    ax.dist = 8

    ax.plot_trisurf(log_a, log_b, log_i,
                    cmap='Blues', edgecolor='grey', linewidth=0.3, alpha=0.5)
    ax.scatter(log_a, log_b, log_i, marker='.', s=50, alpha=0.8, color='C0')

    ax.set_xlabel(fr'$\log_{{10}}({x_label})$')
    ax.set_ylabel(fr'$\log_{{10}}({y_label})$')
    ax.set_zlabel(r'$\log_{10}(I)$')
    ax.view_init(elev=30, azim=-150)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_histogram_3d(df, x_label=X_LABEL, y_label=Y_LABEL, n_hist=25,
                      save_path='images\\part_rate\\part_rate_hist.png'):
    """
    3-D bar chart showing the sample density across the (log_a, log_b) plane.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with columns 'a' and 'b'.
    n_hist : int
        Number of histogram bins along each axis.
    """
    log_a_all = np.log10(df['a'].to_numpy())
    log_b_all = np.log10(df['b'].to_numpy())

    counts, x_edges, y_edges = np.histogram2d(log_a_all, log_b_all, bins=n_hist)

    x_pos, y_pos = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing='xy')
    x_pos, y_pos = x_pos.ravel(), y_pos.ravel()
    dz = counts.T.ravel()

    dx = (x_edges[1] - x_edges[0]) * 0.85
    dy = (y_edges[1] - y_edges[0]) * 0.85

    norm   = plt.Normalize(dz[dz > 0].min(), dz.max())
    colors = plt.cm.turbo(norm(dz))

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection='3d')
    ax.bar3d(x_pos, y_pos, 0, dx, dy, dz, color=colors, alpha=0.85, shade=True)

    mappable = cm.ScalarMappable(norm=norm, cmap=plt.cm.turbo)
    mappable.set_array(dz)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Sample Density')

    ax.set_xlabel(rf'$\log_{{10}}({x_label})$')
    ax.set_ylabel(rf'$\log_{{10}}({y_label})$')
    ax.set_zlabel('Frequency')
    ax.view_init(elev=30, azim=-150)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_histogram_2d(df, x_label=X_LABEL, y_label=Y_LABEL, n_hist=25,
                      save_path='images\\part_rate\\part_rate_hist_2d.png'):
    """
    2-D heatmap (pcolormesh) projection of the sample density.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with columns 'a' and 'b'.
    n_hist : int
        Number of histogram bins along each axis.
    """
    log_a_all = np.log10(df['a'].to_numpy())
    log_b_all = np.log10(df['b'].to_numpy())

    counts, x_edges, y_edges = np.histogram2d(log_a_all, log_b_all, bins=n_hist)

    x_pos, y_pos = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing='xy')
    x_pos, y_pos = x_pos.ravel(), y_pos.ravel()
    dz = counts.T.ravel()

    unique_x = np.unique(x_pos)
    unique_y = np.unique(y_pos)
    dz_grid  = dz.reshape(len(unique_y), len(unique_x))

    X_grid, Y_grid = np.meshgrid(
        np.append(unique_x, unique_x[-1] + (x_edges[1] - x_edges[0])),
        np.append(unique_y, unique_y[-1] + (y_edges[1] - y_edges[0])),
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(X_grid, Y_grid, dz_grid, cmap='turbo', shading='auto')

    ax.set_xlabel(rf'$\log_{{10}}({x_label})$')
    ax.set_ylabel(rf'$\log_{{10}}({y_label})$')
    fig.colorbar(im, ax=ax, label='Frequency')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_fit_surface(log_a_f, log_b_f, log_i_f, log_Y_fit, d_fit, al_fit,
                     x_label=X_LABEL, y_label=Y_LABEL,
                     save_path='images\\part_rate\\part_rate_fit.png'):
    """
    3-D scatter of the regression data overlaid with the fitted power-law surface.

    Parameters
    ----------
    log_a_f, log_b_f, log_i_f : ndarray
        Log10 data used for fitting (significance-filtered).
    log_Y_fit, d_fit, al_fit : float
        Parameters returned by iterative_weighted_regression.
    """
    a_range = np.linspace(log_a_f.min(), log_a_f.max(), 10)
    b_range = np.linspace(log_b_f.min(), log_b_f.max(), 10)
    a_s, b_s = np.meshgrid(a_range, b_range)
    z_s = log_Y_fit + d_fit * a_s + al_fit * b_s

    fig = plt.figure(figsize=(12, 10))
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(log_a_f, log_b_f, log_i_f,
               marker='o', s=30, alpha=0.8, color='#1f77b4', linewidth=0.5)
    ax.plot_surface(a_s, b_s, z_s,
                    color='gray', alpha=0.25, linewidth=0,
                    antialiased=True, shade=False, zorder=0)

    ax.set_xlabel(rf'$\log_{{10}}({x_label})$')
    ax.set_ylabel(rf'$\log_{{10}}({y_label})$')
    ax.set_zlabel(r'$\log_{10}(I)$')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=15, azim=120)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def power_law(x, a, delta):
    return a * x**delta

def plot_bucci(model = ''):
    dir = 'database\\meta'

    if model != "":
        dir = dir + "_" + model

    paths = np.array(listdir(dir))

    image_name = 'bucci'
    if model != "":
        image_name = image_name + "_" + model

    # Two stacked subplots sharing the X axis: top = points, bottom = distributions
    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1,
        figsize=(8, 10),
        sharex=True,
        gridspec_kw={'hspace': 0.08}   # tight vertical gap; x-tick labels only on bottom
    )

    pattern = r"meta_(?:(?P<num_one>1)|(?P<num_others>\d+)_(?P<kind>\w+?)(?:_(?P<exp>[\d.]+))?)\.csv"

    # Use the default matplotlib color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, path in enumerate(paths):
        if path == 'meta_1.csv':
            continue
        match = re.search(pattern, path)
        num_traders = match.group('num_one') or match.group('num_others')
        kind = match.group('kind') if match.group('kind') else ""
        exp = match.group('exp') if match.group('exp') else ""

        parts = [str(num_traders), kind, exp]
        label = " ".join(filter(None, parts))

        color = color_cycle[i % len(color_cycle)]

        synthetic_meta = pd.read_csv(
            f'{dir}\\{path}',
            sep=',',
            parse_dates=['BeginTime', 'EndTime']
        )

        # 1. Preparazione del DataFrame
        synthetic_meta['NbChild'] = pd.to_numeric(synthetic_meta['NbChild'], errors='coerce')

        df_res = synthetic_meta[['MetaVolume', 'MetaImpact', 'TradedVolume', 'NbChild']].copy()
        df_res = df_res[df_res['NbChild'] > 1]

        df_res.drop(columns=['NbChild'], inplace=True)

        # 2. Calcolo delle nuove variabili
        # Ratio: impact / sqrt(MetaVolume)
        df_res['Ratio'] = df_res['MetaImpact'] / np.sqrt(df_res['MetaVolume'])

        # Participation rate: MetaVolume / TradedVolume (volume traded during execution)
        df_res['ParticipationRate'] = df_res['MetaVolume'] / df_res['TradedVolume']

        # Drop rows with invalid values (zero TradedVolume, NaN, inf)
        df_res = df_res.replace([np.inf, -np.inf], np.nan).dropna(subset=['Ratio', 'ParticipationRate'])
        df_res = df_res[df_res['ParticipationRate'] > 0]

        pr = df_res['ParticipationRate'].to_numpy()

        # 3. Histogram on the bottom subplot
        hist_bins = np.logspace(np.log10(pr.min()), np.log10(pr.max()), 51)
        counts, edges = np.histogram(pr, bins=hist_bins, density=False)
        ax_hist.step(edges[:-1], counts, where='post', color=color, alpha=1.0, linewidth=1.5, label=label)

        # 4. Creazione dei Bin Logaritmici sull'asse X (participation rate)
        bins = np.logspace(np.log10(pr.min()), np.log10(pr.max()), 101)

        # 5. Assegnazione ai Bin
        df_res['bin'] = pd.cut(df_res['ParticipationRate'], bins=bins, include_lowest=True)

        # 6. Raggruppamento e Calcolo delle Medie
        grouped = df_res.groupby('bin', observed=True).agg({
            'ParticipationRate': ['mean', 'std'],
            'Ratio': ['mean', 'std', 'count']
        }).dropna()

        grouped.columns = ['PR_mean', 'PR_std', 'Ratio_mean', 'Ratio_std', 'sample_count']

        # 7. Estrazione degli Array di Risultato
        x = grouped['PR_mean'].to_numpy()
        y = grouped['Ratio_mean'].to_numpy()

        # Top subplot: binned mean points
        ax_main.plot(x, y, linestyle="", marker=".", color=color, label=label)

    # --- Top subplot (points) formatting ---
    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_ylabel(r'$I(Q) / \sqrt{Q}$')
    ax_main.grid(True, which="both", ls="-")
    # Single legend placed to the right of the top subplot
    ax_main.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    # Hide x-tick labels on top plot (shared axis, labels shown on bottom)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # --- Bottom subplot (distributions) formatting ---
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel(r'$\phi = Q / Q_P$')
    ax_hist.set_ylabel('Count')
    ax_hist.grid(True, which="both", ls="-")
    # Extend x-axis to 1 so the full participation rate range is always visible.
    # Note: the histogram naturally stops where your data stops (pr.max() per file),
    # which is typically well below 1. This only affects the axis limit, not the data.
    current_xlim = ax_hist.get_xlim()
    ax_hist.set_xlim(right=max(current_xlim[1], 1.0))

    plt.tight_layout()
    plt.savefig(f'images\\part_rate\\{image_name}.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    # --- Load and prepare raw data ---
    synthetic_meta = pd.read_csv(
        f'{DIR}\\{PATH}', sep=',', parse_dates=['BeginTime', 'EndTime']
    )
    df_res = synthetic_meta[['MetaVolume', 'TradedVolume', 'MetaImpact', 'NbChild']].copy()
    df_res = df_res[df_res['NbChild'] > 1]
    df_res['a'] = df_res['MetaVolume']   # participation rate axis
    df_res['b'] = df_res['MetaVolume'] / df_res['TradedVolume'] # relative volume axis

    # --- Compute binned statistics and log-space errors ---
    grouped, log_vals, mask_significance = compute_binned_stats(df_res, n_bins=31)

    log_a = log_vals['log_a']
    log_b = log_vals['log_b']
    log_i = log_vals['log_i']
    zerr  = log_vals['zerr']
    xerr  = log_vals['xerr']
    yerr  = log_vals['yerr']

    # --- Plots over all bins ---
    plot_scatter_3d(log_a, log_b, log_i)
    plot_histogram_3d(df_res)
    plot_histogram_2d(df_res)

    # --- Regression on significance-filtered bins ---
    log_a_f = log_a[mask_significance].values
    log_b_f = log_b[mask_significance].values
    log_i_f = log_i[mask_significance].values
    zerr_f  = zerr[mask_significance]
    xerr_f  = xerr[mask_significance]
    yerr_f  = yerr[mask_significance]

    popt, pcov = iterative_weighted_regression(
        log_a_f, log_b_f, log_i_f, zerr_f, xerr_f, yerr_f
    )

    log_Y_fit, d_fit, al_fit = popt
    log_Y_err, d_err, al_err = np.sqrt(np.diag(pcov))

    print(f"Fit Results:\nY={10**log_Y_fit:.4g} +- {np.log(10) * 10**log_Y_fit * log_Y_err:.4f}\ndelta={d_fit:.4f} +- {d_err:.4f}\nalpha={al_fit:.4f} +- {al_err:.4f}")

    plot_fit_surface(log_a_f, log_b_f, log_i_f, log_Y_fit, d_fit, al_fit)

    plot_bucci()