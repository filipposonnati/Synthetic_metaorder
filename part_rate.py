import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.odr import ODR, Model, RealData
from os import listdir
import re

DIR = 'database\\meta'
PATH = 'meta_20_uniform.csv'
SAMPLE_THRESHOLD_PCT = 0.01   # as in impact_volume_curve.py: keep bins with count > 0.5 * max

X_LABEL = 'V/V_D'
Y_LABEL = 'V/V_P'

def log_function_3_ab(log_x, log_Y, delta, alpha):
    """
    Linearized Power Law: log10(I) = log10(Y) + delta*log10(a) + alpha*log10(b)
    Equivalent to: I = Y * a^delta * b^alpha
    """
    log_a, log_b = log_x
    return log_Y + delta * log_a + alpha * log_b


def linear_func_odr(p, x):
    """ODR model: log10(I) = p0*log_a + p1*log_b + p2   (x has shape (2, n))."""
    return p[0] * x[0] + p[1] * x[1] + p[2]


def safe_err(std_col, count_col, mean_col=None, fallback=1e-8):
    """
    Standard error of the mean = std / sqrt(n); tiny fallback for bins with
    a single sample (same convention as impact_volume_curve.py).
    """
    err = np.where(count_col > 1, std_col / np.sqrt(count_col), fallback)
    return pd.Series(err, index=std_col.index)

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
    mask_significance = grouped['sample_count'] > (grouped['sample_count'].max() * sample_threshold_pct)

    # Log10 transforms and log-space error propagation (derivative method)
    log_a = np.log10(grouped['a_mean'])
    log_b = np.log10(grouped['b_mean'])
    log_i = np.log10(grouped['impact_mean'])

    # floor of 1e-6 on log-space errors, as in impact_volume_curve.py
    zerr = np.maximum(grouped['err'].values   / (grouped['impact_mean'].values * np.log(10)), 1e-12)
    xerr = np.maximum(grouped['a_err'].values / (grouped['a_mean'].values      * np.log(10)), 1e-12)
    yerr = np.maximum(grouped['b_err'].values / (grouped['b_mean'].values      * np.log(10)), 1e-12)

    log_vals = dict(log_a=log_a, log_b=log_b, log_i=log_i,
                    zerr=zerr, xerr=xerr, yerr=yerr)

    return grouped, log_vals, mask_significance


def odr_regression(log_a, log_b, log_i, zerr, xerr, yerr):
    """
    Orthogonal Distance Regression in log-log space (same procedure as
    impact_volume_curve.py, extended to two regressors):

        log10(I) = delta*log10(a) + alpha*log10(b) + log10(Y)

    Errors on both the regressors (xerr, yerr) and on the response (zerr)
    are taken into account directly by ODR.

    Returns
    -------
    popt : [delta, alpha, log_Y]
    perr : standard deviations of the parameters (same order)
    r2   : coefficient of determination computed in log-log space
    """
    log_a, log_b, log_i = (np.asarray(v, dtype=float) for v in (log_a, log_b, log_i))
    x = np.vstack([log_a, log_b])
    sx = np.vstack([np.asarray(xerr, dtype=float), np.asarray(yerr, dtype=float)])

    data = RealData(x, log_i, sx=sx, sy=np.asarray(zerr, dtype=float))
    out = ODR(data, Model(linear_func_odr), beta0=[1.0, 1.0, 0.0]).run()

    popt, perr = out.beta, out.sd_beta

    y_pred = linear_func_odr(popt, x)
    ss_res = np.sum((log_i - y_pred) ** 2)
    ss_tot = np.sum((log_i - log_i.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return popt, perr, r2

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
    plt.show()

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
        Parameters returned by odr_regression.
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
    plt.show()

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

        ax_hist.step(
            np.append(edges[:-1], edges[-1]),
            np.append(counts, 0),
            where='post', color=color, alpha=1.0, linewidth=1.5, label=label
        )

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

        x_err = grouped['PR_std'].to_numpy() / np.sqrt(grouped['sample_count'].to_numpy())
        y_err = grouped['Ratio_std'].to_numpy() / np.sqrt(grouped['sample_count'].to_numpy())

        """
        mask = y_err < 1e-1
        x = x[mask]
        y = y[mask]
        x_err = x_err[mask]
        y_err = y_err[mask]
        """

        # Top subplot: binned mean points
        #ax_main.plot(x, y, linestyle="", marker=".", color=color, label=label)
        ax_main.errorbar(x, y, y_err, x_err, linestyle="", marker=".", color=color, label=label)

        ax_main.set_xlim(1e-3, 1.2)
        ax_hist.set_xlim(1e-3, 1.2)

    # --- Top subplot (points) formatting ---
    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_ylabel(r'$I(Q) / \sqrt{Q}$')
    ax_main.grid(True, which="both", ls="-")
    # Single legend placed to the right of the top subplot
    ax_main.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), borderaxespad=0)
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
    #plot_bucci()

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
    #plot_scatter_3d(log_a, log_b, log_i)
    plot_histogram_2d(df_res)

    # --- Regression on significance-filtered bins ---
    log_a_f = log_a[mask_significance].values
    log_b_f = log_b[mask_significance].values
    log_i_f = log_i[mask_significance].values
    mask_np = mask_significance.values
    zerr_f  = zerr[mask_np]
    xerr_f  = xerr[mask_np]
    yerr_f  = yerr[mask_np]

    # ODR in log-log space (same procedure as impact_volume_curve.py)
    popt, perr, r2 = odr_regression(
        log_a_f, log_b_f, log_i_f, zerr_f, xerr_f, yerr_f
    )

    d_fit, al_fit, log_Y_fit = popt
    d_err, al_err, log_Y_err = perr

    print(f"Fit Results (ODR):\nY={10**log_Y_fit:.4g} +- {np.log(10) * 10**log_Y_fit * log_Y_err:.4f}\ndelta={d_fit:.4f} +- {d_err:.4f}\nalpha={al_fit:.4f} +- {al_err:.4f}\nR2={r2:.4f}")

    plot_fit_surface(log_a_f, log_b_f, log_i_f, log_Y_fit, d_fit, al_fit)