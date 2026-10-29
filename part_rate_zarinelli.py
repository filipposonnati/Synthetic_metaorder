import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
from scipy.odr import ODR, Model, RealData
import os

# Fit dell'impatto I in funzione di
#   pi = Q / Q_P   (tasso di partecipazione = MetaVolume / TradedVolume)
#   D  = Q / pi    (= TradedVolume)
# invece che di (Q, pi) come in part_rate.py.
#   log10(I) = log10(Y) + alpha*log10(pi) + delta*log10(D)

SAMPLE_THRESHOLD_PCT = 0.04   # come in part_rate.py: si tengono i bin con count > 0.1 * max

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 15
})

XLABEL = r'$\pi = Q/Q_P$'
YLABEL = r'$D = Q/\pi$'


def linear_func_odr(p, x):
    """Piano in log-spazio: log10(I) = p[0] + p[1]*log10(pi) + p[2]*log10(D)."""
    return p[0] + p[1] * x[0] + p[2] * x[1]


def power_law_odr(p, x):
    """Power law nello spazio originale: I = Y * pi**alpha * D**delta."""
    return p[0] * x[0] ** p[1] * x[1] ** p[2]


def safe_err(std_col, count_col, fallback=1e-8):
    """Errore standard della media = std / sqrt(n); fallback per i bin con un solo campione."""
    err = np.where(count_col > 1, std_col / np.sqrt(count_col), fallback)
    return pd.Series(err, index=std_col.index)


def compute_binned_stats(df, n_bins=51, sample_threshold_pct=SAMPLE_THRESHOLD_PCT):
    """Binning logaritmico 2D su (a = pi, b = D); le medie non vengono mai loggate."""
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['a', 'b', 'MetaImpact']).copy()
    df = df[(df['a'] > 0) & (df['b'] > 0)]
    if df.empty:
        raise ValueError('No valid positive a/b values to bin')

    bins_a = np.logspace(np.log10(df['a'].min()), np.log10(df['a'].max()), n_bins)
    bins_b = np.logspace(np.log10(df['b'].min()), np.log10(df['b'].max()), n_bins)
    df['bin_a'] = pd.cut(df['a'], bins=bins_a, include_lowest=True)
    df['bin_b'] = pd.cut(df['b'], bins=bins_b, include_lowest=True)

    grouped = df.groupby(['bin_a', 'bin_b'], observed=True).agg(
        a_mean=('a', 'mean'), a_std=('a', 'std'),
        b_mean=('b', 'mean'), b_std=('b', 'std'),
        impact_mean=('MetaImpact', 'mean'),
        impact_std=('MetaImpact', 'std'),
        sample_count=('MetaImpact', 'count'),
    )
    grouped[['a_std', 'b_std', 'impact_std']] = grouped[['a_std', 'b_std', 'impact_std']].fillna(0)
    grouped['err'] = safe_err(grouped['impact_std'], grouped['sample_count'])
    grouped['a_err'] = safe_err(grouped['a_std'], grouped['sample_count'])
    grouped['b_err'] = safe_err(grouped['b_std'], grouped['sample_count'])
    mask = grouped['sample_count'] > grouped['sample_count'].max() * sample_threshold_pct
    return grouped, mask


def odr_regression(a, b, impact, zerr, xerr, yerr, model_kind='linear'):
    """Regressione ODR: fit lineare in log-spazio oppure power law nello spazio originale."""
    a, b, impact = (np.asarray(v, dtype=float) for v in (a, b, impact))
    ln10 = np.log(10)

    if model_kind == 'linear':
        log_a, log_b, log_impact = np.log10(a), np.log10(b), np.log10(impact)
        x = np.vstack((log_a, log_b))
        sx = np.vstack((
            np.maximum(xerr / (a * ln10), 1e-12),
            np.maximum(yerr / (b * ln10), 1e-12)
        ))
        sy = np.maximum(zerr / (impact * ln10), 1e-12)
        beta0 = [float(np.mean(log_impact)), 0.5, 0.5]
        out = ODR(RealData(x, log_impact, sx=sx, sy=sy), Model(linear_func_odr), beta0=beta0).run()
        pred = linear_func_odr(out.beta, x)
        ss_tot = np.sum((log_impact - log_impact.mean()) ** 2)
        r2 = 1 - np.sum((log_impact - pred) ** 2) / ss_tot if ss_tot > 0 else np.nan
        return out.beta, out.sd_beta, r2

    elif model_kind == 'power_law':
        x = np.vstack((a, b))
        sx = np.vstack((np.maximum(xerr, 1e-12), np.maximum(yerr, 1e-12)))
        sy = np.maximum(zerr, 1e-12)
        beta0 = [float(np.median(impact) / (np.median(a) ** 0.5 * np.median(b) ** 0.5)), 0.5, 0.5]
        out = ODR(RealData(x, impact, sx=sx, sy=sy), Model(power_law_odr), beta0=beta0).run()
        pred = power_law_odr(out.beta, x)
        ss_tot = np.sum((impact - impact.mean()) ** 2)
        r2 = 1 - np.sum((impact - pred) ** 2) / ss_tot if ss_tot > 0 else np.nan
        return out.beta, out.sd_beta, r2
    else:
        raise ValueError('model_kind must be linear or power_law')


# =============================================================================
# PLOT
# =============================================================================

def plot_histogram_2d(df, n_hist=51, save_path=None):
    valid = df[['a', 'b']].replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[(valid['a'] > 0) & (valid['b'] > 0)]
    x_edges = np.logspace(np.log10(valid['a'].min()), np.log10(valid['a'].max()), n_hist + 1)
    y_edges = np.logspace(np.log10(valid['b'].min()), np.log10(valid['b'].max()), n_hist + 1)
    counts, _, _ = np.histogram2d(valid['a'], valid['b'], bins=(x_edges, y_edges))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(x_edges, y_edges, counts.T, cmap='turbo', shading='auto')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    fig.colorbar(im, ax=ax, label='Frequency')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _valid_arrays(grouped, mask_significance):
    a = grouped['a_mean'].to_numpy()
    b = grouped['b_mean'].to_numpy()
    impact = grouped['impact_mean'].to_numpy()
    valid = (a > 0) & (b > 0) & (impact > 0)
    mask = mask_significance.loc[grouped.index][valid].to_numpy()
    return grouped, valid, mask


def plot_fit_surface(grouped, mask_significance, popt, model_kind='linear', save_path=None):
    """Superficie di fit e punti binnati dell'impatto, in funzione di (pi, D)."""
    grouped, valid, mask = _valid_arrays(grouped, mask_significance)
    a = grouped['a_mean'].to_numpy()[valid]
    b = grouped['b_mean'].to_numpy()[valid]
    impact = grouped['impact_mean'].to_numpy()[valid]
    xerr = grouped['a_err'].to_numpy()[valid]
    yerr = grouped['b_err'].to_numpy()[valid]
    zerr = grouped['err'].to_numpy()[valid]

    log_a, log_b, log_impact = np.log10(a), np.log10(b), np.log10(impact)
    ln10 = np.log(10)
    log_xerr = xerr / (a * ln10)
    log_yerr = yerr / (b * ln10)
    log_zerr = zerr / (impact * ln10)

    a_s, b_s = np.meshgrid(np.linspace(log_a.min(), log_a.max(), 40),
                           np.linspace(log_b.min(), log_b.max(), 40))
    if model_kind == 'linear':
        z_s = popt[0] + popt[1] * a_s + popt[2] * b_s
    else:
        z_s = np.log10(np.clip(popt[0] * ((10 ** a_s) ** popt[1]) * ((10 ** b_s) ** popt[2]),
                               1e-12, None))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sel, unsel = mask, ~mask

    if np.any(unsel):
        ax.scatter(log_a[unsel], log_b[unsel], log_impact[unsel],
                   s=15, color='gray', alpha=0.25, label='Discarded')
        ax.errorbar(log_a[unsel], log_b[unsel], log_impact[unsel],
                    zerr=log_zerr[unsel], xerr=log_xerr[unsel], yerr=log_yerr[unsel],
                    fmt='none', ecolor='gray', elinewidth=0.5, alpha=0.15)

    ax.scatter(log_a[sel], log_b[sel], log_impact[sel],
               s=30, color='tab:blue', alpha=0.9, edgecolors='k', linewidth=0.3, label='Selected')
    ax.errorbar(log_a[sel], log_b[sel], log_impact[sel],
                zerr=log_zerr[sel], xerr=log_xerr[sel], yerr=log_yerr[sel],
                fmt='none', ecolor='tab:blue', elinewidth=1, alpha=0.7)

    ax.plot_surface(a_s, b_s, z_s, cmap='gray', alpha=0.4, edgecolor='none', rstride=1, cstride=1)

    ax.set_xlabel(r'$\log_{10}(\pi)$', labelpad=10)
    ax.set_ylabel(r'$\log_{10}(D)$', labelpad=10)
    ax.set_zlabel(r'$\log_{10}(I)$', labelpad=20)
    ax.view_init(elev=25, azim=150)
    ax.legend(loc='upper left')
    ax.tick_params(axis='z', pad=10)

    ax.set_zlim([-4, 0])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()


def plot_count_bars(grouped, mask_significance, save_path=None):
    """Istogramma 3D dei conteggi per bin (pi, D)."""
    grouped, valid, mask = _valid_arrays(grouped, mask_significance)
    a = grouped['a_mean'].to_numpy()[valid]
    b = grouped['b_mean'].to_numpy()[valid]
    counts = grouped['sample_count'].to_numpy()[valid]

    log_a, log_b = np.log10(a), np.log10(b)
    elev, azim = 25, 150

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.view_init(elev=elev, azim=azim)

    dx = ((log_a.max() - log_a.min()) / 45) * 0.9
    dy = ((log_b.max() - log_b.min()) / 45) * 0.9

    rgba_colors = np.zeros((len(log_a), 4))
    rgba_colors[~mask] = [0.85, 0.85, 0.85, 1.0]
    if np.any(mask):
        norm = plt.Normalize(vmin=counts[mask].min(), vmax=counts[mask].max())
        rgba_colors[mask] = cm.viridis(norm(counts[mask].astype(float)))

    # ordinamento back-to-front rispetto alla camera
    az, el = np.deg2rad(azim), np.deg2rad(elev)
    cam_dir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az)])
    order = np.argsort(log_a * cam_dir[0] + log_b * cam_dir[1])

    for i in order:
        ax.bar3d(log_a[i] - dx / 2, log_b[i] - dy / 2, 0,
                 dx, dy, float(counts[i]),
                 color=rgba_colors[i], alpha=1.0,
                 edgecolor='black', linewidth=0.4, shade=False)

    ax.set_xlabel(r'$\log_{10}(\pi)$', labelpad=12)
    ax.set_ylabel(r'$\log_{10}(D)$', labelpad=12)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Count', labelpad=38, rotation=90)
    ax.tick_params(axis='z', pad=20)

    legend_elements = [
        Patch(facecolor=[0.85, 0.85, 0.85, 1.0], edgecolor='black', linewidth=0.5, label='Discarded'),
        Patch(facecolor=cm.viridis(0.5), edgecolor='black', linewidth=0.5, label='Selected')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


if __name__ == '__main__':
    model = 'lmf_1.5_0.3_empirical_tim_0.0_1000'
    #model = 'tim_0.0_1000'

    DIR = os.path.join('database', f'meta{"" if model == "" else f"_{model}"}')
    OUT_DIR = os.path.join('images', 'part_rate_zarinelli')

    nb_traders = 20
    kind = 'power'
    exp = '2.0'

    if kind == 'uniform':
        conf = f'{nb_traders}_{kind}'
    else:
        conf = f'{nb_traders}_{kind}_{exp}'

    print(model, conf)

    PATH = f'meta_{conf}.csv'
    MODEL_KIND = 'power_law'   # 'linear' = piano in log-spazio, 'power_law' = spazio originale

    os.makedirs(OUT_DIR, exist_ok=True)

    synthetic_meta = pd.read_csv(os.path.join(DIR, PATH), sep=',',
                                 parse_dates=['BeginTime', 'EndTime'])
    df_res = synthetic_meta[['MetaVolume', 'TradedVolume', 'MetaImpact', 'NbChild']].copy()
    df_res = df_res[df_res['NbChild'] > 1]

    # variabili del fit: a = pi = Q / TradedVolume, b = D = Q / pi
    df_res['a'] = df_res['MetaVolume'] / df_res['TradedVolume']
    df_res['b'] = df_res['MetaVolume'] / df_res['a']

    grouped, mask_significance = compute_binned_stats(df_res, n_bins=51)
    selected = grouped.loc[mask_significance]

    plot_histogram_2d(df_res, save_path=os.path.join(OUT_DIR, f'{f"{model}_" if model != "" else ""}{conf}_part_rate_zarinelli_hist.png'))

    popt, perr, r2 = odr_regression(
        selected['a_mean'].to_numpy(), selected['b_mean'].to_numpy(),
        selected['impact_mean'].to_numpy(),
        selected['err'].to_numpy(), selected['a_err'].to_numpy(), selected['b_err'].to_numpy(),
        model_kind=MODEL_KIND)

    # I = Y * pi^alpha * D^delta
    if MODEL_KIND == 'linear':
        logY_val, alpha_val, delta_val = popt
        logY_err, alpha_err, delta_err = perr
        Y_val = 10 ** logY_val
        Y_err = Y_val * np.log(10) * logY_err
        print(f'log10(Y) = {logY_val:.6g} +/- {logY_err:.3g} (Y = {Y_val:.6g} +/- {Y_err:.3g})')
        print(f'alpha (pi) = {alpha_val:.6g} +/- {alpha_err:.3g}')
        print(f'delta (D)  = {delta_val:.6g} +/- {delta_err:.3g}')
    else:
        for name, val, err in zip(('Y', 'alpha (pi)', 'delta (D)'), popt, perr):
            print(f'{name} = {val:.6g} +/- {err:.3g}')

    print(f'R2 (log scale) = {r2:.4f}')

    plot_fit_surface(grouped, mask_significance, popt, model_kind=MODEL_KIND,
                     save_path=os.path.join(OUT_DIR, f'{f"{model}_" if model != "" else ""}{conf}_part_rate_zarinelli_fit.png'))
    plot_count_bars(grouped, mask_significance,
                    save_path=os.path.join(OUT_DIR, f'{f"{model}_" if model != "" else ""}{conf}_part_rate_zarinelli_counts.png'))