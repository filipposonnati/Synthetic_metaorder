import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.odr import ODR, Model, RealData
import os
from os import listdir
import re

DIR = 'database\\meta'
SAMPLE_THRESHOLD_PCT = 0.1   # as in impact_volume_curve.py: keep bins with count > 0.1 * max

def linear_func_odr(p, x):
    """Linear plane in log-space: log10(I) = p[0] + p[1]*log10(a) + p[2]*log10(b)."""
    return p[0] + p[1] * x[0] + p[2] * x[1]


def power_law_odr(p, x):
    """Optional original-space power law: I = Y*a**delta*b**alpha."""
    return p[0] * x[0] ** p[1] * x[1] ** p[2]


def safe_err(std_col, count_col, mean_col=None, fallback=1e-8):
    """
    Standard error of the mean = std / sqrt(n); tiny fallback for bins with
    a single sample (same convention as impact_volume_curve.py).
    """
    err = np.where(count_col > 1, std_col / np.sqrt(count_col), fallback)
    return pd.Series(err, index=std_col.index)

def compute_binned_stats(df, n_bins=31, sample_threshold_pct=SAMPLE_THRESHOLD_PCT):
    """Bin original values with logarithmically spaced edges; never log the means."""
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
    # Single-observation bins have undefined sample std, not undefined means.
    grouped[['a_std', 'b_std', 'impact_std']] = grouped[['a_std', 'b_std', 'impact_std']].fillna(0)
    grouped['err'] = safe_err(grouped['impact_std'], grouped['sample_count'])
    grouped['a_err'] = safe_err(grouped['a_std'], grouped['sample_count'])
    grouped['b_err'] = safe_err(grouped['b_std'], grouped['sample_count'])
    mask = grouped['sample_count'] > grouped['sample_count'].max() * sample_threshold_pct
    return grouped, mask


def odr_regression(a, b, impact, zerr, xerr, yerr, model_kind='linear'):
    """ODR regression: log-space linear fit or original-space power law."""
    a, b, impact = (np.asarray(v, dtype=float) for v in (a, b, impact))
    ln10 = np.log(10)

    if model_kind == 'linear':
        # Fit in log space: log10(I) = log10(Y) + delta*log10(a) + alpha*log10(b)
        log_a = np.log10(a)
        log_b = np.log10(b)
        log_impact = np.log10(impact)

        x = np.vstack((log_a, log_b))
        # Error propagation for log10: sigma_log(v) = sigma_v / (v * ln(10))
        sx = np.vstack((
            np.maximum(xerr / (a * ln10), 1e-12),
            np.maximum(yerr / (b * ln10), 1e-12)
        ))
        sy = np.maximum(zerr / (impact * ln10), 1e-12)

        func = linear_func_odr
        beta0 = [float(np.mean(log_impact)), 0.5, 0.5]

        out = ODR(RealData(x, log_impact, sx=sx, sy=sy), Model(func), beta0=beta0).run()
        pred = func(out.beta, x)
        ss_tot = np.sum((log_impact - log_impact.mean())**2)
        r2 = 1 - np.sum((log_impact - pred)**2) / ss_tot if ss_tot > 0 else np.nan
        return out.beta, out.sd_beta, r2

    elif model_kind == 'power_law':
        x = np.vstack((a, b))
        sx = np.vstack((np.maximum(xerr, 1e-12), np.maximum(yerr, 1e-12)))
        sy = np.maximum(zerr, 1e-12)
        func = power_law_odr
        beta0 = [float(np.median(impact) / (np.median(a)**0.5 * np.median(b)**0.5)), 0.5, 0.5]
        out = ODR(RealData(x, impact, sx=sx, sy=sy), Model(func), beta0=beta0).run()
        pred = func(out.beta, x)
        ss_tot = np.sum((impact - impact.mean())**2)
        r2 = 1 - np.sum((impact - pred)**2) / ss_tot if ss_tot > 0 else np.nan
        return out.beta, out.sd_beta, r2
    else:
        raise ValueError('model_kind must be linear or power_law')


# =============================================================================
# PLOT FUNCTIONS
# =============================================================================

def plot_histogram_2d(df, n_hist=25, save_path='images\\part_rate\\part_rate_hist_2d.png'):
    valid = df[['a', 'b']].replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[(valid['a'] > 0) & (valid['b'] > 0)]
    x_edges = np.logspace(np.log10(valid['a'].min()), np.log10(valid['a'].max()), n_hist + 1)
    y_edges = np.logspace(np.log10(valid['b'].min()), np.log10(valid['b'].max()), n_hist + 1)
    counts, _, _ = np.histogram2d(valid['a'], valid['b'], bins=(x_edges, y_edges))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(x_edges, y_edges, counts.T, cmap='turbo', shading='auto')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$V/V_D$'); ax.set_ylabel(r'$V/V_P$')
    fig.colorbar(im, ax=ax, label='Frequency')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_fit_surface(grouped, mask_significance, popt, model_kind='linear', save_path='images\\part_rate\\part_rate_fit.png'):
    """Figura 1: Superficie di fit e punti binnati dell'intensità."""
    a = grouped['a_mean'].to_numpy()
    b = grouped['b_mean'].to_numpy()
    impact = grouped['impact_mean'].to_numpy()
    counts = grouped['sample_count'].to_numpy()
    
    xerr = grouped['a_err'].to_numpy()
    yerr = grouped['b_err'].to_numpy()
    zerr = grouped['err'].to_numpy()

    valid = (a > 0) & (b > 0) & (impact > 0)
    a, b, impact, counts = a[valid], b[valid], impact[valid], counts[valid]
    mask = mask_significance.loc[grouped.index][valid].to_numpy()
    xerr, yerr, zerr = xerr[valid], yerr[valid], zerr[valid]

    log_a = np.log10(a)
    log_b = np.log10(b)
    log_impact = np.log10(impact)

    ln10 = np.log(10)
    log_xerr = xerr / (a * ln10)
    log_yerr = yerr / (b * ln10)
    log_zerr = zerr / (impact * ln10)

    a_range = np.linspace(log_a.min(), log_a.max(), 40)
    b_range = np.linspace(log_b.min(), log_b.max(), 40)
    a_s, b_s = np.meshgrid(a_range, b_range)

    if model_kind == 'linear':
        z_s = popt[0] + popt[1] * a_s + popt[2] * b_s
    else:
        z_s_orig = popt[0] * ((10**a_s) ** popt[1]) * ((10**b_s) ** popt[2])
        z_s = np.log10(np.clip(z_s_orig, 1e-12, None))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sel = mask
    unsel = ~mask

    # Punti non selezionati (grigi e trasparenti)
    if np.any(unsel):
        ax.scatter(log_a[unsel], log_b[unsel], log_impact[unsel],
                   s=15, color='gray', alpha=0.25, label='Discarded')
        ax.errorbar(log_a[unsel], log_b[unsel], log_impact[unsel],
                    zerr=log_zerr[unsel], xerr=log_xerr[unsel], yerr=log_yerr[unsel],
                    fmt='none', ecolor='gray', elinewidth=0.5, alpha=0.15)

    # Punti selezionati (usati per il fit)
    ax.scatter(log_a[sel], log_b[sel], log_impact[sel],
               s=30, color='tab:blue', alpha=0.9, edgecolors='k', linewidth=0.3, label='Selected')
    ax.errorbar(log_a[sel], log_b[sel], log_impact[sel],
                zerr=log_zerr[sel], xerr=log_xerr[sel], yerr=log_yerr[sel],
                fmt='none', ecolor='tab:blue', elinewidth=1, alpha=0.7)

    ax.plot_surface(a_s, b_s, z_s, cmap='gray', alpha=0.4, edgecolor='none', rstride=1, cstride=1)

    ax.set_xlabel(r'$log_{10}(V/V_D)$', labelpad=10)
    ax.set_ylabel(r'$log_{10}(V/V_P)$', labelpad=10)
    ax.set_zlabel(r'log_{10}(I)', labelpad=10)
    ax.view_init(elev=25, azim=135)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_fit_surface_selected_only(grouped, mask_significance, popt, model_kind='linear', save_path='images\\part_rate\\part_rate_fit_selected_only.png'):
    """Figura 3: Superficie di fit e unicamente i punti selezionati (senza i punti scartati)."""
    a = grouped['a_mean'].to_numpy()
    b = grouped['b_mean'].to_numpy()
    impact = grouped['impact_mean'].to_numpy()
    
    xerr = grouped['a_err'].to_numpy()
    yerr = grouped['b_err'].to_numpy()
    zerr = grouped['err'].to_numpy()

    valid = (a > 0) & (b > 0) & (impact > 0)
    a, b, impact = a[valid], b[valid], impact[valid]
    mask = mask_significance.loc[grouped.index][valid].to_numpy()
    xerr, yerr, zerr = xerr[valid], yerr[valid], zerr[valid]

    # Manteniamo unicamente i dati selezionati dalla maschera
    log_a = np.log10(a[mask])
    log_b = np.log10(b[mask])
    log_impact = np.log10(impact[mask])

    ln10 = np.log(10)
    log_xerr = xerr[mask] / (a[mask] * ln10)
    log_yerr = yerr[mask] / (b[mask] * ln10)
    log_zerr = zerr[mask] / (impact[mask] * ln10)

    # Creazione della griglia per la superficie di fit
    a_range = np.linspace(log_a.min(), log_a.max(), 40)
    b_range = np.linspace(log_b.min(), log_b.max(), 40)
    a_s, b_s = np.meshgrid(a_range, b_range)

    if model_kind == 'linear':
        z_s = popt[0] + popt[1] * a_s + popt[2] * b_s
    else:
        z_s_orig = popt[0] * ((10**a_s) ** popt[1]) * ((10**b_s) ** popt[2])
        z_s = np.log10(np.clip(z_s_orig, 1e-12, None))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot dei soli punti selezionati con le barre d'errore
    ax.scatter(log_a, log_b, log_impact,
               s=35, color='tab:blue', alpha=0.9, edgecolors='k', linewidth=0.3, label='Binned Data')
    ax.errorbar(log_a, log_b, log_impact,
                zerr=log_zerr, xerr=log_xerr, yerr=log_yerr,
                fmt='none', ecolor='tab:blue', elinewidth=1, alpha=0.8)

    # Superficie di fit
    ax.plot_surface(a_s, b_s, z_s, cmap='gray', alpha=0.5, edgecolor='none', rstride=1, cstride=1)

    ax.set_xlabel(r'$log_{10}(V/V_D)$', labelpad=10)
    ax.set_ylabel(r'$log_{10}(V/V_P)$', labelpad=10)
    ax.set_zlabel(r'log_{10}(I)', labelpad=10)
    ax.view_init(elev=25, azim=135)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_count_bars(grouped, mask_significance, save_path='images\\part_rate\\part_rate_counts.png'):
    """Figura 2: Istogramma 3D con altezza e colore proporzionali al count."""
    a = grouped['a_mean'].to_numpy()
    b = grouped['b_mean'].to_numpy()
    impact = grouped['impact_mean'].to_numpy()
    counts = grouped['sample_count'].to_numpy()
    
    valid = (a > 0) & (b > 0) & (impact > 0)
    a, b, impact, counts = a[valid], b[valid], impact[valid], counts[valid]
    mask = mask_significance.loc[grouped.index][valid].to_numpy()

    log_a = np.log10(a)
    log_b = np.log10(b)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sel = mask
    unsel = ~mask

    dx = (log_a.max() - log_a.min()) / 45
    dy = (log_b.max() - log_b.min()) / 45
    z_floor = 0

    # 1. Barre dei punti scartati (grigie e trasparenti)
    if np.any(unsel):
        ax.bar3d(log_a[unsel] - dx/2, log_b[unsel] - dy/2, z_floor,
                 dx, dy, counts[unsel].astype(float),
                 color='gray', alpha=0.2, edgecolor='none', label='Discarded')

    # 2. Barre dei punti selezionati (altezza e colore basati sul COUNT)
    if np.any(sel):
        norm = plt.Normalize(vmin=counts[sel].min(), vmax=counts[sel].max())
        colors = cm.viridis(norm(counts[sel].astype(float)))

        ax.bar3d(log_a[sel] - dx/2, log_b[sel] - dy/2, z_floor,
                 dx, dy, counts[sel].astype(float),
                 color=colors, alpha=0.85, edgecolor='k', linewidth=0.2, label='Selected')

        # Colorbar basata sul count
        m = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
        m.set_array(counts[sel].astype(float))
        cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=12, pad=0.1)
        cbar.set_label('Count')

    ax.set_xlabel(r'$\log_{10}(V/V_D)$', labelpad=10)
    ax.set_ylabel(r'$\log_{10}(V/V_P)$', labelpad=10)
    ax.set_zlabel('Count', labelpad=10)
    ax.view_init(elev=25, azim=135)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bucci(model = ''):
    dir = 'database\\meta'

    if model != "":
        dir = dir + "_" + model

    paths = np.array(listdir(dir))

    image_name = 'bucci'
    if model != "":
        image_name = image_name + "_" + model

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1,
        figsize=(8, 10),
        sharex=True,
        gridspec_kw={'hspace': 0.08}
    )

    pattern = r"meta_(?:(?P<num_one>1)|(?P<num_others>\d+)_(?P<kind>\w+?)(?:_(?P<exp>[\d.]+))?)\.csv"
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

        synthetic_meta['NbChild'] = pd.to_numeric(synthetic_meta['NbChild'], errors='coerce')
        df_res = synthetic_meta[['MetaVolume', 'MetaImpact', 'TradedVolume', 'NbChild']].copy()
        df_res = df_res[df_res['NbChild'] > 1]
        df_res.drop(columns=['NbChild'], inplace=True)

        df_res['Ratio'] = df_res['MetaImpact'] / np.sqrt(df_res['MetaVolume'])
        df_res['ParticipationRate'] = df_res['MetaVolume'] / df_res['TradedVolume']

        df_res = df_res.replace([np.inf, -np.inf], np.nan).dropna(subset=['Ratio', 'ParticipationRate'])
        df_res = df_res[df_res['ParticipationRate'] > 0]
        pr = df_res['ParticipationRate'].to_numpy()

        hist_bins = np.logspace(np.log10(pr.min()), np.log10(pr.max()), 51)
        counts, edges = np.histogram(pr, bins=hist_bins, density=False)

        ax_hist.step(
            np.append(edges[:-1], edges[-1]),
            np.append(counts, 0),
            where='post', color=color, alpha=1.0, linewidth=1.5, label=label
        )

        bins = np.logspace(np.log10(pr.min()), np.log10(pr.max()), 101)
        df_res['bin'] = pd.cut(df_res['ParticipationRate'], bins=bins, include_lowest=True)

        grouped = df_res.groupby('bin', observed=True).agg({
            'ParticipationRate': ['mean', 'std'],
            'Ratio': ['mean', 'std', 'count']
        }).dropna()

        grouped.columns = ['PR_mean', 'PR_std', 'Ratio_mean', 'Ratio_std', 'sample_count']

        x = grouped['PR_mean'].to_numpy()
        y = grouped['Ratio_mean'].to_numpy()

        x_err = grouped['PR_std'].to_numpy() / np.sqrt(grouped['sample_count'].to_numpy())
        y_err = grouped['Ratio_std'].to_numpy() / np.sqrt(grouped['sample_count'].to_numpy())

        ax_main.errorbar(x, y, y_err, x_err, linestyle="", marker=".", color=color, label=label)
        ax_main.set_xlim(1e-3, 1.2)
        ax_hist.set_xlim(1e-3, 1.2)

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_ylabel(r'$I(Q) / \sqrt{Q}$')
    ax_main.grid(True, which="both", ls="-")
    ax_main.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), borderaxespad=0)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    ax_hist.set_xscale("log")
    ax_hist.set_xlabel(r'$\phi = Q / Q_P$')
    ax_hist.set_ylabel('Count')
    ax_hist.grid(True, which="both", ls="-")
    current_xlim = ax_hist.get_xlim()
    ax_hist.set_xlim(right=max(current_xlim[1], 1.0))

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    plt.savefig(f'images\\part_rate\\{image_name}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_impact_volume_by_deciles(dir_path, path, n_deciles=10, min_nb_child=1, 
                                  save_path='images/part_rate/impact_volume_deciles.png',
                                  dist_save_path=None):
    """
    Replica esattamente la logica di impact_volume_curve.py per il caricamento e i filtri,
    aggiungendo la suddivisione in decili del Participation Rate e la curva All Data.
    Inoltre, genera la figura dedicata alla distribuzione del Participation Rate divisa in decili.
    """
    # 1. Caricamento del file esattamente come in impact_volume_curve.py
    if os.path.exists(os.path.join(dir_path, path + '_' + str(min_nb_child) + '.csv')):
        data_path = os.path.join(dir_path, path + '_' + str(min_nb_child) + '.csv')
    else:
        data_path = os.path.join(dir_path, path + '.csv')
        
    synthetic_meta = pd.read_csv(
        data_path,
        sep=',',
        parse_dates=['BeginTime', 'EndTime']
    )

    # 2. Filtri identici a impact_volume_curve.py
    synthetic_meta = synthetic_meta[synthetic_meta['MetaVolume'] > 0]
    df_res = synthetic_meta[['MetaVolume', 'DailyVolume', 'TradedVolume', 'NbChild', 'MetaImpact']].copy()
    
    # Applicazione del filtro su NbChild (uguale a cfg['min_val'] con op '>')
    df_res = df_res[df_res['NbChild'] > min_nb_child]

    if df_res.empty:
        print("Nessun dato valido disponibile.")
        return

    # Calcolo del Participation Rate
    df_res['ParticipationRate'] = df_res['MetaVolume'] / df_res['TradedVolume']

    # 3. Creazione dei decili (gestendo eventuali inf/nan solo per il qcut dei decili)
    df_valid_pr = df_res.replace([np.inf, -np.inf], np.nan).dropna(subset=['ParticipationRate'])
    df_valid_pr = df_valid_pr[df_valid_pr['ParticipationRate'] > 0]

    try:
        df_valid_pr['decile'] = pd.qcut(df_valid_pr['ParticipationRate'], q=n_deciles, labels=False, duplicates='drop')
    except Exception as e:
        print(f"Errore nella creazione dei decili: {e}")
        return

    actual_deciles = sorted(df_valid_pr['decile'].unique())
    cmap = plt.get_cmap('tab10', len(actual_deciles))

    # --- Generazione della figura della distribuzione del Participation Rate in decili ---
    if dist_save_path is None:
        dist_save_path = save_path.replace('impact_volume_deciles', 'part_rate_deciles_dist')

    pr_array = df_valid_pr['ParticipationRate'].to_numpy()
    percentiles = np.linspace(0, 100, n_deciles + 1)
    decile_edges = np.percentile(pr_array, percentiles)

    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
    hist_bins_dist = np.logspace(np.log10(pr_array.min()), np.log10(pr_array.max()), 60)
    ax_dist.hist(pr_array, bins=hist_bins_dist, color='lightgray', edgecolor='black', linewidth=0.5, label='Distribution')

    for i in range(len(actual_deciles)):
        d_val = actual_deciles[i]
        left_edge = decile_edges[i]
        right_edge = decile_edges[i+1]
        color = cmap(i)
        ax_dist.axvspan(left_edge, right_edge, color=color, alpha=0.2, label=f'D{d_val+1}')
        ax_dist.axvline(left_edge, color=color, linestyle='--', linewidth=1)
    ax_dist.axvline(decile_edges[len(actual_deciles)], color=cmap(len(actual_deciles)-1), linestyle='--', linewidth=1)

    ax_dist.set_xscale('log')
    ax_dist.set_xlabel(r'$\phi = V / V_P$')
    ax_dist.set_ylabel('Count')
    #ax_dist.set_title('Distribution Splitting')
    ax_dist.grid(True, which='both', ls='--', alpha=0.3)
    ax_dist.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(dist_save_path), exist_ok=True)
    fig_dist.savefig(dist_save_path, dpi=300, bbox_inches='tight')
    print(f"Grafico della distribuzione dei decili salvato in: {dist_save_path}")
    plt.close(fig_dist)
    # --------------------------------------------------------------------------------------------

    fig, ax1 = plt.subplots(figsize=(11, 7))

    all_x_min = []
    all_x_max = []

    # Funzione interna di binning e fit ODR (identica a impact_volume_curve.py)
    def process_and_fit(sub_df):
        min_vol = sub_df['MetaVolume'].min()
        max_vol = sub_df['MetaVolume'].max()
        bins = np.logspace(np.log10(min_vol), np.log10(max_vol), 51)

        sub_df_binned = sub_df.copy()
        sub_df_binned['bin'] = pd.cut(sub_df_binned['MetaVolume'], bins=bins, include_lowest=True)

        grouped = sub_df_binned.groupby('bin', observed=True).agg({
            'MetaVolume': ['mean', 'std', 'count'],
            'MetaImpact': ['mean', 'std']
        }).dropna()

        grouped.columns = ['Q_mean', 'Q_std', 'sample_count', 'I_mean', 'I_std']
        grouped = grouped[(grouped['Q_mean'] > 0) & (grouped['I_mean'] > 0)]

        if grouped.empty:
            return None, None

        x_data = grouped['Q_mean'].values
        y_data = grouped['I_mean'].values

        # Soglia a 0.5 * max_samples (come in impact_volume_curve.py)
        max_samples = grouped['sample_count'].max()
        grouped_fit = grouped[grouped['sample_count'] > 0.5 * max_samples].copy()

        fit_res = None
        if len(grouped_fit) > 2:
            x_fit_data = grouped_fit['Q_mean'].values
            y_fit_data = grouped_fit['I_mean'].values
            counts = grouped_fit['sample_count'].values

            x_sem = np.where(counts > 1, grouped_fit['Q_std'].values / np.sqrt(counts), 1e-8)
            y_sem = np.where(counts > 1, grouped_fit['I_std'].values / np.sqrt(counts), 1e-8)

            log_x = np.log10(x_fit_data)
            log_y = np.log10(y_fit_data)

            sigma_log_x = np.maximum(x_sem / (x_fit_data * np.log(10)), 1e-6)
            sigma_log_y = np.maximum(y_sem / (y_fit_data * np.log(10)), 1e-6)

            def linear_func(p, x):
                return p[0] * x + p[1]

            model_odr = Model(linear_func)
            data_odr = RealData(log_x, log_y, sx=sigma_log_x, sy=sigma_log_y)
            
            # beta0 = [1.0, 0.0] come in impact_volume_curve.py
            odr = ODR(data_odr, model_odr, beta0=[1.0, 0.0])
            out = odr.run()

            slope, intercept = out.beta
            slope_err, _ = out.sd_beta

            x_fit = np.logspace(np.log10(x_fit_data.min()), np.log10(x_fit_data.max()), 100)
            y_fit = (10**intercept) * (x_fit**slope)
            fit_res = (x_fit, y_fit, slope, slope_err)

        return (x_data, y_data), fit_res

    # 1. PLOT DEI SINGOLI DECILI
    for i, d in enumerate(actual_deciles):
        sub_df = df_valid_pr[df_valid_pr['decile'] == d]
        if sub_df.empty:
            continue

        eta_min = sub_df['ParticipationRate'].min()
        eta_max = sub_df['ParticipationRate'].max()
        color = cmap(i)
        label_str = f'D{d+1}'

        data_pts, fit_res = process_and_fit(sub_df)
        if data_pts is None:
            continue

        x_data, y_data = data_pts
        all_x_min.append(np.min(x_data))
        all_x_max.append(np.max(x_data))

        ax1.plot(x_data, y_data, linestyle='', marker='o', markersize=3.5, color=color, alpha=1.0)

        if fit_res is not None:
            x_fit, y_fit, slope, slope_err = fit_res
            ax1.plot(
                x_fit, y_fit, linestyle='--', linewidth=1.2, color=color,
                label=f"{label_str} ${slope:.3f} \\pm {slope_err:.3f}$"
            )

    # 2. PLOT DATI COMPLETI ("All Data" usando l'intero df_res filtrato come in impact_volume_curve.py)
    data_pts_all, fit_res_all = process_and_fit(df_res)
    if data_pts_all is not None:
        x_data_all, y_data_all = data_pts_all
        all_x_min.append(np.min(x_data_all))
        all_x_max.append(np.max(x_data_all))
        
        ax1.plot(x_data_all, y_data_all, linestyle='', marker='o', markersize=5, color='black', alpha=1.0)

        if fit_res_all is not None:
            x_fit_all, y_fit_all, slope_all, slope_err_all = fit_res_all
            ax1.plot(
                x_fit_all, y_fit_all, linestyle='-', linewidth=2.5, color='black',
                label=f"All Data ${slope_all:.3f} \\pm {slope_err_all:.3f}$"
            )

    # 3. CURVA DI RIFERIMENTO TEORICA (sqrt(Q))
    if all_x_min and all_x_max:
        global_x_min = min(all_x_min)
        global_x_max = max(all_x_max)
        x_ref = np.logspace(np.log10(global_x_min), np.log10(global_x_max), 100)
        y_ref = x_ref**0.5
        ax1.plot(x_ref, y_ref, linestyle='-', color='black', linewidth=1.2, label=r'$\sqrt{Q}$')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$Q$')
    ax1.set_ylabel(r'$I$', color='black')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {save_path}")
    plt.close()

if __name__ == '__main__':
    nb_traders = 20
    kind = 'power'
    exp = '2.0'
    #algorithm = 'maitrier'

    if kind == 'uniform':
        conf = f'{nb_traders}_{kind}'
    else:
        conf = f'{nb_traders}_{kind}_{exp}'

    path = 'meta_' + conf
    PATH = f'meta_{conf}.csv'

    dir_path = os.path.join('database', 'meta') # o 'database\meta'

    #plot_bucci()
    # Set 'linear' per fittare il piano in log-space: log10(I) = log10(Y) + delta*log10(a) + alpha*log10(b)
    MODEL_KIND = 'power_law'
    synthetic_meta = pd.read_csv(
        f'{DIR}\\{PATH}', sep=',', parse_dates=['BeginTime', 'EndTime']
    )
    df_res = synthetic_meta[['MetaVolume', 'TradedVolume', 'MetaImpact', 'NbChild']].copy()
    df_res = df_res[df_res['NbChild'] > 1]
    df_res['a'] = df_res['MetaVolume']
    df_res['b'] = df_res['MetaVolume'] / df_res['TradedVolume']

    grouped, mask_significance = compute_binned_stats(df_res, n_bins=31)
    selected = grouped.loc[mask_significance]
    a_f = selected['a_mean'].to_numpy()
    b_f = selected['b_mean'].to_numpy()
    i_f = selected['impact_mean'].to_numpy()
    xerr_f = selected['a_err'].to_numpy()
    yerr_f = selected['b_err'].to_numpy()
    zerr_f = selected['err'].to_numpy()

    plot_histogram_2d(df_res, save_path=f'images\\part_rate\\part_rate_his_{conf}.png')
    popt, perr, r2 = odr_regression(a_f, b_f, i_f, zerr_f, xerr_f, yerr_f, model_kind=MODEL_KIND)
    
    if MODEL_KIND == 'linear':
        logY_val, delta_val, alpha_val = popt
        logY_err, delta_err, alpha_err = perr
        Y_val = 10**logY_val
        Y_err = Y_val * np.log(10) * logY_err  
        print(f'log10(Y) = {logY_val:.6g} +/- {logY_err:.3g} (Y = {Y_val:.6g} +/- {Y_err:.3g})')
        print(f'delta = {delta_val:.6g} +/- {delta_err:.3g}')
        print(f'alpha = {alpha_val:.6g} +/- {alpha_err:.3g}')
    else:
        for name, val, err in zip(('Y', 'delta', 'alpha'), popt, perr):
            print(f'{name}={val:.6g} +/- {err:.3g}')
            
    print(f'R2 (log scale)={r2:.4f}')

    # Figura 1: Superficie di fit e punti dell'intensità
    plot_fit_surface(grouped, mask_significance, popt, model_kind=MODEL_KIND, save_path=f'images\\part_rate\\part_rate_fit_{conf}.png')

    plot_fit_surface_selected_only(grouped, mask_significance, popt, model_kind=MODEL_KIND, save_path=f'images\\part_rate\\part_rate_fit_selected_{conf}.png')

    # Figura 2: Istogramma 3D dei conteggi con colore basato sull'intensità
    plot_count_bars(grouped, mask_significance, save_path=f'images\\part_rate\\part_rate_counts_{conf}.png')

    plot_impact_volume_by_deciles(
        dir_path=dir_path,
        path=path,
        n_deciles=10,
        min_nb_child=1,
        save_path=f'images/part_rate/impact_volume_deciles_{conf}.png',
        dist_save_path=f'images/part_rate/part_rate_deciles_dist_{conf}.png'
    )