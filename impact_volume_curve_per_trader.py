import os
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.odr import ODR, Model, RealData

# ==========================================
# CONFIGURATION
# ==========================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14
})

model = ''                 # '' -> dati reali, altrimenti nome del modello simulato
nb_traders = 20
kind = 'power'
exponent = 2.0

min_nb_child = 1           # si tengono i metaordini con NbChild > min_nb_child
n_bins = 50                # bin logaritmici per trader
min_orders_per_trader = 50 # sotto questa soglia il trader viene saltato
min_bin_count = 2          # un bin entra nel grafico solo con almeno questi metaordini
high_freq_frac = 0.5       # per il fit: bin con count > frac * count massimo del trader
min_fit_points = 3         # punti minimi per eseguire il fit
save_individual = True     # salva anche una figura per ogni trader
show_q_dist = True         # istogramma di Q (asse destro) in ogni pannello

# ==========================================
# PATHS
# ==========================================
dir_name = 'meta' if model == '' else 'meta_' + model
dir_path = os.path.join('database', dir_name)

conf = f'{nb_traders}_{kind}' if kind == 'uniform' else f'{nb_traders}_{kind}_{exponent}'
path = 'meta_' + conf

output_dir = os.path.join('images', 'impact_volume_curve_per_trader',
                          conf if model == '' else model + '_' + conf)
os.makedirs(output_dir, exist_ok=True)

data_path = os.path.join(dir_path, f'{path}_{min_nb_child}.csv')
if not os.path.exists(data_path):
    data_path = os.path.join(dir_path, path + '.csv')

# ==========================================
# HELPERS
# ==========================================
def trader_sort_key(name):
    """'Trader 10' viene dopo 'Trader 2'."""
    m = re.search(r'(\d+)', str(name))
    return (int(m.group(1)), str(name)) if m else (10 ** 9, str(name))


def bin_trader(df):
    """Binning logaritmico di MetaVolume e medie per bin.
    Ritorna (grouped, bins, df con colonna 'bin')."""
    lo, hi = df['MetaVolume'].min(), df['MetaVolume'].max()
    if not (lo > 0 and hi > lo):
        return None, None
    bins = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    df = df.copy()
    df['bin'] = pd.cut(df['MetaVolume'], bins=bins, include_lowest=True)

    g = df.groupby('bin', observed=True).agg(
        vol_mean=('MetaVolume', 'mean'),
        vol_std=('MetaVolume', 'std'),
        count=('MetaVolume', 'count'),
        imp_mean=('MetaImpact', 'mean'),
        imp_std=('MetaImpact', 'std'),
    ).dropna()
    g = g[(g['vol_mean'] > 0) & (g['imp_mean'] > 0) & (g['count'] >= min_bin_count)]
    return g, bins


def fit_loglog(g):
    """Fit ODR in log-log sui bin ad alta frequenza. Ritorna dict o None."""
    if g is None or g.empty:
        return None
    hf = g[g['count'] > high_freq_frac * g['count'].max()]
    if len(hf) < min_fit_points:
        return None

    x, y = hf['vol_mean'].values, hf['imp_mean'].values
    cnt = hf['count'].values
    x_sem = np.where(cnt > 1, hf['vol_std'].values / np.sqrt(cnt), 1e-8)
    y_sem = np.where(cnt > 1, hf['imp_std'].values / np.sqrt(cnt), 1e-8)

    log_x, log_y = np.log10(x), np.log10(y)
    s_lx = np.maximum(x_sem / (x * np.log(10)), 1e-6)
    s_ly = np.maximum(y_sem / (y * np.log(10)), 1e-6)

    odr = ODR(RealData(log_x, log_y, sx=s_lx, sy=s_ly),
              Model(lambda p, t: p[0] * t + p[1]), beta0=[1.0, 0.0])
    out = odr.run()
    slope, intercept = out.beta
    slope_err, intercept_err = out.sd_beta

    y_pred = slope * log_x + intercept
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {'slope': slope, 'slope_err': slope_err,
            'intercept': intercept, 'intercept_err': intercept_err,
            'r2': r2, 'x_min': x.min(), 'x_max': x.max(), 'n_fit_bins': len(hf)}


def draw_trader(ax, df, g, bins, fit, name, compact=False):
    """Disegna dati binnati, fit e riferimento sqrt(Q) su ax. Ritorna ax2 (o None)."""
    ax2 = None
    if show_q_dist:
        ax2 = ax.twinx()
        counts, edges = np.histogram(df['MetaVolume'], bins=bins)
        ax2.bar(edges[:-1], counts, width=np.diff(edges), align='edge',
                color='tab:blue', edgecolor='tab:blue', alpha=0.3,
                label=r'$Q$ frequency')
        ax2.set_yscale('linear')
        if compact:
            ax2.tick_params(axis='y', labelsize=7)

    ax.plot(g['vol_mean'], g['imp_mean'], ls='', marker='o', ms=4,
            color='tab:blue', label='Data', zorder=3)

    if fit is not None:
        xf = np.logspace(np.log10(fit['x_min']), np.log10(fit['x_max']), 100)
        ax.plot(xf, 10 ** fit['intercept'] * xf ** fit['slope'], '--', lw=1.8,
                color='tab:red', zorder=4,
                label=rf"Fit: ${fit['slope']:.3f} \pm {fit['slope_err']:.3f}$"
                      + '\n' + rf"($R^2 = {fit['r2']:.3f}$)")

    xr = np.logspace(np.log10(g['vol_mean'].min() / 2), np.log10(g['vol_mean'].max() * 2), 100)
    ax.plot(xr, xr ** 0.5, '-', color='black', lw=1.0, label=r'$\sqrt{Q}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f"{name}  (n={len(df):,})")
    ax.set_xlabel(r'$Q$')
    ax.set_ylabel(r'$I$')
    ax.grid(True, which='both', ls='-', alpha=0.2)
    return ax2


# ==========================================
# LOAD DATA
# ==========================================
meta = pd.read_csv(data_path, sep=',', parse_dates=['BeginTime', 'EndTime'])
if 'trader' not in meta.columns:
    raise KeyError("La colonna 'trader' non è presente nel CSV: serve l'output di generate(), non di generate_slim().")

meta = meta[(meta['MetaVolume'] > 0) & (meta['NbChild'] > min_nb_child)]
meta = meta[['trader', 'MetaVolume', 'MetaImpact', 'NbChild']].copy()

traders = sorted(meta['trader'].unique(), key=trader_sort_key)
print(f"Caricati {len(meta):,} metaordini da {data_path} | {len(traders)} trader")

# ==========================================
# PER-TRADER ANALYSIS
# ==========================================
results = []
per_trader = {}   # name -> (df, g, bins, fit)

for name in traders:
    df = meta[meta['trader'] == name]
    if len(df) < min_orders_per_trader:
        print(f"[{name}] saltato: solo {len(df)} metaordini")
        continue

    g, bins = bin_trader(df)
    if g is None or g.empty:
        print(f"[{name}] saltato: nessun bin valido")
        continue

    fit = fit_loglog(g)
    per_trader[name] = (df, g, bins, fit)

    row = {
        'trader': name,
        'n_metaorders': len(df),
        'share_metaorders': len(df) / len(meta),
        'total_volume': df['MetaVolume'].sum(),
        'mean_nb_child': df['NbChild'].mean(),
        'mean_Q': df['MetaVolume'].mean(),
        'mean_impact': df['MetaImpact'].mean(),
    }
    if fit is not None:
        row.update({k: fit[k] for k in ('slope', 'slope_err', 'intercept', 'intercept_err', 'r2', 'n_fit_bins')})
        print(f"[{name}] n={len(df):>8,} | slope {fit['slope']:.4f} ± {fit['slope_err']:.4f} "
              f"| intercept {fit['intercept']:.4f} | R2 {fit['r2']:.4f}")
    else:
        print(f"[{name}] n={len(df):>8,} | fit non eseguibile (bin insufficienti)")
    results.append(row)

summary = pd.DataFrame(results)
summary_path = os.path.join(output_dir, 'summary_per_trader.csv')
summary.to_csv(summary_path, index=False)
print(f"Riepilogo salvato in: {summary_path}")

# ==========================================
# Griglia, un pannello per trader
# ==========================================
names = list(per_trader.keys())
if names:
    ncols = min(4, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), squeeze=False)

    for ax, name in zip(axes.ravel(), names):
        df, g, bins, fit = per_trader[name]
        draw_trader(ax, df, g, bins, fit, name, compact=True)
        ax.legend(loc='upper left', fontsize=7)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)

    plt.tight_layout()
    grid_path = os.path.join(output_dir, 'grid_per_trader.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Griglia salvata in: {grid_path}")

# ==========================================
# Pendenza del fit per trader (asse destro: R^2)
# ==========================================
if 'slope' in summary.columns and summary['slope'].notna().any():
    s = summary.dropna(subset=['slope'])
    xpos = np.arange(len(s))

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(s)), 5))
    ax_r2 = ax.twinx()

    # Asse sinistro: esponente del fit
    ax.errorbar(xpos, s['slope'], yerr=s['slope_err'], fmt='o', capsize=3,
                color='tab:blue', label='Fitted exponent', zorder=3)
    ax.axhline(0.5, color='black', lw=1.2, label=r'$\sqrt{Q}$ (0.5)')
    ax.set_xticks(xpos)
    ax.set_xticklabels(s['trader'], rotation=60, ha='right')
    ax.set_ylabel('Fitted exponent')
    ax.tick_params(axis='y')
    ax.grid(True, alpha=0.3)

    # Asse destro: R^2 del fit
    ax_r2.plot(xpos, s['r2'], ls='', marker='s', ms=6, color='tab:orange', label=r'$R^2$', zorder=4)
    ax_r2.set_ylabel(r'$R^2$')
    ax_r2.tick_params(axis='y')
    r2_lo = min(0.0, float(np.nanmin(s['r2']))) if s['r2'].notna().any() else 0.0
    ax_r2.set_ylim(r2_lo - 0.02, 1.02)

    # Legenda unica per entrambi gli assi
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='best')

    plt.tight_layout()
    slope_path = os.path.join(output_dir, 'slope_per_trader.png')
    plt.savefig(slope_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pendenze salvate in: {slope_path}")

# ==========================================
# INDIVIDUAL FIGURES
# ==========================================
if save_individual:
    ind_dir = os.path.join(output_dir, 'individual')
    os.makedirs(ind_dir, exist_ok=True)
    for name, (df, g, bins, fit) in per_trader.items():
        fig, ax1 = plt.subplots(figsize=(10, 7))
        ax2 = draw_trader(ax1, df, g, bins, fit, name)
        h1, l1 = ax1.get_legend_handles_labels()
        if ax2 is not None:
            ax2.set_ylabel('Frequency')
            h2, l2 = ax2.get_legend_handles_labels()
            h1, l1 = h1 + h2, l1 + l2
        ax1.legend(h1, l1, loc='upper left', bbox_to_anchor=(1.12, 1), frameon=True)
        plt.tight_layout()
        fname = re.sub(r'\W+', '_', str(name)).strip('_') + '.png'
        plt.savefig(os.path.join(ind_dir, fname), bbox_inches='tight')
        plt.close(fig)
    print(f"Figure individuali salvate in: {ind_dir}")