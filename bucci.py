import os
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURAZIONE DEI SET DI DATI
# =============================================================================

@dataclass
class BucciConfig:
    """Descrive un set di dati da plottare in plot_bucci."""
    path: str                          # file csv (meta_*.csv)
    label: str                         # etichetta in legenda
    color: Optional[str] = None        # None -> ciclo colori di matplotlib
    marker: str = '.'                  # marker dei punti binnati
    min_nb_child: int = 1              # tieni solo le righe con NbChild > min_nb_child
    n_bins: int = 101                  # numero di edge dei bin logaritmici
    show_hist: bool = True             # disegna anche l'istogramma del participation rate
    alpha: float = 1.0                 # trasparenza di punti e istogramma


_META_PATTERN = re.compile(
    r"meta_(?:(?P<num_one>1)|(?P<num_others>\d+)_(?P<kind>\w+?)(?:_(?P<exp>[\d.]+))?)\.csv"
)


def label_from_filename(filename):
    """Etichetta automatica dal nome file, con la stessa logica del vecchio plot_bucci."""
    if filename == 'meta_lmf.csv':
        return 'lmf'
    match = _META_PATTERN.search(filename)
    if match is None:
        return os.path.splitext(filename)[0]
    num_traders = match.group('num_one') or match.group('num_others')
    parts = [str(num_traders), match.group('kind') or '', match.group('exp') or '']
    return ' '.join(p for p in parts if p)


def configs_from_dir(directory, skip=('meta_1.csv',), **kwargs):
    """Costruisce una BucciConfig per ogni csv della cartella (comportamento del vecchio
    plot_bucci). I kwargs sono passati a tutte le configurazioni."""
    return [BucciConfig(path=os.path.join(directory, f), label=label_from_filename(f), **kwargs)
            for f in sorted(os.listdir(directory))
            if f.endswith('.csv') and f not in skip]


# =============================================================================
# PLOT
# =============================================================================

def plot_bucci(configurations: Sequence[BucciConfig],
               save_path='images/part_rate/bucci.png',
               xlim=(1e-3, 1.2), ylim=(2e-1, 1e1), show=False):
    """
    Per ogni configurazione: curva I(Q)/sqrt(Q) in funzione del participation rate
    pi = Q/Q_P (pannello alto) e istogramma di pi (pannello basso).
    """
    if not configurations:
        raise ValueError('configurations is empty')

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11
    })

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(8, 10), sharex=True, gridspec_kw={'hspace': 0.08}
    )
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, cfg in enumerate(configurations):
        color = cfg.color or color_cycle[i % len(color_cycle)]

        meta = pd.read_csv(cfg.path, sep=',', parse_dates=['BeginTime', 'EndTime'])
        meta['NbChild'] = pd.to_numeric(meta['NbChild'], errors='coerce')
        df_res = meta[['MetaVolume', 'MetaImpact', 'TradedVolume', 'NbChild']].copy()
        df_res = df_res[df_res['NbChild'] > cfg.min_nb_child].drop(columns=['NbChild'])

        df_res['Ratio'] = df_res['MetaImpact'] / np.sqrt(df_res['MetaVolume'])
        df_res['ParticipationRate'] = df_res['MetaVolume'] / df_res['TradedVolume']
        df_res = df_res.replace([np.inf, -np.inf], np.nan).dropna(subset=['Ratio', 'ParticipationRate'])
        df_res = df_res[df_res['ParticipationRate'] > 0]
        if df_res.empty:
            print(f'[plot_bucci] nessun dato valido per "{cfg.label}" ({cfg.path}), saltato')
            continue

        pr = df_res['ParticipationRate'].to_numpy()
        bins = np.logspace(np.log10(pr.min()), np.log10(pr.max()), cfg.n_bins)

        if cfg.show_hist:
            counts, edges = np.histogram(pr, bins=bins)
            ax_hist.step(np.append(edges[:-1], edges[-1]), np.append(counts, 0),
                         where='post', color=color, alpha=cfg.alpha, linewidth=1.5,
                         label=cfg.label)

        df_res['bin'] = pd.cut(df_res['ParticipationRate'], bins=bins, include_lowest=True)
        grouped = df_res.groupby('bin', observed=True).agg({
            'ParticipationRate': ['mean', 'std'],
            'Ratio': ['mean', 'std', 'count']
        }).dropna()
        grouped.columns = ['PR_mean', 'PR_std', 'Ratio_mean', 'Ratio_std', 'sample_count']

        sqrt_n = np.sqrt(grouped['sample_count'].to_numpy())
        ax_main.errorbar(grouped['PR_mean'].to_numpy(), grouped['Ratio_mean'].to_numpy(),
                         yerr=grouped['Ratio_std'].to_numpy() / sqrt_n,
                         xerr=grouped['PR_std'].to_numpy() / sqrt_n,
                         linestyle='', marker=cfg.marker, color=color, alpha=cfg.alpha,
                         label=cfg.label)

    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.set_xlim(*xlim)
    ax_main.set_ylim(*ylim)
    ax_main.set_ylabel(r'$I(Q) / \sqrt{Q}$')
    ax_main.grid(True, which='both', ls='-')
    ax_main.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), borderaxespad=0)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    ax_hist.set_xscale('log')
    ax_hist.set_xlim(xlim[0], max(xlim[1], 1.0))
    ax_hist.set_xlabel(r'$\pi = Q / Q_P$')
    ax_hist.set_ylabel('Count')
    ax_hist.grid(True, which='both', ls='-')

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()
    print(f'Grafico salvato in: {save_path}')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Opzione 1: tutti i csv di una cartella, etichette ricavate dal nome file (equivalente al vecchio plot_bucci(model)).
    """
    model = ''   # es. 'lmf_emp_tim_param_0.3'
    directory = os.path.join('database', 'meta' if model == '' else f'meta_{model}')
    configurations = configs_from_dir(directory)

    # Opzione 2: elenco esplicito, anche da cartelle diverse e con opzioni per set.
    configurations = [
        BucciConfig(path='database/meta/meta_20_power_2.0.csv', label='maitrier 20 power 2.0'),
        BucciConfig(path='database/meta/meta_lmf.csv', label='lmf', color='black', marker='s'),
        BucciConfig(path='database/meta_tim_0.0/meta_20_uniform.csv', label='tim 0.0', min_nb_child=2, show_hist=False),
    ]

    save_name = 'real_data' if model == '' else f'{model}'
    plot_bucci(configurations, save_path=os.path.join('images', 'bucci', 'singular_model', f'{save_name}.png'))
    """

    # Opzione 2: elenco esplicito, anche da cartelle diverse e con opzioni per set.
    """
    configurations = [
        BucciConfig(path='database/meta_lmf_emp_tim_param_0.3/meta_lmf.csv', label='lmf emp tim param 0.3 lmf'),
        BucciConfig(path='database/meta_lmf_emp_tim_param_0.5/meta_lmf.csv', label='lmf emp tim param 0.5 lmf'),
        BucciConfig(path='database/meta_lmf_emp_tim_param_1.0/meta_lmf.csv', label='lmf emp tim param 1.0 lmf'),
    ]

    plot_bucci(configurations, save_path=os.path.join('images', 'bucci', f'lmf_tim_emp_param_lmf.png'))
    """

    configurations = [
        BucciConfig(path='database/meta/meta_20_power_2.0.csv', label='real data 20 power 2.0'),
        BucciConfig(path='database/meta_lmf_emp_tim_param_0.3/meta_20_power_2.0.csv', label='lmf emp tim param 0.3 20 power 2.0'),
        BucciConfig(path='database/meta_lmf_emp_tim_0.0/meta_20_power_2.0.csv', label='lmf emp tim 0.0 20 power 2.0'),
        BucciConfig(path='database/meta/meta_20_uniform.csv', label='real data 20 uniform', marker='x'),
        BucciConfig(path='database/meta_lmf_emp_tim_param_0.3/meta_20_uniform.csv', label='lmf emp tim param 0.3 20 uniform', marker='x'),
        BucciConfig(path='database/meta_lmf_emp_tim_0.0/meta_20_uniform.csv', label='lmf emp tim 0.0 20 uniform', marker='x'),
    ]

    plot_bucci(configurations, save_path=os.path.join('images', 'bucci', f'maitrier.png'))