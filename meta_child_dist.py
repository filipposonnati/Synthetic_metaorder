import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import powerlaw

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
BASE_DIR = Path('database')
REAL_SUBDIR = 'meta_child_dist'

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 18, 'axes.labelsize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_filename(nb_traders: int, kind: str, exponent: float) -> str:
    if nb_traders == 1:
        return "1"
    return f"{nb_traders}_{kind}" if kind == 'uniform' else f"{nb_traders}_{kind}_{exponent}"

def get_pdf_ccdf(data: np.ndarray):
    """Calcola sia la PDF che la CCDF per dati interi."""
    bins = np.arange(1, max(data) + 2)
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    x_vals = bin_edges[:-1]
    ccdf = np.cumsum(counts[::-1])[::-1]
    return x_vals, counts, ccdf

def load_data(results_dir: Path, nb_traders: int = None, kind: str = None, exponent: float = None, stem: str = None):
    """Carica i dati .npz dato un insieme di parametri o direttamente uno stem."""
    if stem is None:
        stem = build_filename(nb_traders, kind, exponent)
    filepath = results_dir / f"realizations_{stem}.npz"

    if not filepath.exists():
        raise FileNotFoundError(f"File non trovato: '{filepath}'")

    archive = np.load(filepath)
    iterations = int(archive['iterations'])
    data = archive['data'].ravel()
    print(f"[load] {filepath} | iter={iterations} | n={len(data):,}")
    return iterations, data

# ---------------------------------------------------------------------------
# Fit e Plot singolo
# ---------------------------------------------------------------------------
def dist_fit(data: np.ndarray, iterations: int, filename: str) -> None:
    print(len(data) / iterations)

    fit = powerlaw.Fit(data, discrete=True)
    tpl, pl, ln = fit.truncated_power_law, fit.power_law, fit.lognormal

    R_tpl_vs_pl, p_tpl_vs_pl = fit.distribution_compare('truncated_power_law', 'power_law')
    R_tpl_vs_ln, p_tpl_vs_ln = fit.distribution_compare('truncated_power_law', 'lognormal')

    print("=" * 55 + "\n  RISULTATI DEL FIT\n" + "=" * 55)
    print(f"\n--- Truncated Power Law:\n  alpha: {tpl.alpha:.4f}\n  Lambda: {tpl.Lambda:.6f}\n  x_min: {tpl.xmin}")
    print(f"\n--- Power Law pura:\n  alpha: {pl.alpha:.4f}\n  x_min: {pl.xmin}")
    print(f"\n--- Lognormal:\n  mu: {ln.mu:.4f}\n  sigma: {ln.sigma:.4f}")
    print(f"\n--- Likelihood Ratio Tests:\n  vs Power Law: R = {R_tpl_vs_pl:+.3f}, p = {p_tpl_vs_pl:.4f}")
    print(f"  vs Lognormal: R = {R_tpl_vs_ln:+.3f}, p = {p_tpl_vs_ln:.4f}\n" + "=" * 55)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.35)

    models = [
        (fit.truncated_power_law, f'Trunc. PL α={tpl.alpha:.2f}, Λ={tpl.Lambda:.4f}', '-', 2.2),
        (fit.power_law, f'Power Law α={pl.alpha:.2f}', '--', 1.6),
        (fit.lognormal, f'Lognormal μ={ln.mu:.2f} σ={ln.sigma:.2f}', ':', 1.6)
    ]

    for ax, title, plot_method in [(ax1, 'PDF', 'plot_pdf'), (ax2, 'CCDF', 'plot_ccdf')]:
        ax.tick_params(labelsize=9)
        ax.set_title(title, fontsize=11, pad=10)
        ax.grid(True, lw=0.5, ls='--', alpha=0.6)
        ax.set_xlabel('x')
        ax.set_ylabel('P(x)' if title == 'PDF' else 'P(X ≥ x)')

        getattr(fit, plot_method)(lw=0, marker='o', ms=4, label='Dati empirici', ax=ax)
        for model, label, ls, lw in models:
            getattr(model, label, ls, lw in models)
            getattr(model, plot_method)(lw=lw, ls=ls, label=label if title == 'PDF' else label.split()[0], ax=ax)

        ax.axvline(tpl.xmin, color='grey', lw=1, ls=':', alpha=0.5, label=f'x_min = {tpl.xmin}')
        ax.legend(fontsize=7.5, framealpha=0.3)

    Path('images').mkdir(exist_ok=True)
    fig.savefig(f'images/dist_meta_child_{filename}.png', dpi=300)
    plt.show()

# ---------------------------------------------------------------------------
# Plot di tutte le configurazioni
# ---------------------------------------------------------------------------
def plot_all(name: str, configs: list[dict] = None, lmf_init_conf=None, comparison=False, show_ratio=False, x_log=False) -> None:
    dir_path = BASE_DIR / name
    all_data = {}

    if configs:
        for cfg in configs:
            stem = build_filename(cfg['nb_traders'], cfg['kind'], cfg['exponent'])
            try:
                all_data[stem] = load_data(dir_path, stem=stem)
            except FileNotFoundError as e:
                print(f"[warn] {e}")
    else:
        for f in sorted(dir_path.glob("realizations_*.npz")):
            stem = f.stem.replace('realizations_', '')
            all_data[stem] = load_data(dir_path, stem=stem)

    colors, markers = plt.cm.tab10.colors, ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    pdf_dict = {}

    num_plots = 3 if show_ratio else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

    if show_ratio:
        ax_pdf, ax_ccdf, ax_ratio = axes
        plot_configs = [(ax_pdf, 'PDF', 'P(x)'), (ax_ccdf, 'CCDF', 'P(X ≥ x)'), (ax_ratio, 'PDF Ratios', 'Ratio')]
    else:
        ax_pdf, ax_ccdf = axes
        ax_ratio = None
        plot_configs = [(ax_pdf, 'PDF', 'P(x)'), (ax_ccdf, 'CCDF', 'P(X ≥ x)')]

    for ax, title, ylabel in plot_configs:
        if x_log:
            ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('x')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, lw=0.5, ls='--', alpha=0.6)

    for i, (stem, (_, data)) in enumerate(all_data.items()):
        kw = dict(ls='', marker=markers[i % len(markers)], ms=3, color=colors[i % len(colors)], label=stem)
        x_vals, counts, ccdf = get_pdf_ccdf(data)

        traders_key = stem.split('_')[0]
        pdf_dict[traders_key] = (x_vals, counts)

        ax_pdf.plot(x_vals, counts, **kw)
        ax_ccdf.plot(x_vals, ccdf, **kw)

    if show_ratio and ax_ratio is not None:
        ratios_to_compute = [('1', '4'), ('1', '20'), ('4', '20')]
        ratio_colors = ['purple', 'orange', 'green']

        for (num_key, den_key), r_color in zip(ratios_to_compute, ratio_colors):
            if num_key in pdf_dict and den_key in pdf_dict:
                x_num, pdf_num = pdf_dict[num_key]
                x_den, pdf_den = pdf_dict[den_key]

                common_x = np.intersect1d(x_num, x_den)
                val_num = pdf_num[np.isin(x_num, common_x)]
                val_den = pdf_den[np.isin(x_den, common_x)]

                valid_mask = val_den > 0
                ax_ratio.plot(
                    common_x[valid_mask], val_num[valid_mask] / val_den[valid_mask],
                    ls='', marker='o', ms=4, color=r_color, label=f'PDF({num_key}) / PDF({den_key})'
                )

    if lmf_init_conf and comparison:
        from lmf import simulate_lmf_lambda
        if lmf_init_conf[0] == 'lmf':
            alpha, n = lmf_init_conf[1], lmf_init_conf[2]
            dist = (np.random.pareto(alpha, size=10_000_000) + 1).astype(int)
            lmf_label = f"LMF (α={alpha}, N={n})"
        elif lmf_init_conf[0] == 'lmf_lambda':
            alpha, lam = lmf_init_conf[1], lmf_init_conf[2]
            _, _, storico = simulate_lmf_lambda(alpha, lam, total_steps=10_000_000)
            dist = np.array([m['lunghezza_iniziale'] for m in storico])
            lmf_label = f"LMF λ (α={alpha}, λ={lam})"

        if 'dist' in locals() and len(dist) > 0:
            x_v, cnts, ccdf_l = get_pdf_ccdf(dist)
            kw_lmf = dict(ls='', marker='.', lw=1.5, color='black', label=lmf_label)
            ax_pdf.plot(x_v, cnts, **kw_lmf)
            ax_ccdf.plot(x_v, ccdf_l, **kw_lmf)

    axes_to_loop = axes if hasattr(axes, '__iter__') else [axes]
    for ax in axes_to_loop:
        ax.legend(framealpha=0.4)

    plt.tight_layout()
    output_dir = Path(f'images/meta_child_dist')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'{name}{"_log" if x_log else ""}.png', dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# Griglia di confronto tra dati reali e simulazioni
# ---------------------------------------------------------------------------
def compare_all_distributions_grid(base_dir: Path, real_subdir: str, configs: list[dict] = None, x_log=False) -> None:
    real_dir = base_dir / real_subdir
    sim_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('meta_child_dist') and d != real_dir])

    if not real_dir.exists():
        raise FileNotFoundError(f"Directory reale '{real_dir}' non trovata.")
    if not sim_dirs:
        print("[compare] Nessuna variante simulata trovata.")
        return

    target_stems = None # = [build_filename(c['nb_traders'], c['kind'], c['exponent']) for c in configs] if configs else None

    def _load_stems(directory: Path):
        loaded = {}
        if target_stems:
            for stem in target_stems:
                f_path = directory / f"realizations_{stem}.npz"
                if f_path.exists():
                    loaded[stem] = np.load(f_path)['data'].ravel()
        else:
            for f in directory.glob("realizations_*.npz"):
                stem = f.stem.replace('realizations_', '')
                loaded[stem] = np.load(f)['data'].ravel()
        return loaded

    real_data = _load_stems(real_dir)
    sim_datasets = {d.name: _load_stems(d) for d in sim_dirs}

    stems = sorted(real_data.keys())
    if not stems:
        print("[compare] Nessun dataset corrispondente trovato.")
        return

    variant_names = list(sim_datasets.keys())
    colors = plt.cm.tab10.colors
    styles = {
        name: dict(ls='', marker='o', ms=3.5, color=colors[i % len(colors)], alpha=0.85, label=name)
        for i, name in enumerate(variant_names)
    }
    real_style = dict(ls='', marker='o', ms=3.5, color='black', alpha=0.9, label=real_subdir)

    n = len(stems)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.5 * n), squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    for row, stem in enumerate(stems):
        ax_pdf, ax_ccdf = axes[row, 0], axes[row, 1]

        for ax, title, ylabel in [(ax_pdf, 'PDF', f'{stem.replace("_", " ")}\nP(x)'), (ax_ccdf, 'CCDF', r'$P(X \geq x)$')]:
            if x_log:
                ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('x')
            ax.set_ylabel(ylabel)
            ax.grid(True, lw=0.4, ls='--', alpha=0.6)
            if row == 0:
                ax.set_title(title, fontsize=12, pad=8)

        # Plot dati reali
        x_p, cnt_r, ccdf_r = get_pdf_ccdf(real_data[stem])
        ax_pdf.plot(x_p, cnt_r, **real_style)
        ax_ccdf.plot(x_p, ccdf_r, **real_style)

        # Plot simulazioni
        for v_name, sim_dict in sim_datasets.items():
            if stem in sim_dict:
                x_ps, cnt_s, ccdf_s = get_pdf_ccdf(sim_dict[stem])
                ax_pdf.plot(x_ps, cnt_s, **styles[v_name])
                ax_ccdf.plot(x_ps, ccdf_s, **styles[v_name])

        if row == 0:
            ax_pdf.legend(fontsize=8, framealpha=0.4, loc='upper right')
            ax_ccdf.legend(fontsize=8, framealpha=0.4, loc='upper right')

    out_fname = 'selected' if target_stems else 'all'
    out_path = Path(f'images/meta_child_dist/compare_real_vs_simulated_{out_fname}.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[save] {out_path}")
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_trader_distributions(configuration: str, min_orders_per_trader: int = 10, x_log: bool = True) -> None:
    """
    Legge il file database/meta/meta_{configuration}.csv e calcola la distribuzione
    della dimensione degli ordini (NbChild) separatamente per ciascun trader.
    """
    csv_path = Path("database") / "meta" / f"meta_{configuration}.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"File CSV non trovato: '{csv_path}'")
        
    # Caricamento dei soli campi utili per ottimizzare la memoria
    df = pd.read_csv(csv_path, usecols=['trader', 'NbChild'])
    
    # Raggruppamento per trader
    grouped = df.groupby('trader')
    
    fig, (ax_pdf, ax_ccdf) = plt.subplots(1, 2, figsize=(13, 6))
    
    colors = plt.cm.tab20.colors
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    
    for i, (trader_id, group) in enumerate(grouped):
        nb_child_data = group['NbChild'].values
        
        # Filtro per trader con un numero sufficiente di dati
        if len(nb_child_data) < min_orders_per_trader:
            continue
            
        x_vals, counts, ccdf = get_pdf_ccdf(nb_child_data)
        
        kw = dict(
            ls='', 
            marker=markers[i % len(markers)], 
            ms=4, 
            color=colors[i % len(colors)], 
            alpha=0.8,
            label=f'{trader_id}'
        )
        
        ax_pdf.plot(x_vals, counts, **kw)
        ax_ccdf.plot(x_vals, ccdf, **kw)

    for ax, title, ylabel in [(ax_pdf, 'PDF', 'P(n)'), (ax_ccdf, 'CCDF', 'P(N ≥ n)')]:
        if x_log:
            ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('n')
        ax.set_ylabel(ylabel)
        #ax.set_title(f'{title} - Traders in Config: {configuration}')
        ax.grid(True, lw=0.5, ls='--', alpha=0.6)
        #ax.legend(fontsize=8, framealpha=0.4, loc='best')

    plt.tight_layout()
    
    output_dir = Path('images/meta_child_dist')
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"traders_{configuration}{'_log' if x_log else ''}.png"
    
    fig.savefig(out_path, dpi=300)
    print(f"[save] Grafico salvato in: {out_path}")
    plt.close()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    my_configs = [
        {'nb_traders': 1, 'kind': 'uniform', 'exponent': 0.0},
        {'nb_traders': 4, 'kind': 'uniform', 'exponent': 0.0},
        {'nb_traders': 4, 'kind': 'power', 'exponent': 2.0},
        {'nb_traders': 20, 'kind': 'uniform', 'exponent': 0.0},
        {'nb_traders': 20, 'kind': 'power', 'exponent': 2.0}
    ]

    plot_all('meta_child_dist', configs=my_configs)
    plot_all('meta_child_dist', configs=my_configs, x_log=True)





    my_configs = [
        {'nb_traders': 1, 'kind': 'uniform', 'exponent': 0.0},
        {'nb_traders': 4, 'kind': 'uniform', 'exponent': 0.0},
        {'nb_traders': 20, 'kind': 'uniform', 'exponent': 0.0}
    ]

    # Esecuzione di plot_all passando la lista di configurazioni
    #plot_all('meta_child_dist_lmf_lambda_1.5_0.3', configs=my_configs, lmf_init_conf=['lmf_lambda', 1.5, 0.3], comparison=True)





    """
    my_configs = [
        {'nb_traders': 20, 'kind': 'uniform', 'exponent': 0.0},
        {'nb_traders': 20, 'kind': 'power_law', 'exponent': 2.0}
    ]

    # Esecuzione della comparazione su griglia filtrando per le stesse configurazioni
    compare_all_distributions_grid(BASE_DIR, REAL_SUBDIR, configs=my_configs)
    """

    # Esempio: legge 'database/meta/meta_4_uniform.csv' o il nome esatto della configurazione
    config_scelta = "20_power_2.0"
    
    # Esegui l'analisi per singolo trader
    #plot_trader_distributions(config_scelta, min_orders_per_trader=20)
    #plot_trader_distributions(config_scelta, min_orders_per_trader=20, x_log=False)