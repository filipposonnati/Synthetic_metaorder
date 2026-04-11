import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import methods
import os
from os import listdir
import powerlaw
import matplotlib.gridspec as gridspec

def dist_fit(data, filename):
    print(len(data) / iterations)

    # --- Fit con la libreria powerlaw ---
    # Il parametro discrete=True è fondamentale se NbChild sono numeri interi
    fit = powerlaw.Fit(data, discrete=True)

    tpl = fit.truncated_power_law
    alpha   = tpl.alpha       # esponente
    Lambda  = tpl.Lambda      # parametro di taglio (decay rate)
    x_min   = tpl.xmin        # x_min stimato

    # --- Power Law pura (per confronto) ---
    pl      = fit.power_law
    alpha_pl = pl.alpha
    x_min_pl = pl.xmin

    # --- Lognormal (per confronto) ---
    ln       = fit.lognormal
    mu_ln    = ln.mu
    sigma_ln = ln.sigma


    # ── LIKELIHOOD RATIO TESTS ───────────────────────────────────────────────────
    # R > 0  →  prima distribuzione preferita; p < 0.05 → confronto significativo
    R_tpl_vs_pl,  p_tpl_vs_pl  = fit.distribution_compare('truncated_power_law', 'power_law')
    R_tpl_vs_ln,  p_tpl_vs_ln  = fit.distribution_compare('truncated_power_law', 'lognormal')

    # ── STAMPA PARAMETRI ─────────────────────────────────────────────────────────
    print("=" * 55)
    print("  RISULTATI DEL FIT")
    print("=" * 55)
    print(f"\n{'--- Truncated Power Law':}")
    print(f"  alpha  (esponente)   : {alpha:.4f}")
    print(f"  Lambda (decay rate)  : {Lambda:.6f}")
    print(f"  x_min                : {x_min}")

    print(f"\n{'--- Power Law pura':}")
    print(f"  alpha                : {alpha_pl:.4f}")
    print(f"  x_min                : {x_min_pl}")

    print(f"\n{'--- Lognormal':}")
    print(f"  mu                   : {mu_ln:.4f}")
    print(f"  sigma                : {sigma_ln:.4f}")


    print(f"\n{'--- Likelihood Ratio Tests (vs Truncated PL)':}")
    print(f"  TPL vs Power Law     : R = {R_tpl_vs_pl:+.3f},  p = {p_tpl_vs_pl:.4f}")
    print(f"  TPL vs Lognormal     : R = {R_tpl_vs_ln:+.3f},  p = {p_tpl_vs_ln:.4f}")
    print("=" * 55)

    # ── GRAFICO ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 5.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    def _style_ax(ax, title):
        ax.tick_params(labelsize=9)
        ax.set_title(title, fontsize=11, pad=10)
        ax.grid(True, linewidth=0.5, linestyle='--', alpha=0.6)

    # ── Pannello 1: PDF (log-log) ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    _style_ax(ax1, 'PDF')

    fit.plot_pdf(linewidth=0, marker='o',
                markersize=4, filename='Dati empirici', ax=ax1)

    fit.truncated_power_law.plot_pdf(linewidth=2.2,
                                    filename=f'Trunc. PL  α={alpha:.2f}, Λ={Lambda:.4f}', ax=ax1)
    fit.power_law.plot_pdf(linewidth=1.6, linestyle='--',
                            filename=f'Power Law  α={alpha_pl:.2f}', ax=ax1)
    fit.lognormal.plot_pdf(linewidth=1.6, linestyle=':',
                            filename=f'Lognormal  μ={mu_ln:.2f} σ={sigma_ln:.2f}', ax=ax1)

    ax1.axvline(x_min, color='white', linewidth=1, linestyle=':', alpha=0.5,
                filename=f'x_min = {x_min}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('P(x)')
    legend1 = ax1.legend(fontsize=7.5, framealpha=0.3)

    # ── Pannello 2: CCDF (log-log) ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2, 'CCDF')

    fit.plot_ccdf(linewidth=0, marker='o',
                markersize=4, filename='Dati empirici', ax=ax2)

    fit.truncated_power_law.plot_ccdf(linewidth=2.2,
                                    filename='Trunc. PL', ax=ax2)
    fit.power_law.plot_ccdf(linewidth=1.6, linestyle='--',
                            filename='Power Law', ax=ax2)
    fit.lognormal.plot_ccdf(linewidth=1.6, linestyle=':',
                            filename='Lognormal', ax=ax2)

    ax2.axvline(x_min, color='white', linewidth=1, linestyle=':', alpha=0.5,
                filename=f'x_min = {x_min}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('P(X ≥ x)')
    ax2.legend(fontsize=7.5, framealpha=0.3)

    # Salvataggio
    fig.savefig(f'images\\dist_meta_child_{filename}.png', dpi=300)

def generate(data_dir, iterations, nb_traders, kind, exponent):
    paths = np.array(listdir(data_dir))
    meta_tot = pd.DataFrame()

    for i in range(iterations):
        print(f'Iteration: {i + 1}/{iterations}')
        for path in paths:
            l = len(meta_tot)
            meta, _ = methods.generate(path, nb_traders, kind, exponent, l, data_dir)
            meta_tot = pd.concat([meta_tot, meta['NbChild']])

    data = meta_tot['NbChild'].to_numpy()
    return data

# --- Style Configuration ---
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

# --- Configuration ---
data_dir = 'database\\data'
iterations = 1
nb_traders = 4
kind = 'uniform'
exponent = 2.0

if nb_traders == 1:
    filename = "1"
elif kind == 'uniform':
    filename = f"{nb_traders}_{kind}"
else:
    filename = f"{nb_traders}_{kind}_{exponent}"

data = generate(data_dir, iterations, nb_traders, kind, exponent)

dist_fit(data, filename)