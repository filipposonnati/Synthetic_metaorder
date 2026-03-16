import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf
import powerlaw
from collections import Counter
from fbm import fgn
from series_creation import generate_ffm, generate_arfima, generate_fgn

def compute_runs(sign_series):
    """
    Computes the lengths of consecutive runs of identical symbols in a binary series.

    A "run" is a maximal sequence of consecutive identical values (e.g. [+1,+1,+1]
    is a run of length 3). Run-length statistics carry information about the
    correlation structure of the series: long-range correlated signals tend to
    produce heavy-tailed run-length distributions.

    Parameters
    ----------
    sign_series : array of {+1, -1} — the binary input series

    Returns
    -------
    run_lengths : array of int — length of each consecutive run
    """
    # Detect positions where the value changes (True = a change occurred)
    changes = np.diff(sign_series) != 0

    # Indices of change points (i.e. where each run ends)
    change_indices = np.where(changes)[0]

    # Prepend -1 and append the last index so that np.diff gives the full run lengths,
    # including the first and last run
    run_lengths = np.diff(np.concatenate([[-1], change_indices, [len(sign_series) - 1]]))

    return run_lengths

def dist_plot(runs, filename, min = False):
    # ── FIT ──────────────────────────────────────────────────────────────────────
    if min:
        fit = powerlaw.Fit(runs, discrete=True)
    else:
        fit = powerlaw.Fit(runs, discrete=True, xmin=1.0)

    # --- Truncated Power Law ---
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

    fit.plot_pdf(linewidth=0, marker='o', markersize=4, label='Empirical Data', ax=ax1, linear_bins=False)

    fit.truncated_power_law.plot_pdf(linewidth=2.2,
                                    label=f'Trunc. PL  α={alpha:.3f}, Λ={Lambda:.3f}', ax=ax1)
    fit.power_law.plot_pdf(linewidth=1.6, linestyle='--',
                            label=f'Power Law  α={alpha_pl:.3f}', ax=ax1)
    fit.lognormal.plot_pdf(linewidth=1.6, linestyle=':',
                            label=f'Lognormal  μ={mu_ln:.3f} σ={sigma_ln:.3f}', ax=ax1)

    ax1.axvline(x_min, color='white', linewidth=1, linestyle=':', alpha=0.5,
                label=f'x_min = {x_min}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('P(x)')
    legend1 = ax1.legend(fontsize=7.5, framealpha=0.3)

    # ── Pannello 2: CCDF (log-log) ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2, 'CCDF')

    fit.plot_ccdf(linewidth=0, marker='o',
                markersize=4, label='Empirical Data', ax=ax2)

    fit.truncated_power_law.plot_ccdf(linewidth=2.2,
                                    label='Trunc. PL', ax=ax2)
    fit.power_law.plot_ccdf(linewidth=1.6, linestyle='--',
                            label='Power Law', ax=ax2)
    fit.lognormal.plot_ccdf(linewidth=1.6, linestyle=':',
                            label='Lognormal', ax=ax2)

    ax2.axvline(x_min, color='white', linewidth=1, linestyle=':', alpha=0.5,
                label=f'x_min = {x_min}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('P(X ≥ x)')
    ax2.legend(fontsize=7.5, framealpha=0.3)

    # ── Annotazione LRT ──────────────────────────────────────────────────────────
    """
    lrt_text = (
        f"LRT  TPL vs PL   R={R_tpl_vs_pl:+.2f}  p={p_tpl_vs_pl:.3f}\n"
        f"LRT  TPL vs LN   R={R_tpl_vs_ln:+.2f}  p={p_tpl_vs_ln:.3f}"
    )
    fig.text(0.5, 0.01, lrt_text, ha='center', va='bottom',
            fontsize=8,
            fontfamily='monospace')
    """

    if min:
        plt.savefig(f'images\\{filename}_min.png', dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    else:
        plt.savefig(f'images\\{filename}.png', dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())

if __name__ == "__main__":
    n = 100_000_000
    gamma = 0.3
    print(f'Processing gamma = {gamma:.2f}')

    d = 0.5 - 0.5 * gamma
    H = d + 0.5

    # Series generation
    # Generate a continuous long-memory series via FFM, then binarize by sign
    #series = generate_ffm(n, gamma)
    #series = generate_arfima(n, d)
    series = generate_fgn(n, H)

    #standard library approach
    #series = fgn(n=1_000_000, hurst=H, length=1, method='daviesharte')

    sign_series = np.where(series >= 0, 1, -1)

    # Compute run lengths of the binary series
    runs = compute_runs(sign_series)

    dist_plot(runs, f'series_analysis_fgn\\powerlaw_fit_{gamma}')
    dist_plot(runs, f'series_analysis_fgn\\powerlaw_fit_{gamma}', min = True)
    #fit_plot_manual(runs, f'series\\powerlaw_fit_{gamma}_manual')