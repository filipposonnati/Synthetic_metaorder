import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf
import powerlaw
from series_creation import generate_ffm, generate_arfima

import matplotlib.gridspec as gridspec

def linear_func(x, slope, intercept):
    """
    Standard linear function y = slope * x + intercept.
    Used for fitting in log-log space, where power-law relationships
    F ~ x^alpha become straight lines: log F = alpha * log x + const.
    """
    return intercept + slope * x

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

def fit_plot(gamma_target, n_total=100000000, min = False):
    """
    Generates a long-memory binary series with target exponent gamma_target,
    then performs two complementary analyses:
      1. ACF analysis — fits a power law to the autocorrelation function tail
         to recover the correlation exponent gamma.
      2. Run-length CCDF analysis — fits a power law to the complementary
         cumulative distribution of run lengths to recover the run exponent mu.

    Both analyses produce a dual plot and return the fitted exponents.

    Parameters
    ----------
    gamma_target : float — target ACF decay exponent (C(tau) ~ tau^-gamma)
    n_total      : int   — length of the series to generate (default: 10 million)
    lim          : float — lower cutoff for the CCDF power-law fit (default: 1.0)

    Returns
    -------
    gamma_fit : float — ACF exponent estimated from the tail fit
    mu_ccdf   : float — run-length exponent estimated from the CCDF fit
    """

    # Series generation
    # Generate a continuous long-memory series via FFM, then binarize by sign
    d = 0.5 - 0.5 * gamma_target
    series = generate_ffm(n_total, d)
    sign_series = np.where(series >= 0, 1, -1)

    # (The DFA block below is commented out but left for reference.
    #  It was an alternative way to estimate gamma via Detrended Fluctuation Analysis.)
    """
    print('fit dfa')
    scales, F_n = bin_gen.compute_dfa(sign_series, min_scale=10, max_scale=10000, num_scales=40)
    log_scales = np.log(scales)
    log_Fn = np.log(F_n)
    alpha, intercept = np.polyfit(log_scales, log_Fn, 1)
    gamma_estimated = 2.0 - 2.0 * alpha
    print(gamma_estimated)
    """

    # Compute ACF up to lag 1000 using FFT for efficiency
    acf_values = acf(sign_series, nlags=1000, fft=True)

    # Compute run lengths of the binary series
    runs = compute_runs(sign_series)

    """
    # (Histogram bins — computed but not directly used in the final plot below)
    bins = np.linspace(1.0, 1000.0, 1000)
    hist, _ = np.histogram(runs, bins=bins)

    # Lag axis for the ACF (starts at lag 1 to align with the power-law fit)
    x_axis = np.arange(1, len(acf_values) + 1)

    # --- 3. ACF power-law fit ---
    # Fit only the tail of the ACF (lags >= 10) where the power-law behavior dominates.
    # We also restrict to positive ACF values to avoid log of non-positive numbers.
    mask_acf_tail = (x_axis >= 10) & (acf_values > 0)

    # Fit log(ACF) = slope * log(lag) + intercept in log-log space
    popt_acf_t, _ = curve_fit(
        linear_func,
        np.log(x_axis[mask_acf_tail]),
        np.log(acf_values[mask_acf_tail])
    )
    # The ACF exponent gamma is the negative of the slope (C(tau) ~ tau^-gamma)
    gamma_fit = -popt_acf_t[0]
    """

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

    fit.plot_pdf(linewidth=0, marker='o',
                markersize=4, label='Dati empirici', ax=ax1)

    fit.truncated_power_law.plot_pdf(linewidth=2.2,
                                    label=f'Trunc. PL  α={alpha:.2f}, Λ={Lambda:.4f}', ax=ax1)
    fit.power_law.plot_pdf(linewidth=1.6, linestyle='--',
                            label=f'Power Law  α={alpha_pl:.2f}', ax=ax1)
    fit.lognormal.plot_pdf(linewidth=1.6, linestyle=':',
                            label=f'Lognormal  μ={mu_ln:.2f} σ={sigma_ln:.2f}', ax=ax1)

    ax1.axvline(x_min, color='white', linewidth=1, linestyle=':', alpha=0.5,
                label=f'x_min = {x_min}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('P(x)')
    legend1 = ax1.legend(fontsize=7.5, framealpha=0.3)

    # ── Pannello 2: CCDF (log-log) ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2, 'CCDF')

    fit.plot_ccdf(linewidth=0, marker='o',
                markersize=4, label='Dati empirici', ax=ax2)

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
        plt.savefig(f'images\\series\\powerlaw_fit_{gamma_target}_min.png', dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    else:
        plt.savefig(f'images\\series\\powerlaw_fit_{gamma_target}.png', dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())

if __name__ == "__main__":
    gamma = 0.6
    print(f'Processing gamma = {gamma:.2f}')

    fit_plot(gamma)
    fit_plot(gamma, min = True)