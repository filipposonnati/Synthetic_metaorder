import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import sem
from statsmodels.tsa.stattools import acf
from scipy.stats import linregress
from pathlib import Path
import os

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14
})

# ══════════════════════════════════════════════════════════════════════════════
# FIT MODELS
# ══════════════════════════════════════════════════════════════════════════════

def power_law(x, A, delta):
    return A * x**delta


# ══════════════════════════════════════════════════════════════════════════════
# POOLED ACF
# ══════════════════════════════════════════════════════════════════════════════

def pooled_acf(series_list, nlags):
    """
    Estimates the pooled ACF by aggregating per-series autocovariances
    (each normalized by its own N) before taking the global ratio.

    This is statistically correct: it is equivalent to weighting each
    series by its length, and avoids the downward bias in the tail that
    arises when raw (un-normalized) dot products are summed across series
    of different lengths.

    Uses FFT-based autocovariance for speed (O(N log N) per series
    instead of O(N * nlags)).

    Returns R[0..nlags] normalized so that R[0] = 1.
    """
    sum_autocov = np.zeros(nlags + 1)

    for s in series_list:
        s = np.asarray(s, dtype=float)
        s = s - s.mean()          # per-series demeaning (correct for daily sign series)
        n = len(s)

        # FFT-based circular autocovariance, normalized by N (biased estimator).
        # Padding to 2*n avoids circular wrap-around artifacts.
        f    = np.fft.rfft(s, n=2 * n)
        acov = np.fft.irfft(f * np.conj(f))[:nlags + 1].real / n

        sum_autocov += acov

    # Normalize so R[0] = 1
    return sum_autocov / sum_autocov[0]

# ══════════════════════════════════════════════════════════════════════════════
# PLOT ACF
# ══════════════════════════════════════════════════════════════════════════════

def plot_acf(pooled, all_daily_corrs, gamma_dfa, max_lag):
    """
    Single-panel figure: pooled ACF with:
      * direct power-law fit to the tail (lag >= 20)
      * theoretical prediction from DFA exponent

    Parameters
    ----------
    pooled          : ndarray, shape (max_lag+1,)  R[0..max_lag]
    all_daily_corrs : list of ndarrays, each shape (max_lag+1,)
    gamma_dfa       : float   ACF tail exponent predicted by DFA  (2 - 2*alpha)
    max_lag         : int
    """
    data_matrix = np.array(all_daily_corrs)          # (n_days, max_lag+1)
    err_vals    = sem(data_matrix, axis=0)[1:]        # SEM over days, lags 1..max_lag

    lags        = np.arange(1, max_lag + 1)
    pooled_vals = pooled[1:]                          # drop lag-0

    # ── tail fit (log-log linear regression for lags >= 20) ─────────────────
    tail_mask = lags >= 20

    # Guard: drop non-positive pooled values before log transform
    valid = tail_mask & (pooled_vals > 0)
    log_lags = np.log10(lags[valid])
    log_y    = np.log10(pooled_vals[valid])

    slope, intercept, r_value, _, std_err_fit = linregress(log_lags, log_y)

    gamma_empirico = -slope          # ACF tail exponent from direct fit
    A_fit          = 10 ** intercept

    # Anchor the DFA prediction at the first valid tail point (avoids free offset)
    A_dfa = 10 ** (log_y[0] + gamma_dfa * log_lags[0])

    print("\n--- ACF Tail Fitting Report ---")
    print(f"  Delta from ACF direct fit : {gamma_empirico:.4f}")
    print(f"  Delta predicted by DFA    : {gamma_dfa:.4f}")
    print(f"  R² of tail fit            : {r_value**2:.4f}")
    print(f"  Std error of slope        : {std_err_fit:.4f}")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # Pooled ACF + fits
    ax.fill_between(
        lags,
        pooled_vals - err_vals,
        pooled_vals + err_vals,
        color='steelblue', alpha=0.30, label='±SEM'
    )
    ax.plot(lags, pooled_vals, color='steelblue', lw=1.5, label='Pooled ACF')

    # Direct ACF tail fit
    ax.plot(
        lags[valid],
        power_law(lags[valid], A_fit, slope),
        color='tomato', lw=2.5, ls='--',
        label=rf'ACF tail fit  $\delta_{{ACF}}$={gamma_empirico:.3f}'
    )

    # DFA-predicted power law
    ax.plot(
        lags[valid],
        power_law(lags[valid], A_dfa, -gamma_dfa),
        color='purple', lw=2, ls=':',
        label=rf'DFA prediction  $\delta_{{DFA}}$={gamma_dfa:.3f}'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Lag $\tau$')
    ax.set_ylabel('ACF')
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.join('images', 'acf'), exist_ok=True)
    fig.savefig(os.path.join('images', 'acf', 'acf.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    data_dir = os.path.join('database', 'data')
    paths    = listdir(data_dir)
    max_lag  = 1_000

    all_signs       = []
    all_daily_corrs = []

    for path in paths:
        trades = pd.read_csv(os.path.join(data_dir, path), header=None)
        signs  = trades[3].values.astype(float)
        all_signs.append(signs)
        all_daily_corrs.append(acf(signs, nlags=max_lag, fft=True))

    # ── Pooled ACF (compute once, then cache) ────────────────────────────────
    cache_path = Path("database/acf_binary.npy")
    if cache_path.is_file():
        pooled = np.load(cache_path)
        print("Loaded cached pooled ACF.")
    else:
        print("Computing pooled ACF …")
        pooled = pooled_acf(all_signs, nlags=max_lag)
        np.save(cache_path, pooled)
        print("Saved pooled ACF to cache.")

    gamma_dfa = 0.752

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_acf(pooled, all_daily_corrs, gamma_dfa, max_lag)

    # ── Save auxiliary outputs for downstream scripts ─────────────────────────
    all_signs_concat = np.concatenate(all_signs)

    p_plus = float(np.mean(all_signs_concat > 0))
    np.save('database/p_plus.npy', np.array(p_plus))
    print(f"\nEmpirical p(+1) = {p_plus:.4f}")

    median_len = int(np.median([len(s) for s in all_signs]))
    np.save('database/median_len.npy', np.array(median_len))

    print("\nSaved: database/acf_binary.npy, database/p_plus.npy, database/median_len.npy")