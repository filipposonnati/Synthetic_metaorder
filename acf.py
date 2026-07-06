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
      * direct power-law fit to the log-binned tail (lag >= 20)
      * theoretical prediction from DFA exponent
    """
    data_matrix = np.array(all_daily_corrs)          # (n_days, max_lag+1)
    err_vals    = sem(data_matrix, axis=0)[1:]        # SEM over days, lags 1..max_lag

    lags        = np.arange(1, max_lag + 1)
    pooled_vals = pooled[1:]                          # drop lag-0

    # ── 1. LOG-BINNING DEI DATI ORIGINARI ───────────────────────────────────
    num_bins = 50
    bin_edges = np.unique(np.logspace(0, np.log10(max_lag), num_bins).astype(int))
    
    bin_centers = []
    binned_acf = []
    
    for i in range(len(bin_edges) - 1):
        start, end = bin_edges[i], bin_edges[i+1]
        # Seleziona i lag inclusi in questo specifico bin
        mask = (lags >= start) & (lags < end)
        if np.any(mask) and np.any(pooled_vals[mask] > 0):
            # Media aritmetica dei lag nel bin (centro del bin)
            bin_centers.append(np.mean(lags[mask]))
            # Media dell'ACF nel bin 
            binned_acf.append(np.mean(pooled_vals[mask]))

    bin_centers = np.array(bin_centers)
    binned_acf = np.array(binned_acf)

    # ── 2. FIT SULLA CODA DEI DATI BANNATI (lag >= 20) ──────────────────────
    # Applichiamo la maschera di taglio direttamente sui centri dei bin calcolati
    tail_mask_binned = bin_centers >= 20

    # Guard: assicuriamoci di prendere solo valori strettamente positivi per il log
    valid_binned = tail_mask_binned & (binned_acf > 0)
    
    log_lags_binned = np.log10(bin_centers[valid_binned])
    log_y_binned    = np.log10(binned_acf[valid_binned])

    # Regressione lineare sui dati binnati
    slope, intercept, r_value, _, std_err_fit = linregress(log_lags_binned, log_y_binned)

    gamma_empirico = -slope          # Esponente della coda ACF dal fit binnato
    A_fit          = 10 ** intercept

    # Ancoriamo la predizione DFA al primo punto valido della coda binnata
    A_dfa = 10 ** (log_y_binned[0] + gamma_dfa * log_lags_binned[0])

    print("\n--- ACF Tail Fitting Report (On Log-Binned Data) ---")
    print(f"  Delta from ACF direct fit : {gamma_empirico:.4f}")
    print(f"  Delta predicted by DFA    : {gamma_dfa:.4f}")
    print(f"  R² of tail fit            : {r_value**2:.4f}")
    print(f"  Std error of slope        : {std_err_fit:.4f}")

    # ── 3. PLOTTING ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # Grafico a dispersione dei punti binnati e ripuliti dal rumore
    ax.plot(bin_centers, binned_acf, color='black', alpha=1.0, linestyle='-', marker = 'o', label='Pooled ACF')

    # Curva di fit calcolata sui dati binnati (mostrata sull'intervallo di fit)
    ax.plot(
        bin_centers[valid_binned],
        power_law(bin_centers[valid_binned], A_fit, slope),
        color='tomato', lw=2.5, ls='--',
        label=rf'ACF tail fit (binned) $\gamma_{{ACF}}$={gamma_empirico:.3f}'
    )

    # Predizione DFA ancorata ai dati binnati
    ax.plot(
        bin_centers[valid_binned],
        power_law(bin_centers[valid_binned], A_dfa, -gamma_dfa),
        color='purple', lw=2, ls=':',
        label=rf'DFA prediction  $\gamma_{{DFA}}$={gamma_dfa:.3f}'
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
    max_lag  = 5_000

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