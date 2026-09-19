import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import sem
from statsmodels.tsa.stattools import acf
from scipy.stats import linregress
from pathlib import Path
import os
from scipy.optimize import curve_fit

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

def log_power_law(log_x, log_A, delta):
    return log_A + delta * log_x


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

def plot_acf(pooled, max_lag):
    """
    Single-panel figure: pooled ACF with:
      * raw unbinned ACF shown as cyan points
      * log-binned ACF
      * direct power-law fit using scipy.optimize.curve_fit
    """

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

    # ── 2. FIT CON CURVE_FIT SULLA CODA DEI DATI BINNATI ───────────────────
    tail_mask_binned = (bin_centers >= 10) & (bin_centers < 1000)

    # Prendiamo solo valori strettamente positivi per la coda
    valid_binned = tail_mask_binned & (binned_acf > 0)
    
    x_fit = bin_centers[valid_binned]
    y_fit = binned_acf[valid_binned]

    # Stima iniziale per i parametri [A, delta]
    p0 = [1.0, -0.5]

    # Fit non lineare con curve_fit
    popt, pcov = curve_fit(log_power_law, np.log10(x_fit), np.log10(y_fit))
    log_A_fit, delta_fit = popt
    A_fit = 10**log_A_fit
    
    # Calcolo incertezza sugli errori dai parametri (deviazione standard)
    perr = np.sqrt(np.diag(pcov))
    delta_err = perr[1]

    gamma_empirico = -delta_fit          # Esponente della coda ACF

    print("\n--- ACF Tail Fitting Report (curve_fit) ---")
    print(f"  A fit                     : {A_fit:.4f} ± {perr[0]:.4f}")
    print(f"  Delta (slope)             : {delta_fit:.4f} ± {delta_err:.4f}")
    print(f"  Gamma ACF                 : {gamma_empirico:.4f}")

    # ── 3. PLOTTING ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # 1. Dati NON binnati (Raw ACF) con pallini celesti
    valid_raw = pooled_vals > 0
    ax.plot(
        lags[valid_raw], 
        pooled_vals[valid_raw], 
        color='tab:blue', 
        marker='o', 
        linestyle='', 
        markersize=5, 
        markeredgecolor='white',    # Bordo bianco
        markeredgewidth=0.5,
        alpha=0.75, 
        label='Raw ACF'
    )

    # 2. Dati binnati
    ax.plot(
        bin_centers, 
        binned_acf, 
        color='black', 
        alpha=1.0, 
        linestyle='-', 
        marker='o', 
        label='Binned ACF'
    )

    # 3. Curva di fit calcolata con curve_fit
    ax.plot(
        x_fit,
        power_law(x_fit, A_fit, delta_fit),
        color='red', lw=2.0, ls='--',
        label=rf'ACF fit $\gamma_{{ACF}}={gamma_empirico:.3f} \pm {delta_err:.3f}$'
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
    plt.show()

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

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_acf(pooled, max_lag)

    # ── Save auxiliary outputs for downstream scripts ─────────────────────────
    all_signs_concat = np.concatenate(all_signs)

    p_plus = float(np.mean(all_signs_concat > 0))
    np.save('database/p_plus.npy', np.array(p_plus))
    print(f"\nEmpirical p(+1) = {p_plus:.4f}")

    median_len = int(np.median([len(s) for s in all_signs]))
    np.save('database/median_len.npy', np.array(median_len))

    print("\nSaved: database/acf_binary.npy, database/p_plus.npy, database/median_len.npy")