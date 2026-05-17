import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import sem
from scipy.fft import fft
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
    Estimates ACF by aggregating autocovariances from all series
    before computing the ratio. More statistically correct than
    averaging individual ACFs.
    Returns one-sided R[0..nlags], normalized so R[0]=1.
    """
    sum_autocov = np.zeros(nlags + 1)
    for s in series_list:
        s = np.asarray(s, dtype=float)
        s = s - s.mean()
        n = len(s)
        for k in range(nlags + 1):
            if k < n:
                sum_autocov[k] += np.dot(s[k:], s[:n - k])
    return sum_autocov / sum_autocov[0]


# ══════════════════════════════════════════════════════════════════════════════
# DFA
# ══════════════════════════════════════════════════════════════════════════════

def pooled_dfa(series_list, n_vals=None):
    """
    Performs Pooled Detrended Fluctuation Analysis on a list of time series.
    Assumes all series share the same scaling exponent.
    """
    min_len = min(len(s) for s in series_list)
    if n_vals is None:
        n_vals = np.unique(np.logspace(np.log10(16), np.log10(min_len // 4), num=20, dtype=int))

    F_n = []

    for n in n_vals:
        total_squared_fluct = 0.0
        total_segments = 0

        for x in series_list:
            N = len(x)
            Y = np.cumsum(x - np.mean(x))

            num_segments = N // n
            if num_segments == 0:
                continue

            # Forward segments
            for m in range(num_segments):
                start = m * n
                end = start + n
                t = np.arange(n)
                poly = np.polyfit(t, Y[start:end], 1)
                trend = np.polyval(poly, t)
                total_squared_fluct += np.sum((Y[start:end] - trend) ** 2)
                total_segments += 1

            # Backward segments to use leftover data
            for m in range(num_segments):
                start = N - (m + 1) * n
                end = start + n
                t = np.arange(n)
                poly = np.polyfit(t, Y[start:end], 1)
                trend = np.polyval(poly, t)
                total_squared_fluct += np.sum((Y[start:end] - trend) ** 2)
                total_segments += 1

        if total_segments > 0:
            F_n.append(np.sqrt(total_squared_fluct / (total_segments * n)))
        else:
            F_n.append(np.nan)

    n_vals, F_n = np.array(n_vals), np.array(F_n)
    mask = (F_n > 0) & ~np.isnan(F_n)

    slope, intercept, r_value, p_value, std_err = linregress(np.log10(n_vals[mask]), np.log10(F_n[mask]))

    alpha = slope
    r_squared = r_value**2

    print(f"--- DFA Verification Report ---")
    print(f"R-squared (Linear Fit Quality): {r_squared:.4f}")
    print(f"Standard Error of Alpha: {std_err:.4f}")

    plt.figure()
    plt.loglog(n_vals, F_n, 'bo-', label='Pooled Fluctuation F(n)')
    plt.loglog(n_vals, 10**intercept * (n_vals**alpha), 'r--',
               label=f'Fit (alpha={alpha:.2f}, R²={r_squared:.3f})')
    plt.xlabel('Window size (n)')
    plt.ylabel('Fluctuation F(n)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    os.makedirs(os.path.join('images', 'acf'), exist_ok=True)
    plt.savefig(os.path.join('images', 'acf', 'dfa_check.png'))
    plt.close()

    return alpha, n_vals[mask], F_n[mask]


# ══════════════════════════════════════════════════════════════════════════════
# PLOT ACF
# ══════════════════════════════════════════════════════════════════════════════

def plot_acf(pooled, all_daily_corrs, gamma):
    data_matrix = np.array(all_daily_corrs)
    std_err     = sem(data_matrix, axis=0)

    lags        = np.arange(1, max_lag + 1)
    pooled_vals = pooled[1:]
    err_vals    = std_err[1:]

    tail_mask = lags >= 10

    log_lags = np.log(lags[tail_mask])
    log_y    = np.log(pooled_vals[tail_mask])

    log_A_fit = np.mean(log_y + gamma * log_lags)
    A_fit     = np.exp(log_A_fit)

    fig1, axes1 = plt.subplots(1, 2, figsize=(18, 6))

    ax = axes1[0]
    for i, dc in enumerate(all_daily_corrs[:5]):
        ax.plot(lags, dc[1:], lw=1.0, label=f'Day {i+1}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
    ax.set_title('Daily ACF'); ax.set_xlim([1, 50]); ax.legend(fontsize=9)

    ax = axes1[1]
    ax.fill_between(lags, pooled_vals - err_vals, pooled_vals + err_vals,
                    color='steelblue', alpha=0.30, label='SEM')
    ax.plot(lags, pooled_vals, color='steelblue', lw=1.0, label='Pooled ACF')
    ax.plot(lags, power_law(lags, A_fit, -gamma), color='tomato', lw=2,
            linestyle='--', label=f'Fit (δ={gamma:.3f})')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
    ax.set_title('Pooled ACF'); ax.legend(fontsize=9)

    fig1.tight_layout()
    os.makedirs(os.path.join('images', 'acf'), exist_ok=True)
    fig1.savefig(os.path.join('images', 'acf', 'acf_original.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    data_dir = 'database\\data'
    paths    = listdir(data_dir)
    max_lag  = 1_000

    all_signs       = []
    all_daily_corrs = []

    for path in paths:
        trades = pd.read_csv(f"{data_dir}\\{path}", header=None)
        signs  = trades[3].values.astype(float)
        all_signs.append(signs)
        all_daily_corrs.append(acf(signs, nlags=max_lag, fft=True))

    # Compute or load pooled binary ACF
    file_path = Path("database/acf_binary.npy")
    if file_path.is_file():
        pooled = np.load("database/acf_binary.npy")
    else:
        pooled = pooled_acf(all_signs, nlags=max_lag)
        np.save('database/acf_binary.npy', pooled)

    # Run DFA
    alpha, scales, fluctuations = pooled_dfa(all_signs)
    print(f"Pooled DFA Exponent (Alpha): {alpha:.4f}")
    print(f"Estimated ACF Tail Exponent (Gamma): {-2 + 2*alpha:.4f}")

    # Plot ACF
    plot_acf(pooled, all_daily_corrs, 2 - 2*alpha)

    # Save empirical p_plus for use in part 2
    all_signs_concat = np.concatenate(all_signs)
    p_plus = float(np.mean(all_signs_concat > 0))
    print(f"\nEmpirical p(+1) = {p_plus:.4f}")
    np.save('database/p_plus.npy', np.array(p_plus))

    # Save median series length for use in part 2
    median_len = int(np.median([len(s) for s in all_signs]))
    np.save('database/median_len.npy', np.array(median_len))

    print("\nSaved: database/acf_binary.npy, database/p_plus.npy, database/median_len.npy")