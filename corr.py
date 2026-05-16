import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.optimize import curve_fit, brentq
from scipy.stats import sem, kurtosis, norm, multivariate_normal
from scipy.fft import fft, ifft
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
    Stima l'ACF aggregando le autocovarianze di tutte le serie
    prima di calcolare il rapporto. Più corretta statisticamente
    rispetto alla media delle singole ACF.
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
# VAN VLECK INVERSION
# ══════════════════════════════════════════════════════════════════════════════

def _r_bin_from_r_gauss(r_gauss, theta):
    """
    Exact forward map: R_binary given R_gaussian and threshold theta.
    Uses E[sign(X-θ)·sign(Y-θ)] for bivariate normal (X,Y) with corr r_gauss.
    """
    cov  = [[1.0, r_gauss], [r_gauss, 1.0]]
    mv   = multivariate_normal(mean=[0.0, 0.0], cov=cov)
    p_pp = mv.cdf([-theta, -theta])   # P(X > θ, Y > θ)
    p_nn = mv.cdf([ theta,  theta])   # P(X < θ, Y < θ)
    return 2.0 * (p_pp + p_nn) - 1.0


def van_vleck_invert(acf_bin, p_plus):
    """
    Invert the van Vleck relation: given the one-sided binary ACF R_bin[0..K]
    and the fraction of +1s p_plus, return the Gaussian ACF R_gauss[0..K]
    such that threshold(Gaussian with R_gauss) reproduces R_bin.

    Forward relation (exact):
        R_bin(k) = E[sign(X-θ)·sign(Y-θ)],  (X,Y) ~ N(0, [[1,r],[r,1]])
        θ = Φ^{-1}(1 - p_plus)

    We invert numerically with brentq; arcsin approximation used as fallback.
    """
    theta     = norm.ppf(1.0 - p_plus)
    acf_bin   = np.asarray(acf_bin, dtype=float)
    acf_gauss = np.empty_like(acf_bin)
    acf_gauss[0] = 1.0

    # Inside van_vleck_invert
    mu = 2.0 * p_plus - 1.0
    var = 1.0 - mu**2

    for k in range(1, len(acf_bin)):
        target_raw = acf_bin[k] * var + mu**2

        acf_gauss[k] = brentq(
            lambda rg: _r_bin_from_r_gauss(rg, theta) - target_raw,
            -0.9999, 0.9999, xtol=1e-7
        )

    return acf_gauss


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL RECONSTRUCTION FROM ONE-SIDED ACF  (Wiener-Khinchin)
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_gaussian(acf_onesided, N, n_realizations=1, seed=42):
    """
    Generate Gaussian signal(s) of length N whose ACF matches acf_onesided.

    acf_onesided : R[0..K], R[0]=1  (one-sided, as from pooled_acf)
    N            : output length — should be >> K for reliable round-trip
    """
    rng = np.random.default_rng(seed)
    r   = np.asarray(acf_onesided, dtype=float)
    K   = len(r) - 1

    # Build two-sided periodic extension of length N:
    # [R0, R1, ..., RK, 0, ..., 0, RK, ..., R1]
    acf_full          = np.zeros(N)
    acf_full[0]       = r[0]
    acf_full[1:K+1]   = r[1:K+1]
    acf_full[N-K:N]   = r[K:0:-1]

    psd = np.real(fft(acf_full))
    psd = np.clip(psd, 0, None)
    amp = np.sqrt(psd)

    signals = []
    for _ in range(n_realizations):
        ph = rng.uniform(0, 2*np.pi, N)
        sp = amp * np.exp(1j * ph)

        # Enforce conjugate symmetry → purely real IFFT
        sp[0] = amp[0]
        if N % 2 == 0:
            sp[N // 2]    = amp[N // 2]
            sp[N//2 + 1:] = np.conj(sp[1:N//2][::-1])
        else:
            sp[(N+1)//2:] = np.conj(sp[1:(N+1)//2][::-1])

        sig = np.real(ifft(sp))
        sig = (sig - sig.mean()) / sig.std()
        signals.append(sig)

    return np.array(signals), psd


# ══════════════════════════════════════════════════════════════════════════════
# BINARIZATION
# ══════════════════════════════════════════════════════════════════════════════

def binarize(gaussian_signal, p_plus):
    """
    Threshold a N(0,1) signal to ±1.
    θ = Φ^{-1}(1 - p_plus) ensures P(x > θ) = p_plus.
    """
    theta = norm.ppf(1.0 - p_plus)
    return np.where(gaussian_signal > theta, 1.0, -1.0)


# ══════════════════════════════════════════════════════════════════════════════
# SURROGATE DATA  (Theiler et al. 1992)
# ══════════════════════════════════════════════════════════════════════════════

def make_surrogates(signal, n_surrogates=200, seed=0):
    """
    Phase-randomized surrogates: preserve |FFT| amplitude, randomize phases.
    Each surrogate is a realization of a linear Gaussian process with the
    same ACF as `signal` (the null hypothesis H0).
    """
    rng = np.random.default_rng(seed)
    n   = len(signal)
    amp = np.abs(fft(signal))

    surrogates = np.zeros((n_surrogates, n))
    for k in range(n_surrogates):
        ph = rng.uniform(0, 2*np.pi, n)
        sp = amp * np.exp(1j * ph)

        sp[0] = amp[0]
        if n % 2 == 0:
            sp[n // 2]    = amp[n // 2]
            sp[n//2 + 1:] = np.conj(sp[1:n//2][::-1])
        else:
            sp[(n+1)//2:] = np.conj(sp[1:(n+1)//2][::-1])

        surrogates[k] = np.real(ifft(sp))

    return surrogates


def surrogate_test(all_signs, n_surrogates=200, seed=0):
    """
    Test H0 (linear Gaussian process) for each daily series using
    excess kurtosis as the test statistic.

    Returns
    -------
    obs_kurt  : observed excess kurtosis per day  (n_days,)
    surr_kurt : surrogate kurtosis matrix          (n_days, n_surrogates)
    p_values  : two-sided p-value per day          (n_days,)
    """
    obs_kurt  = np.array([kurtosis(s, fisher=True) for s in all_signs])
    surr_kurt = np.zeros((len(all_signs), n_surrogates))

    for i, s in enumerate(all_signs):
        surr = make_surrogates(np.asarray(s, dtype=float),
                               n_surrogates=n_surrogates,
                               seed=seed + i)
        surr_kurt[i] = [kurtosis(r, fisher=True) for r in surr]

    p_values = np.mean(np.abs(surr_kurt) >= np.abs(obs_kurt)[:, None], axis=1)
    return obs_kurt, surr_kurt, p_values

def generate_gaussian_signs(length, acf_path = 'database/acf.npy', p_plus = 0.5, seed=42):
    """
    Legge i file originali, genera segni sintetici con la stessa lunghezza e 
    struttura temporale, e li salva in una nuova cartella.
    """
    acf = np.load(acf_path)
    
    # 2. Generazione segnale Gaussiano con la memoria target (Wiener-Khinchin)
    # Usiamo reconstruct_gaussian definita precedentemente
    # Nota: N_recon deve essere almeno lungo quanto il segnale originale
    gauss_sig, _ = reconstruct_gaussian(acf, N=length, n_realizations=1, seed=seed)
    gauss_sig = gauss_sig[0] # Estraiamo la singola realizzazione
    
    # 3. Binarizzazione (Soglia Van Vleck)
    synth_signs = binarize(gauss_sig, p_plus=p_plus)
    
    return synth_signs

def plot_acf(pooled, all_daily_corrs, gamma):
    data_matrix = np.array(all_daily_corrs)
    std_err     = sem(data_matrix, axis=0)

    lags        = np.arange(1, max_lag + 1)
    pooled_vals = pooled[1:]
    err_vals    = std_err[1:]

    tail_mask = lags >= 10
    
    # Calculate A in log-log space so all lags are weighted equally
    # $log(y) = log(A) - gamma * log(x)$  -->  $log(A) = log(y) + gamma * log(x)$
    
    log_lags = np.log(lags[tail_mask])
    log_y = np.log(pooled_vals[tail_mask])
    
    log_A_fit = np.mean(log_y + gamma * log_lags)
    A_fit = np.exp(log_A_fit)

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
    
    # Plotting using your exact gamma definition
    ax.plot(lags, power_law(lags, A_fit, -gamma), color='tomato', lw=2,
            linestyle='--', label=f'Fit (δ={gamma:.3f})')
            
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
    ax.set_title('Pooled ACF'); ax.legend(fontsize=9)

    fig1.tight_layout()
    
    # Ensure directory exists before saving
    os.makedirs(os.path.join('images', 'acf'), exist_ok=True)
    fig1.savefig(os.path.join('images', 'acf', 'acf_original.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_round_trip(pooled, acf_recon_binary, acf_gauss_target, acf_recon_gauss):
    fig2, ax = plt.subplots(1, 2, figsize=(18, 6))

    pooled_vals = pooled[1:]

    lags = np.arange(1, max_lag + 1)

    ax[0].plot(lags, pooled_vals, color='steelblue',  lw=1.0, label='Pooled ACF (original)')
    ax[0].plot(lags, acf_recon_binary[1:], color='red', lw=1.0, linestyle=':', label=f'ACF of reconstructed signal')

    ax[0].set_title('Binary')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Lag')
    ax[0].set_ylabel('ACF')
    ax[0].legend()

    ax[1].plot(lags, acf_gauss_target[1:], color='blue', lw=1.0, linestyle='-.', label='Van Vleck target')
    ax[1].plot(lags, acf_recon_gauss[1:], color='darkorange', lw=1.0, linestyle=':', label=f'ACF of reconstructed signal')

    ax[1].set_title('Gaussian')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('ACF')
    ax[1].legend()

    fig2.tight_layout()
    fig2.savefig('images\\acf\\acf_roundtrip.png', dpi=300, bbox_inches='tight')
    plt.close()

def pooled_dfa(series_list, n_vals=None):
    """
    Performs Pooled Detrended Fluctuation Analysis on a list of time series.
    Assumes all series share the same scaling exponent.
    """
    # 1. Set default window sizes (n) based on the length of the shortest series
    min_len = min(len(s) for s in series_list)
    if n_vals is None:
        n_vals = np.unique(np.logspace(np.log10(16), np.log10(min_len // 4), num=20, dtype=int))
    
    F_n = []
    
    # 2. Loop through each window size n
    for n in n_vals:
        total_squared_fluct = 0.0
        total_segments = 0
        
        # 3. Process each series independently for the current window size
        for x in series_list:
            N = len(x)
            # Integrate the series
            Y = np.cumsum(x - np.mean(x))
            
            # Determine number of segments (forward and backward)
            num_segments = N // n
            if num_segments == 0:
                continue
                
            # Forward segments
            for m in range(num_segments):
                start = m * n
                end = start + n
                t = np.arange(n)
                # Linear fit (DFA1)
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
        
        # 4. Calculate the pooled RMS fluctuation for this specific 'n'
        if total_segments > 0:
            F_n.append(np.sqrt(total_squared_fluct / (total_segments * n)))
        else:
            F_n.append(np.nan)
            
    # 5. Linear regression on log-log scale to find alpha
    # Filter out any NaNs or zeros just in case
    n_vals, F_n = np.array(n_vals), np.array(F_n)
    mask = (F_n > 0) & ~np.isnan(F_n)
    
    alpha, intercept = np.polyfit(np.log10(n_vals[mask]), np.log10(F_n[mask]), 1)

    # Instead of np.polyfit, use scipy's linregress to get error metrics
    slope, intercept, r_value, p_value, std_err = linregress(np.log10(n_vals), np.log10(F_n))

    alpha = slope
    r_squared = r_value**2

    print(f"--- Verification Report ---")
    print(f"R-squared (Linear Fit Quality): {r_squared:.4f}")
    print(f"Standard Error of Alpha: {std_err:.4f}")

    # Optional: Diagnostic Plotting built into the function
    plt.figure()
    plt.loglog(n_vals, F_n, 'bo-', label='Pooled Fluctuation F(n)')
    plt.loglog(n_vals, 10**intercept * (n_vals**alpha), 'r--', label=f'Fit (alpha={alpha:.2f}, R²={r_squared:.3f})')
    plt.xlabel('Window size (n)')
    plt.ylabel('Fluctuation F(n)')
    #plt.title('DFA Verification Plot')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('images\\acf\\dfa_check.png')

    return alpha, n_vals[mask], F_n[mask]

if __name__ == '__main__':
    # DATA LOADING
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

    file_path = Path("database/acf_binary.npy")
    if file_path.is_file():
        pooled = np.load("database/acf_binary.npy")
    else:
        # POOLED ACF + ERROR BARS
        pooled = pooled_acf(all_signs, nlags=max_lag)   # one-sided R[0..max_lag]

        np.save('database/acf_binary.npy', pooled)

    # Run DFA
    # 'order=1' corresponds to DFA1 (linear detrending)
    alpha, scales, fluctuations = pooled_dfa(all_signs)
    print(f"Pooled DFA Exponent (Alpha): {alpha:.4f}")
    print(f"Estimated ACF Tail Exponent (Gamma): {-2 + 2*alpha:.4f}")

    print(f"DFA Exponent (Alpha): {alpha}")

    plot_acf(pooled, all_daily_corrs, 2 - 2*alpha)

    # RECONSTRUCTION: van Vleck → Gaussian → binary
    all_signs_concat = np.concatenate(all_signs)
    p_plus = float(np.mean(all_signs_concat > 0))
    print(f"\nEmpirical p(+1) = {p_plus:.4f}")

    # Step 1 — invert van Vleck: binary ACF → Gaussian ACF
    print("Inverting van Vleck relation")
    acf_gauss_target = van_vleck_invert(pooled, p_plus=p_plus)

    np.save('database/acf_gaussian.npy', pooled)

    # Step 2 — reconstruct Gaussian signal
    N_REAL  = 250
    N_recon = max(int(np.median([len(s) for s in all_signs])), 1000 * max_lag)
    print(f"Reconstructing {N_REAL} Gaussian realizations (N={N_recon} each)")
    gauss_signals, psd_used = reconstruct_gaussian(
        acf_gauss_target, N=N_recon, n_realizations=N_REAL, seed=42
    )

    # Step 3 — threshold → binary ±1
    binary_signals = np.array([binarize(g, p_plus=p_plus) for g in gauss_signals])

    # Average ACF over all realizations to suppress estimator noise
    acf_recon_gauss  = np.mean([acf(g, nlags=max_lag, fft=True) for g in gauss_signals],  axis=0)
    acf_recon_binary = np.mean([acf(b, nlags=max_lag, fft=True) for b in binary_signals], axis=0)

    print(f"Reconstructed p(+1) = {np.mean([np.mean(b > 0) for b in binary_signals]):.4f}  "
        f"(target {p_plus:.4f})")
    print(f"\nRound-trip check (lags 1-5):")
    print(f"  {'lag':>4}  {'R_bin_orig':>12}  {'R_gauss_target':>15}  {'R_bin_recon':>12}")
    for k in range(1, 6):
        print(f"  {k:>4}  {pooled[k]:>12.5f}  {acf_gauss_target[k]:>15.5f}  "
            f"{acf_recon_binary[k]:>12.5f}")
        
    plot_round_trip(pooled, acf_recon_binary, acf_gauss_target, acf_recon_gauss)