import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.optimize import curve_fit, brentq
from scipy.stats import sem, kurtosis, norm, multivariate_normal
from scipy.fft import fft, ifft
from statsmodels.tsa.stattools import acf

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

    for k in range(1, len(acf_bin)):
        rb = acf_bin[k]
        try:
            acf_gauss[k] = brentq(
                lambda rg: _r_bin_from_r_gauss(rg, theta) - rb,
                -0.9999, 0.9999, xtol=1e-7, maxiter=60
            )
        except ValueError:
            # Fallback: closed-form arc-sine approximation (exact for p_plus=0.5)
            acf_gauss[k] = np.clip(np.sin(rb * np.pi / 2.0), -0.9999, 0.9999)

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


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
# POOLED ACF + ERROR BARS
# ══════════════════════════════════════════════════════════════════════════════

pooled      = pooled_acf(all_signs, nlags=max_lag)   # one-sided R[0..max_lag]
data_matrix = np.array(all_daily_corrs)
std_err     = sem(data_matrix, axis=0)

lags        = np.arange(1, max_lag + 1)
pooled_vals = pooled[1:]
err_vals    = std_err[1:]

# ══════════════════════════════════════════════════════════════════════════════
# POWER-LAW FITS
# ══════════════════════════════════════════════════════════════════════════════

fit_range_early = np.arange(1, 21)
popt_early, pcov_early = curve_fit(power_law, fit_range_early, pooled[1:21])
perr_early = np.sqrt(np.diag(pcov_early))
print(f"Fit lag  1-20:")
print(f"  A     = {popt_early[0]:.6f} ± {perr_early[0]:.6f}")
print(f"  delta = {popt_early[1]:.6f} ± {perr_early[1]:.6f}")

fit_range_late = np.arange(21, max_lag + 1)
popt_late, pcov_late = curve_fit(power_law, fit_range_late, pooled[21:])
perr_late = np.sqrt(np.diag(pcov_late))
print(f"Fit lag 21-{max_lag}:")
print(f"  A     = {popt_late[0]:.6f} ± {perr_late[0]:.6f}")
print(f"  delta = {popt_late[1]:.6f} ± {perr_late[1]:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCTION: van Vleck → Gaussian → binary
# ══════════════════════════════════════════════════════════════════════════════

all_signs_concat = np.concatenate(all_signs)
p_plus = float(np.mean(all_signs_concat > 0))
print(f"\nEmpirical p(+1) = {p_plus:.4f}")

# Step 1 — invert van Vleck: binary ACF → Gaussian ACF
print("Inverting van Vleck relation")
acf_gauss_target = van_vleck_invert(pooled, p_plus=p_plus)

# Step 2 — reconstruct Gaussian signal
# Use many long realizations and average their ACFs to suppress noise.
# For a binary process with long-range dependence, ACF variance at lag K
# scales as ~1/N, so N=200*max_lag with N_REAL realizations gives
# effective N_eff = N_REAL * N per lag.
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

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Daily ACF  +  Pooled ACF with fits
# ══════════════════════════════════════════════════════════════════════════════

fig1, axes1 = plt.subplots(1, 2, figsize=(18, 6))

# ── (0) Daily ACF ─────────────────────────────────────────────────────────────
ax = axes1[0]
for i, dc in enumerate(all_daily_corrs[:5]):
    ax.plot(lags, dc[1:], lw=1.0, label=f'Day {i+1}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
ax.set_title('Daily ACF'); ax.set_xlim([1, 50]); ax.legend(fontsize=9)

# ── (1) Pooled ACF + fits ─────────────────────────────────────────────────────
ax = axes1[1]
ax.fill_between(lags, pooled_vals - err_vals, pooled_vals + err_vals,
                color='steelblue', alpha=0.30, label='SEM')
ax.plot(lags, pooled_vals, color='steelblue', lw=1.0, label='Pooled ACF')
ax.plot(fit_range_early, power_law(fit_range_early, *popt_early),
        color='tomato', lw=2, linestyle='--',
        label=f'Fit lag 1-20  (δ={popt_early[1]:.3f})')
ax.plot(fit_range_late, power_law(fit_range_late, *popt_late),
        color='darkorange', lw=2, linestyle='--',
        label=f'Fit lag 21-{max_lag}  (δ={popt_late[1]:.3f})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
ax.set_title('Pooled ACF'); ax.legend(fontsize=9)

fig1.tight_layout()
fig1.savefig('images\\acf_original.png', dpi=300, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — ACF round-trip verification
# ══════════════════════════════════════════════════════════════════════════════

fig2, ax = plt.subplots(1, 2, figsize=(18, 6))

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
fig2.savefig('images\\acf_roundtrip.png', dpi=300, bbox_inches='tight')

plt.close()

def generate_and_save_synthetic_signs(data_dir, output_dir, acf_gauss_target, p_plus, seed=42):
    """
    Legge i file originali, genera segni sintetici con la stessa lunghezza e 
    struttura temporale, e li salva in una nuova cartella.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    paths = listdir(data_dir)
    rng = np.random.default_rng(seed)
    
    print(f"Inizio generazione segni sintetici in: {output_dir}")
    
    for path in paths:
        # 1. Caricamento dati originali
        file_path = os.path.join(data_dir, path)
        df_orig = pd.read_csv(file_path, header=None)
        N = len(df_orig)
        
        # 2. Generazione segnale Gaussiano con la memoria target (Wiener-Khinchin)
        # Usiamo reconstruct_gaussian definita precedentemente
        # Nota: N_recon deve essere almeno lungo quanto il segnale originale
        gauss_sig, _ = reconstruct_gaussian(acf_gauss_target, N=N, n_realizations=1, seed=seed)
        gauss_sig = gauss_sig[0] # Estraiamo la singola realizzazione
        
        # 3. Binarizzazione (Soglia Van Vleck)
        synth_signs = binarize(gauss_sig, p_plus=p_plus)
        
        # 4. Creazione del nuovo DataFrame
        # Manteniamo Tempo (col 0), Prezzo (col 1), Volume (col 2) originali
        # Sostituiamo il Segno (col 3) con quello sintetico
        df_synth = df_orig.copy()
        df_synth[3] = synth_signs
        
        # 5. Salvataggio
        output_path = os.path.join(output_dir, path)
        df_synth.to_csv(output_path, header=None, index=False)
        print(f"File salvato: {path} (N={N})")


# Definiamo le cartelle
output_folder = 'database\\data_signs'

# Chiamata alla funzione
generate_and_save_synthetic_signs(
    data_dir=data_dir, 
    output_dir=output_folder, 
    acf_gauss_target=acf_gauss_target, 
    p_plus=p_plus
)