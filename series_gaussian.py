import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.optimize import brentq
from scipy.stats import norm, multivariate_normal
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
    """
    theta     = norm.ppf(1.0 - p_plus)
    acf_bin   = np.asarray(acf_bin, dtype=float)
    acf_gauss = np.empty_like(acf_bin)
    acf_gauss[0] = 1.0

    mu  = 2.0 * p_plus - 1.0
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
# FULL PIPELINE: BINARY ACF → BINARY SEQUENCE
# ══════════════════════════════════════════════════════════════════════════════

def generate_binary_sequence(acf_binary, p_plus, N, n_realizations=1, seed=42):
    """
    Generate binary ±1 sequence(s) of length N that reproduce the given
    binary ACF, using the van Vleck + Wiener-Khinchin pipeline.

    Parameters
    ----------
    acf_binary     : array R[0..K], one-sided binary ACF (R[0]=1)
    p_plus         : empirical fraction of +1s in the original series
    N              : length of each output sequence
                     (should be >> K for accurate ACF reproduction; 
                      a safe default is N = 1000 * (len(acf_binary) - 1))
    n_realizations : number of independent sequences to generate
    seed           : random seed for reproducibility

    Returns
    -------
    binary_signals : ndarray of shape (n_realizations, N), dtype float, values ±1
    acf_gauss      : the intermediate Gaussian ACF (van Vleck target), shape (K+1,)
    """
    # Step 1 — van Vleck inversion: binary ACF → Gaussian ACF
    acf_gauss = van_vleck_invert(acf_binary, p_plus=p_plus)

    # Step 2 — Wiener-Khinchin: Gaussian ACF → Gaussian signals
    gauss_signals, _ = reconstruct_gaussian(acf_gauss, N=N,
                                            n_realizations=n_realizations,
                                            seed=seed)

    # Step 3 — threshold → binary ±1
    binary_signals = np.array([binarize(g, p_plus=p_plus) for g in gauss_signals])

    return binary_signals, acf_gauss


# ══════════════════════════════════════════════════════════════════════════════
# PLOT ROUND-TRIP COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_round_trip(pooled, acf_recon_binary, acf_gauss_target, acf_recon_gauss, max_lag):
    fig2, ax = plt.subplots(1, 2, figsize=(18, 6))

    lags        = np.arange(1, max_lag + 1)
    pooled_vals = pooled[1:]

    ax[0].plot(lags, pooled_vals,          color='steelblue', lw=1.0,             label='Pooled ACF (original)')
    ax[0].plot(lags, acf_recon_binary[1:], color='red',       lw=1.0, linestyle=':', label='ACF of reconstructed signal')
    ax[0].set_title('Binary')
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel('Lag'); ax[0].set_ylabel('ACF')
    ax[0].legend()

    ax[1].plot(lags, acf_gauss_target[1:], color='blue',       lw=1.0, linestyle='-.', label='Van Vleck target')
    ax[1].plot(lags, acf_recon_gauss[1:],  color='darkorange', lw=1.0, linestyle=':', label='ACF of reconstructed signal')
    ax[1].set_title('Gaussian')
    ax[1].set_xscale('log'); ax[1].set_yscale('log')
    ax[1].set_xlabel('Lag'); ax[1].set_ylabel('ACF')
    ax[1].legend()

    fig2.tight_layout()
    os.makedirs(os.path.join('images', 'acf'), exist_ok=True)
    fig2.savefig(os.path.join('images', 'acf', 'acf_roundtrip.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Load outputs from part 1
    pooled     = np.load('database/acf_binary.npy')
    p_plus     = float(np.load('database/p_plus.npy'))
    median_len = int(np.load('database/median_len.npy'))

    max_lag = len(pooled) - 1

    print(f"Loaded binary ACF (max_lag={max_lag}), p(+1)={p_plus:.4f}, median series length={median_len}")

    # Generate binary sequences matching the original ACF
    N_REAL  = 250
    N_recon = max(median_len, 1000 * max_lag)
    print(f"Generating {N_REAL} binary sequences (N={N_recon} each)...")
    binary_signals, acf_gauss_target = generate_binary_sequence(
        pooled, p_plus=p_plus, N=N_recon, n_realizations=N_REAL, seed=42
    )
    np.save('database/acf_gaussian.npy', acf_gauss_target)

    # Reconstruct Gaussian signals for the round-trip plot (reuse same seed)
    gauss_signals, _ = reconstruct_gaussian(acf_gauss_target, N=N_recon,
                                            n_realizations=N_REAL, seed=42)

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

    plot_round_trip(pooled, acf_recon_binary, acf_gauss_target, acf_recon_gauss, max_lag)
    print("\nSaved: database/acf_gaussian.npy, images/acf/acf_roundtrip.png")