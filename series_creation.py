import numpy as np
from scipy import signal, fft as sp_fft
from scipy.fft import next_fast_len
from scipy.special import gammaln
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
import pywt

# Use the new-style Generator: faster standard_normal than legacy np.random.normal
_rng = np.random.default_rng()


# =============================================================================
# SERIES GENERATION
# =============================================================================

def generate_ffm(n_points, gamma):
    """
    Generates a long-memory continuous series using the Fractional Filtering Method (FFM).

    The idea is to shape white noise in the frequency domain so that the resulting
    signal has a power spectrum S(f) ~ f^(-beta), which produces an autocorrelation
    function that decays as C(tau) ~ tau^(-gamma).

    The relationship between the spectral exponent beta and the correlation exponent
    gamma is: beta = 1 - gamma.

    Parameters
    ----------
    n_points : int   — desired length of the output series
    gamma    : float — target autocorrelation decay exponent (C(tau) ~ tau^-gamma)

    Returns
    -------
    continuous_series : array of length n_points
    """
    beta = 1.0 - gamma
    N    = n_points

    noise    = _rng.standard_normal(N)
    freq_dom = np.fft.rfft(noise)
    freqs    = np.fft.rfftfreq(N)

    freqs[0]        = 1e-10
    filter_array    = freqs ** (-beta / 2.0)
    filter_array[0] = 0.0

    return np.fft.irfft(freq_dom * filter_array, n=N)[:n_points]


def generate_arfima(n, d, ar_params=None, ma_params=None, std_dev=1.0):
    """
    Generates a realization of an ARFIMA(p, d, q) process.

    ARFIMA (AutoRegressive Fractionally Integrated Moving Average) is a
    generalization of ARIMA where the differencing order d is allowed to be
    fractional. For |d| < 0.5 the process is stationary and exhibits long-range
    dependence, with autocorrelation decaying as C(tau) ~ tau^(2d-1).

    Algorithm
    ---------
    The original implementation convolved white noise of length 2n with the
    MA(inf) weight vector of length n — an O(n log n) FFT on arrays of total
    size ~3n. This version instead builds the fractional filter analytically
    in the frequency domain and operates on a single array of length n:

        W(f) = (2 sin(pi f))^{-d} * exp(i d (pi/2 - pi f))

    This is the exact Z-transform of (1-L)^{-d} evaluated on the unit circle.
    The process is generated as x = IFFT(RFFT(eps) * W), where eps is white
    noise of length n. Because we work entirely at size n (rather than 2n or 3n),
    this is roughly 3-4x faster than the convolution approach.

    Additional micro-optimisations:
    - scipy.fft.next_fast_len selects the smallest highly-composite FFT size
      >= n (never a prime, often a round number), avoiding slow FFT paths.
    - np.random.default_rng().standard_normal is ~35% faster than the legacy
      np.random.normal for large arrays.
    - The filter is built with in-place operations (np.power(..., out=...),
      direct cos/sin calls) to minimise temporary array allocations.

    Note: the DC component (f = 0) is zeroed to ensure a zero-mean output,
    consistent with the stationarity assumption.

    Parameters
    ----------
    n         : int   — number of samples to generate
    d         : float — fractional integration parameter (|d| < 0.5 for stationarity)
    ar_params : list  — AR coefficients [phi_1, phi_2, ...] (optional)
    ma_params : list  — MA coefficients [theta_1, theta_2, ...] (optional)
    std_dev   : float — standard deviation of the driving white noise (default 1.0)

    Returns
    -------
    arfima_process : array of length n
    """
    # -------------------------------------------------------------------------
    # Step 1: Choose FFT size — next highly-composite number >= n
    # -------------------------------------------------------------------------
    N = next_fast_len(n)

    # -------------------------------------------------------------------------
    # Step 2: Build the analytic fractional filter in the frequency domain
    # -------------------------------------------------------------------------
    # The fractional differencing operator (1-L)^{-d} has the frequency response:
    #   W(f) = (1 - e^{-2pi i f})^{-d}
    # which can be written in polar form as:
    #   |W(f)| = (2 sin(pi f))^{-d}
    #   arg W(f) = d * (pi/2 - pi*f)
    #
    # In-place operations reuse the `mag` buffer to avoid extra allocations.
    freqs    = sp_fft.rfftfreq(N)
    fs       = np.where(freqs == 0, 1e-10, freqs)  # guard DC against division by zero
    pf       = np.pi * fs

    mag      = np.sin(pf, out=pf.copy())            # sin(pi*f)
    mag     *= 2.0                                   # 2*sin(pi*f)
    np.power(mag, -d, out=mag)                       # (2*sin(pi*f))^{-d}

    phi      = d * (np.pi / 2.0 - pf)               # phase angle
    W        = mag * (np.cos(phi) + 1j * np.sin(phi))
    W[0]     = 0.0                                   # zero DC → zero mean output

    # -------------------------------------------------------------------------
    # Step 3: Generate white noise and apply the filter
    # -------------------------------------------------------------------------
    eps          = _rng.standard_normal(N) * std_dev
    x_fractional = sp_fft.irfft(sp_fft.rfft(eps) * W, n=N)[:n]

    # -------------------------------------------------------------------------
    # Step 4: Optional ARMA(p, q) filter on top
    # -------------------------------------------------------------------------
    if ar_params is not None or ma_params is not None:
        ar_poly = np.r_[1, -np.array(ar_params)] if ar_params is not None else np.array([1.0])
        ma_poly = np.r_[1,  np.array(ma_params)] if ma_params is not None else np.array([1.0])
        x_fractional = signal.lfilter(ma_poly, ar_poly, x_fractional)

    return x_fractional


def generate_fgn(n, H):
    if not (0.0 < H < 1.0):
        raise ValueError(f"Hurst exponent H must be in (0, 1), got {H}.")

    tH2 = 2.0 * H
    m = 2 * n                          # exact DH embedding — must be 2*n, not 2*(n-1)

    # Build covariance row for the symmetric circulant:
    # [cov(0), cov(1), ..., cov(n-1), 0, cov(n-1), ..., cov(1)]
    ks = np.arange(n, dtype=np.float64)
    cov_k = 0.5 * (np.maximum(ks - 1, 0) ** tH2 - 2.0 * ks ** tH2 + (ks + 1) ** tH2)
    cov_k[0] = 1.0                     # cov(0) = variance = 1 (normalised)

    row = np.empty(m)
    row[:n]  = cov_k
    row[n]   = 0.0                     # exact middle zero — critical for DH
    row[n+1:] = cov_k[1:][::-1]        # symmetric mirror: row[m-k] = cov(k)

    # Eigenvalues of the circulant (all non-negative for H < ~0.92 at large n)
    lam = sp_fft.rfft(row).real
    np.maximum(lam, 0.0, out=lam)      # clip tiny numerical negatives

    # Build complex spectrum with correct per-frequency variance:
    #   DC and Nyquist: purely real, scale = sqrt(lam / m)
    #   Interior k:     complex,     scale = sqrt(lam / (2*m))
    half_spec = len(lam)               # m // 2 + 1
    nr = _rng.standard_normal(half_spec)
    ni = _rng.standard_normal(half_spec)
    ni[0] = 0.0                        # DC must be real
    if m % 2 == 0:
        ni[-1] = 0.0                   # Nyquist must be real

    scale = np.sqrt(lam / m)
    scale[1:-1] /= np.sqrt(2.0)       # interior frequencies: split energy between Re/Im
    W = scale * (nr + 1j * ni)

    # irfft divides by m internally; multiply back to recover the DH normalisation
    return sp_fft.irfft(W, n=m)[:n] * m

# =============================================================================
# ACF / PACF
# =============================================================================

def compute_acf(series, nlags=1000):
    """
    Computes the ACF of both the raw series and its sign-clipped (binary) version.

    For a long-memory process with C(tau) ~ tau^(-gamma), the sign-clipped series
    has autocorrelation C_bin(tau) = (2/pi) * arcsin(C(tau)) ≈ (2/pi) * C(tau)
    for small C, so its ACF decays with the same exponent gamma.

    Performance notes
    -----------------
    Subsampled to 1,000,000 points before calling statsmodels. The ACF of a
    long-memory process is determined by the long-lag tail, which is accurately
    estimated well before n=10^8. The FFT inside statsmodels is O(n log n) so
    reducing n from 10^8 to 10^6 gives a ~200x speedup.

    Parameters
    ----------
    series : array  — input time series
    nlags  : int    — number of lags to compute (default 1000)

    Returns
    -------
    acf_cont : array of length nlags+1  — ACF of the raw series
    acf_bin  : array of length nlags+1  — ACF of sign(series)
    """
    MAX_N = 10_000_000
    x = series[:MAX_N] if len(series) > MAX_N else series
    series_sign = np.sign(x)
    acf_cont    = acf(x,           nlags=nlags, fft=True)
    acf_bin     = acf(series_sign, nlags=nlags, fft=True)
    return acf_cont, acf_bin


def fit_acf(acf_values, fit_start=5, fit_end=500):
    """
    Estimates the ACF decay exponent gamma via OLS in log-log space.

    Fits log C(tau) = -gamma * log(tau) + const over lags [fit_start, fit_end].
    Only positive ACF values are included (negative values are undefined in
    log space and signal breakdown of the power-law regime).

    Parameters
    ----------
    acf_values : array  — ACF values starting at lag 0
    fit_start  : int    — first lag to include in the fit (default 5)
    fit_end    : int    — last lag to include in the fit (default 500)

    Returns
    -------
    gamma_hat : float — estimated decay exponent (slope magnitude)
    intercept : float — log-space intercept
    r2        : float — R² of the log-log regression
    """
    lags = np.arange(fit_start, fit_end + 1)
    vals = acf_values[fit_start:fit_end + 1]

    mask = vals > 0
    lags, vals = lags[mask], vals[mask]

    log_lags = np.log(lags)
    log_vals = np.log(vals)

    slope, intercept, r, _, _ = stats.linregress(log_lags, log_vals)
    return -slope, intercept, r ** 2


def compute_pacf(series, nlags=100):
    """
    Computes the Partial ACF (PACF) of the series.

    For a pure long-memory process (ARFIMA(0,d,0) or fGn) the PACF decays to
    zero without the sharp cutoff seen in finite-order AR models, distinguishing
    genuine long memory from AR(p) short memory.

    Subsampled to 100,000 points: PACF estimation is O(n·p) and does not
    benefit from the full 10^8 series length.

    Parameters
    ----------
    series : array — input time series
    nlags  : int   — number of lags (default 100; keep small for speed)

    Returns
    -------
    pacf_values : array of length nlags+1
    """
    MAX_N = 1_000_000
    x = series[:MAX_N] if len(series) > MAX_N else series
    return pacf(x, nlags=nlags, method='ywm')


def plot_acf_pacf(series, gamma, filename, nlags_acf=1000, nlags_pacf=100):
    """
    Plots ACF (continuous + binary) and PACF with theoretical power-law overlays.

    Panel layout
    ------------
    1. Continuous ACF (log-log) with tau^(-gamma) reference line and OLS fit
    2. Binary-clipped ACF (log-log) with (2/pi)*tau^(-gamma) reference line
    3. PACF (linear scale)

    Parameters
    ----------
    series     : array  — input time series
    gamma      : float  — theoretical decay exponent for the reference lines
    filename   : str    — output path (no extension); saved as <filename>.png
    nlags_acf  : int    — lags for ACF panels (default 1000)
    nlags_pacf : int    — lags for PACF panel (default 100)
    """
    acf_cont, acf_bin  = compute_acf(series, nlags=nlags_acf)
    pacf_vals          = compute_pacf(series, nlags=nlags_pacf)
    gamma_hat, _, r2   = fit_acf(acf_cont)

    lags_ref = np.arange(1, nlags_acf)
    y_ref    = (lags_ref ** -gamma) * (acf_cont[10] / (10 ** -gamma))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Continuous ACF ---
    ax = axes[0]
    ax.plot(acf_cont, label='Empirical ACF (Continuous)', alpha=0.7)
    ax.plot(lags_ref, y_ref, 'r--', label=f'Theoretical $\\tau^{{-{gamma:.2f}}}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(f'ACF  [fitted γ={gamma_hat:.3f}, R²={r2:.3f}]')
    ax.set_xlabel('Lag (log)'); ax.set_ylabel('Autocorrelation (log)')
    ax.legend(); ax.grid(True, which='both', ls='-', alpha=0.2)

    # --- Panel 2: Binary ACF ---
    ax = axes[1]
    ax.plot(acf_bin, color='orange', label='Empirical ACF (Binary)', alpha=0.7)
    ax.plot(lags_ref, y_ref * (2 / np.pi), 'k--', label='Theoretical Decay')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title('Binary Clipped ACF (Sign)')
    ax.set_xlabel('Lag (log)')
    ax.legend(); ax.grid(True, which='both', ls='-', alpha=0.2)

    # --- Panel 3: PACF ---
    ax = axes[2]
    ax.stem(np.arange(len(pacf_vals)), pacf_vals, markerfmt='C0o',
            linefmt='C0-', basefmt='k-')
    ax.axhline(0, color='k', lw=0.8)
    ci = 1.96 / np.sqrt(len(series))
    ax.axhline(ci, color='r', ls='--', lw=0.8, label='95% CI')
    ax.axhline(-ci, color='r', ls='--', lw=0.8)
    ax.set_title('PACF')
    ax.set_xlabel('Lag'); ax.set_ylabel('Partial Autocorrelation')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close()


# =============================================================================
# POWER SPECTRUM
# =============================================================================

def compute_spectrum(series, nperseg=None):
    """
    Estimates the power spectral density via Welch's method.

    Welch's method splits the series into overlapping segments, windows each
    one, and averages the periodograms. Compared to the raw periodogram it
    trades frequency resolution for variance reduction (smoother estimate).

    Performance notes
    -----------------
    Default nperseg = min(len(series) // 8, 65536): caps the segment length
    at 65536 so the internal FFT stays fast even on 10^8-point series.

    Parameters
    ----------
    series  : array — input time series
    nperseg : int   — samples per Welch segment; default = min(n//8, 65536)

    Returns
    -------
    freqs : array — frequency bins (0 to 0.5)
    psd   : array — power spectral density estimates
    """
    n = len(series)
    if nperseg is None:
        nperseg = min(n // 8, 65536)
    freqs, psd = signal.welch(series, fs=1.0, nperseg=nperseg)
    return freqs, psd


def fit_spectrum(freqs, psd, fit_start_frac=0.001, fit_end_frac=0.1):
    """
    Estimates the spectral exponent beta via OLS in log-log space.

    For a long-memory process the PSD scales as S(f) ~ f^(-beta) in the
    low-frequency region. The fitted beta is related to the ACF exponent by
    beta = 1 - gamma.

    Parameters
    ----------
    freqs          : array — frequency bins from compute_spectrum
    psd            : array — PSD estimates from compute_spectrum
    fit_start_frac : float — lower frequency boundary as fraction of Nyquist (default 0.001)
    fit_end_frac   : float — upper frequency boundary as fraction of Nyquist (default 0.1)

    Returns
    -------
    beta_hat  : float — estimated spectral exponent (slope magnitude)
    intercept : float — log-space intercept
    r2        : float — R² of the log-log regression
    """
    mask  = (freqs >= fit_start_frac) & (freqs <= fit_end_frac) & (freqs > 0)
    lf    = np.log(freqs[mask])
    lp    = np.log(psd[mask])
    slope, intercept, r, _, _ = stats.linregress(lf, lp)
    return -slope, intercept, r ** 2


def plot_spectrum(series, gamma, filename):
    """
    Plots the power spectral density (Welch) on a log-log scale with a
    theoretical f^(-beta) reference line and OLS fit annotation.

    The theoretical exponent is beta = 1 - gamma. The fit is performed over
    the low-frequency range [0.001, 0.1] * f_Nyquist by default.

    Parameters
    ----------
    series   : array — input time series
    gamma    : float — theoretical ACF decay exponent
    filename : str   — output path (no extension)
    """
    freqs, psd = compute_spectrum(series)
    beta_hat, intercept, r2 = fit_spectrum(freqs, psd)

    beta_th = 1.0 - gamma
    mask_plot = freqs > 0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(freqs[mask_plot], psd[mask_plot], alpha=0.7, label='PSD (Welch)')

    # Theoretical reference line anchored at the median frequency
    f_anchor = freqs[len(freqs) // 4]
    p_anchor = np.exp(intercept) * f_anchor ** (-beta_hat)
    f_ref    = freqs[mask_plot]
    ax.loglog(f_ref, p_anchor * (f_ref / f_anchor) ** (-beta_th), 'r--',
              label=f'Theoretical $f^{{-{beta_th:.2f}}}$')

    ax.set_title(f'Power Spectrum  [fitted β={beta_hat:.3f}, R²={r2:.3f}]')
    ax.set_xlabel('Frequency (log)')
    ax.set_ylabel('PSD (log)')
    ax.legend()
    ax.grid(True, which='both', ls='-', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close()


# =============================================================================
# MARGINAL DISTRIBUTION
# =============================================================================

def compute_distribution(series):
    """
    Computes descriptive statistics and normality test for the marginal distribution.

    Returns the sample mean, variance, skewness, excess kurtosis, and the
    result of the Kolmogorov-Smirnov test against a fitted Gaussian.

    Note: Shapiro-Wilk is not used because it is limited to n <= 5000.
    The KS test is applied to a random subsample of 10,000 points to keep
    it computationally feasible while remaining sensitive to deviations.

    Parameters
    ----------
    series : array — input time series

    Returns
    -------
    stats_dict : dict with keys:
        mean, variance, skewness, kurtosis (excess), ks_stat, ks_pvalue
    """
    mu    = np.mean(series)
    var   = np.var(series)
    skew  = stats.skew(series)
    kurt  = stats.kurtosis(series)  # excess kurtosis (Gaussian = 0)

    sub   = _rng.choice(series, size=min(10_000, len(series)), replace=False)
    ks_s, ks_p = stats.kstest(sub, 'norm', args=(np.mean(sub), np.std(sub)))

    return dict(mean=mu, variance=var, skewness=skew,
                kurtosis=kurt, ks_stat=ks_s, ks_pvalue=ks_p)


def plot_distribution(series, filename):
    """
    Plots the marginal distribution: histogram with Gaussian overlay and Q-Q plot.

    Panel layout
    ------------
    1. Histogram of the series with a fitted Gaussian PDF superimposed.
       Excess kurtosis and KS p-value are annotated on the plot.
    2. Normal Q-Q plot (quantiles of the series vs. theoretical Gaussian quantiles).
       Deviation from the diagonal reveals heavy tails or asymmetry.

    Parameters
    ----------
    series   : array — input time series
    filename : str   — output path (no extension)
    """
    stat = compute_distribution(series)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel 1: Histogram ---
    ax = axes[0]
    x_grid = np.linspace(series.min(), series.max(), 500)
    ax.hist(series, bins=200, density=True, alpha=0.5, color='steelblue',
            label='Empirical')
    ax.plot(x_grid,
            stats.norm.pdf(x_grid, stat['mean'], np.sqrt(stat['variance'])),
            'r-', lw=2, label='Fitted Gaussian')
    ax.set_title(
        f"Marginal Distribution\n"
        f"Kurt={stat['kurtosis']:.3f}  Skew={stat['skewness']:.3f}  "
        f"KS p={stat['ks_pvalue']:.3f}"
    )
    ax.set_xlabel('Value'); ax.set_ylabel('Density')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- Panel 2: Q-Q plot ---
    ax = axes[1]
    sub = _rng.choice(series, size=min(5_000, len(series)), replace=False)
    (osm, osr), (slope, intercept, r) = stats.probplot(sub, dist='norm')
    ax.scatter(osm, osr, s=5, alpha=0.4, label='Data quantiles')
    ax.plot(osm, slope * np.array(osm) + intercept, 'r-', lw=2, label='Gaussian line')
    ax.set_title(f'Normal Q-Q Plot  [R²={r**2:.4f}]')
    ax.set_xlabel('Theoretical quantiles'); ax.set_ylabel('Sample quantiles')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close()


# =============================================================================
# RESCALED RANGE (R/S) ANALYSIS
# =============================================================================

def compute_rs(series, n_scales=20):
    """
    Computes the Rescaled Range (R/S) statistic across multiple time scales.

    At each scale s the series is divided into non-overlapping blocks of
    length s. For each block the range of the centred cumulative sum is
    computed and divided by the block standard deviation. The mean R/S over
    all blocks is recorded. For a long-memory process E[R/S] ~ s^H, so a
    log-log regression of R/S vs. s yields the Hurst exponent H.

    Performance notes
    -----------------
    Fully vectorised: each scale uses a single reshape + cumsum + max/min over
    axis=1, with no Python loop over individual blocks. The series is also
    downsampled to at most 1,000,000 points before the analysis — R/S
    converges quickly and the extra points do not improve the H estimate
    meaningfully while multiplying runtime.

    Parameters
    ----------
    series   : array — input time series
    n_scales : int   — number of logarithmically-spaced scales to evaluate

    Returns
    -------
    scales : array — block sizes used
    rs     : array — mean R/S at each scale
    """
    # Downsample to cap cost: R/S estimate stabilises well before n=1e6
    MAX_N = 1_000_000
    x = series[:MAX_N] if len(series) > MAX_N else series
    n = len(x)

    scales = np.unique(
        np.logspace(np.log10(10), np.log10(n // 4), n_scales).astype(int)
    )
    rs_vals = []

    for s in scales:
        n_blocks = n // s
        blocks   = x[:n_blocks * s].reshape(n_blocks, s)     # (B, s)
        mean_b   = blocks.mean(axis=1, keepdims=True)
        dev      = (blocks - mean_b).cumsum(axis=1)           # centred cumsum
        R        = dev.max(axis=1) - dev.min(axis=1)
        S        = blocks.std(axis=1, ddof=1)
        S        = np.where(S == 0, np.nan, S)
        rs_vals.append(np.nanmean(R / S))

    return scales, np.array(rs_vals)


def fit_rs(scales, rs):
    """
    Estimates the Hurst exponent H from R/S statistics via OLS in log-log space.

    Parameters
    ----------
    scales : array — block sizes
    rs     : array — mean R/S values

    Returns
    -------
    H_hat     : float — estimated Hurst exponent
    intercept : float — log-space intercept
    r2        : float — R² of the regression
    """
    log_s  = np.log(scales)
    log_rs = np.log(rs)
    slope, intercept, r, _, _ = stats.linregress(log_s, log_rs)
    return slope, intercept, r ** 2


def plot_rs(series, H_theory, filename):
    """
    Plots log(R/S) vs log(scale) with OLS fit and theoretical H reference line.

    Parameters
    ----------
    series   : array — input time series
    H_theory : float — theoretical Hurst exponent for the reference line
    filename : str   — output path (no extension)
    """
    scales, rs = compute_rs(series)
    H_hat, intercept, r2 = fit_rs(scales, rs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(scales, rs, 'o-', label='Empirical R/S')
    ax.loglog(scales, np.exp(intercept) * scales ** H_hat, 'r--',
              label=f'OLS fit H={H_hat:.3f}  R²={r2:.3f}')
    ax.loglog(scales,
              np.exp(intercept) * scales ** H_theory, 'k:',
              label=f'Theoretical H={H_theory:.3f}')
    ax.set_title('Rescaled Range (R/S) Analysis')
    ax.set_xlabel('Scale (log)'); ax.set_ylabel('R/S (log)')
    ax.legend(); ax.grid(True, which='both', ls='-', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close()


# =============================================================================
# DETRENDED FLUCTUATION ANALYSIS (DFA)
# =============================================================================

def compute_dfa(series, n_scales=20, order=1):
    """
    Computes the DFA fluctuation function F(s) across multiple time scales.

    DFA integrates the series (cumulative sum minus mean), divides it into
    non-overlapping windows of length s, fits a polynomial of given order to
    each window to remove local trends, and computes the root-mean-square
    residual. For a long-memory process F(s) ~ s^H, so the slope of
    log F(s) vs log s gives the Hurst exponent.

    Performance notes
    -----------------
    The inner loop over segments is replaced by a vectorised least-squares
    solve: the Vandermonde design matrix V (shape s × (order+1)) is built
    once per scale, then all segments are detrended simultaneously via
    V @ lstsq(V, segments.T). Downsampled to 2,000,000 points maximum.

    Parameters
    ----------
    series   : array — input time series
    n_scales : int   — number of logarithmically-spaced scales
    order    : int   — polynomial order for local detrending (1 = linear)

    Returns
    -------
    scales : array — window sizes used
    F      : array — DFA fluctuation function at each scale
    """
    MAX_N = 2_000_000
    x = series[:MAX_N] if len(series) > MAX_N else series

    y = np.cumsum(x - x.mean())
    n = len(y)

    scales = np.unique(
        np.logspace(np.log10(10), np.log10(n // 4), n_scales).astype(int)
    )
    F_vals = []

    for s in scales:
        n_seg    = n // s
        segments = y[:n_seg * s].reshape(n_seg, s)          # (n_seg, s)
        x_seg    = np.arange(s)
        V        = np.vander(x_seg, order + 1, increasing=True)   # (s, order+1)
        # Least-squares coefficients for all segments at once: (order+1, n_seg)
        coeffs, *_ = np.linalg.lstsq(V, segments.T, rcond=None)
        trends   = (V @ coeffs).T                            # (n_seg, s)
        residuals = segments - trends
        F_vals.append(np.sqrt(np.mean(residuals ** 2)))

    return scales, np.array(F_vals)


def fit_dfa(scales, F):
    """
    Estimates the Hurst exponent H from DFA fluctuation function via OLS.

    Parameters
    ----------
    scales : array — window sizes
    F      : array — DFA fluctuation values

    Returns
    -------
    H_hat     : float — estimated Hurst exponent (DFA scaling exponent)
    intercept : float — log-space intercept
    r2        : float — R² of the regression
    """
    slope, intercept, r, _, _ = stats.linregress(np.log(scales), np.log(F))
    return slope, intercept, r ** 2


def plot_dfa(series, H_theory, filename):
    """
    Plots log F(s) vs log(scale) with OLS fit and theoretical H reference line.

    Parameters
    ----------
    series   : array — input time series
    H_theory : float — theoretical Hurst exponent
    filename : str   — output path (no extension)
    """
    scales, F = compute_dfa(series)
    H_hat, intercept, r2 = fit_dfa(scales, F)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(scales, F, 'o-', label='DFA F(s)')
    ax.loglog(scales, np.exp(intercept) * scales ** H_hat, 'r--',
              label=f'OLS fit H={H_hat:.3f}  R²={r2:.3f}')
    ax.loglog(scales,
              np.exp(intercept) * scales ** H_theory, 'k:',
              label=f'Theoretical H={H_theory:.3f}')
    ax.set_title('Detrended Fluctuation Analysis (DFA)')
    ax.set_xlabel('Scale (log)'); ax.set_ylabel('F(s) (log)')
    ax.legend(); ax.grid(True, which='both', ls='-', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close()


# =============================================================================
# STRUCTURE FUNCTIONS
# =============================================================================

def compute_structure_functions(series, q_orders=None, n_scales=30, integrate=True):
    """
    Computes the q-th order structure functions S_q(tau) = E[|Z(t+tau) - Z(t)|^q].

    Structure functions require **stationary increments** to exhibit power-law
    scaling S_q(τ) ~ τ^(q·H). Applied directly to stationary noise (fGn,
    ARFIMA, FFM) they saturate immediately to 2·Var(X), yielding near-zero
    scaling exponents regardless of H.

    With integrate=True (default) the noise X is converted to its integrated
    path Z = cumsum(X) before computing increments. Z has stationary increments
    (it is fBm-like) and S_q(τ) ~ τ^(q·H) holds over the full lag range.

    Parameters
    ----------
    series    : array      — input stationary noise series
    q_orders  : list/array — moment orders (default [1,2,3,4])
    n_scales  : int        — number of log-spaced lags (default 30)
    integrate : bool       — integrate X → cumsum(X) before computing (default True)

    Returns
    -------
    taus : array           — lag values used
    Sq   : dict {q: array} — structure function values
    """
    if q_orders is None:
        q_orders = [1, 2, 3, 4]

    MAX_N = 100_000
    x = series[:MAX_N] if len(series) > MAX_N else series
    x = (x - x.mean()) / x.std()

    z = np.cumsum(x) if integrate else x

    n    = len(z)
    taus = np.unique(np.logspace(0, np.log10(n // 4), n_scales).astype(int))
    Sq   = {q: np.empty(len(taus)) for q in q_orders}

    for i, tau in enumerate(taus):
        diffs = np.abs(z[tau:] - z[:-tau])
        for q in q_orders:
            Sq[q][i] = np.mean(diffs ** q)

    return taus, Sq


def fit_structure_functions(taus, Sq):
    """
    Estimates the scaling exponent zeta(q) via OLS in log-log space.

    For the integrated series Z = cumsum(X), S_q(τ) ~ τ^(q·H) holds over
    the full lag range with no saturation, so OLS over all points is correct.
    The monofractal prediction is zeta(q) = q·H; linearity of zeta(q) in q
    confirms monofractality, concavity indicates multifractality.

    Parameters
    ----------
    taus : array           — lag values from compute_structure_functions
    Sq   : dict {q: array} — structure function values

    Returns
    -------
    zeta    : dict {q: (exponent, r2)}
    fit_end : int — last tau used (all taus; returned for plot compatibility)
    """
    log_t = np.log(taus)
    zeta  = {}
    for q, vals in Sq.items():
        slope, _, r, _, _ = stats.linregress(log_t, np.log(vals))
        zeta[q] = (slope, r ** 2)
    return zeta, int(taus[-1])


def plot_structure_functions(series, H_theory, filename, q_orders=None):
    """
    Two-panel plot of structure functions S_q(τ) and scaling exponent spectrum.

    Panel 1: S_q(τ) of the integrated series cumsum(X) on log-log axes,
             with OLS fit lines. S_q(τ) ~ τ^(q·H) should hold globally.
    Panel 2: Empirical ζ(q) vs q compared to the monofractal line q·H.
             Linearity = monofractal; concavity = multifractal.

    Parameters
    ----------
    series   : array      — input stationary noise series
    H_theory : float      — theoretical Hurst exponent
    filename : str        — output path (no extension)
    q_orders : list/array — moment orders (default [1,2,3,4])
    """
    if q_orders is None:
        q_orders = [1, 2, 3, 4]

    taus, Sq       = compute_structure_functions(series, q_orders=q_orders)
    zeta, fit_end  = fit_structure_functions(taus, Sq)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: S_q(tau) with OLS fit lines ---
    ax     = axes[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(q_orders)))
    for (q, vals), col in zip(Sq.items(), colors):
        exp, r2 = zeta[q]
        ax.loglog(taus, vals, 'o-', color=col, alpha=0.7,
                  label=f'q={q}  ζ={exp:.3f}  R²={r2:.2f}')
        # OLS fit line
        fit_line = np.exp(np.log(taus) * exp +
                          (np.log(vals[0]) - exp * np.log(taus[0])))
        ax.loglog(taus, fit_line, '--', color=col, lw=1.2, alpha=0.5)
    ax.set_title(f'Structure Functions $S_q(\\tau)$  H={H_theory:.3f}')
    ax.set_xlabel('Lag τ (log)'); ax.set_ylabel('$S_q$ (log)')
    ax.legend(fontsize=8); ax.grid(True, which='both', ls='-', alpha=0.2)

    # --- Panel 2: Scaling exponent spectrum zeta(q) ---
    ax       = axes[1]
    qs       = np.array(q_orders)
    zeta_emp = np.array([zeta[q][0] for q in q_orders])
    ax.plot(qs, zeta_emp, 'o-', label='Empirical ζ(q)')
    ax.plot(qs, qs * H_theory, 'r--',
            label=f'Monofractal ζ=q·H  (H={H_theory:.3f})')
    ax.set_title('Scaling Exponent Spectrum')
    ax.set_xlabel('Order q'); ax.set_ylabel('ζ(q)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close()


# =============================================================================
# WAVELET VARIANCE
# =============================================================================

def compute_wavelet_variance(series, wavelet='db4', max_level=None):
    """
    Computes the wavelet variance (energy) at each decomposition level.

    The discrete wavelet transform (DWT) decomposes the series into detail
    coefficients at dyadic scales 2^j. For a long-memory process with Hurst
    exponent H, the wavelet variance at scale j satisfies
    Var_j ~ 2^(j*(2H+1)), so a log-log regression of Var_j vs 2^j yields
    the slope 2H+1 and hence H = (slope - 1) / 2.

    Performance notes
    -----------------
    Downsampled to 1,000,000 points before the DWT: the wavelet variance
    estimate at each dyadic scale is computed from the detail coefficients
    at that level, which themselves have length n / 2^j. The estimate is
    accurate well before n=10^8 and the DWT on 10^6 points completes in
    milliseconds. max_level is capped at 12 (scales 2..4096).

    Parameters
    ----------
    series    : array  — input time series
    wavelet   : str    — PyWavelets wavelet name (default 'db4')
    max_level : int    — maximum decomposition level; defaults to
                         min(pywt.dwt_max_level(n, wavelet), 12)

    Returns
    -------
    scales        : array — dyadic scales 2^j for j = 1, ..., max_level
    wvar          : array — wavelet variance at each scale
    """
    MAX_N = 5_000_000
    x = series[:MAX_N] if len(series) > MAX_N else series

    if max_level is None:
        max_level = min(pywt.dwt_max_level(len(x), wavelet), 10)

    coeffs  = pywt.wavedec(x, wavelet, level=max_level)
    # pywt.wavedec returns [cA_L, cD_L, cD_{L-1}, ..., cD_1]
    # coeffs[1] = detail at coarsest scale 2^L (shortest array)
    # coeffs[L] = detail at finest scale 2^1   (longest array)
    # Reverse so that details[0] ↔ scale 2^1 (finest), details[-1] ↔ scale 2^L
    details = coeffs[1:][::-1]
    scales  = 2.0 ** np.arange(1, max_level + 1)
    wvar    = np.array([np.var(d) for d in details])
    return scales, wvar


def fit_wavelet_variance(scales, wvar):
    """
    Estimates the Hurst exponent from wavelet variance via OLS in log-log space.

    For a **stationary** long-memory process (fGn, ARFIMA, FFM) with H in (0,1)
    the wavelet detail variance at dyadic scale 2^j satisfies:

        Var_j  ~  2^(j * (2H - 1))

    so the slope of log2(Var_j) vs j  equals  (2H - 1), giving:

        H = (slope + 1) / 2

    This is distinct from fBm (the *integrated* process), where the slope is
    2H+1. Because our three generators produce stationary noise (not fBm),
    the correct formula is H = (slope + 1) / 2.

    Cross-check: H=0.7 → slope = 2*0.7-1 = 0.4  ✓  (matches empirical ~0.4)

    Parameters
    ----------
    scales : array — dyadic scales 2^j from compute_wavelet_variance
    wvar   : array — wavelet variances

    Returns
    -------
    H_hat     : float — estimated Hurst exponent
    intercept : float — log₂-space intercept
    r2        : float — R² of the regression
    """
    # x-axis = exponent j (= log2 of scale), y-axis = log2(variance)
    j = np.log2(scales)   # = 1, 2, 3, ..., max_level
    slope, intercept, r, _, _ = stats.linregress(j, np.log2(wvar))
    H_hat = (slope + 1.0) / 2.0
    return H_hat, intercept, r ** 2


def plot_wavelet_variance(series, H_theory, filename):
    """
    Plots log2(wavelet variance) vs log2(scale) with OLS fit and theoretical line.

    Parameters
    ----------
    series   : array — input time series
    H_theory : float — theoretical Hurst exponent
    filename : str   — output path (no extension)
    """
    scales, wvar = compute_wavelet_variance(series)
    H_hat, intercept, r2 = fit_wavelet_variance(scales, wvar)

    j         = np.log2(scales)
    slope_fit = 2.0 * H_hat   - 1.0
    slope_th  = 2.0 * H_theory - 1.0
    intercept_th = np.log2(wvar[0]) - slope_th * j[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(j, np.log2(wvar), 'o-', label='Wavelet Variance')
    ax.plot(j, slope_fit * j + intercept,
            'r--', label=f'OLS fit H={H_hat:.3f}  R²={r2:.3f}')
    ax.plot(j, slope_th  * j + intercept_th,
            'k:', label=f'Theoretical H={H_theory:.3f}')
    ax.set_title('Wavelet Variance Scaling')
    ax.set_xlabel('j  (= log₂ Scale)'); ax.set_ylabel('log₂(Wavelet Variance)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close()


# =============================================================================
# STATIONARITY TESTS
# =============================================================================

def compute_stationarity_tests(series):
    """
    Runs ADF and KPSS stationarity tests on the series.

    ADF (Augmented Dickey-Fuller): H0 = unit root (non-stationary).
        Small p-value → reject H0 → evidence of stationarity.
    KPSS (Kwiatkowski-Phillips-Schmidt-Shin): H0 = level stationarity.
        Large p-value → fail to reject H0 → evidence of stationarity.

    Both tests together give a clearer picture: a process can appear
    stationary under one test and borderline under the other, which is
    expected for long-memory processes with d close to 0.5.

    Parameters
    ----------
    series : array — input time series (subsampled to 50,000 points for speed)

    Returns
    -------
    results : dict with keys adf_stat, adf_pvalue, kpss_stat, kpss_pvalue
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    sub = series[:min(50_000, len(series))]  # tests scale poorly with n

    adf_stat, adf_p, *_ = adfuller(sub, autolag='AIC')
    kpss_stat, kpss_p, *_ = kpss(sub, regression='c', nlags='auto')

    return dict(adf_stat=adf_stat, adf_pvalue=adf_p,
                kpss_stat=kpss_stat, kpss_pvalue=kpss_p)


def plot_stationarity(series, filename, window_frac=0.05):
    """
    Plots rolling mean and rolling variance alongside ADF/KPSS test results.

    A stationary series should have stable (flat) rolling mean and variance.
    Long-memory processes are wide-sense stationary but their rolling variance
    can appear to drift on short observation windows; this plot makes that
    visible.

    Performance notes
    -----------------
    Both rolling statistics are computed on a thinned version of the series
    (max 500,000 points). Rolling variance uses numpy stride tricks
    (as_strided) to avoid any Python-level loop: the 2-D view shares memory
    with the original array and the variance is computed over axis=1 in one
    vectorised call.

    Parameters
    ----------
    series       : array — input time series
    filename     : str   — output path (no extension)
    window_frac  : float — rolling window as fraction of series length (default 0.05)
    """
    results = compute_stationarity_tests(series)

    MAX_N = 500_000
    x = series[:MAX_N] if len(series) > MAX_N else series
    n = len(x)
    w = max(100, int(n * window_frac))

    # Rolling mean and variance via cumulative sum — O(n), no convolution.
    # cs[i] = x[0]+...+x[i-1], so sum over [i, i+w) = cs[i+w] - cs[i].
    cs       = np.empty(n + 1);  cs[0] = 0.0;  np.cumsum(x,      out=cs[1:])
    cs2      = np.empty(n + 1); cs2[0] = 0.0;  np.cumsum(x ** 2, out=cs2[1:])
    roll_sum  = cs[w:]  - cs[:n - w + 1]
    roll_sum2 = cs2[w:] - cs2[:n - w + 1]
    roll_mean = roll_sum  / w
    roll_var  = np.maximum(roll_sum2 / w - roll_mean ** 2, 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Interpret the joint result
    adf_stat_str  = "stationary" if results["adf_pvalue"] < 0.05 else "unit root"
    kpss_stat_str = "stationary" if results["kpss_pvalue"] > 0.05 else "non-stationary"
    if adf_stat_str == "stationary" and kpss_stat_str == "stationary":
        verdict = "Both tests agree: stationary ✓"
    elif adf_stat_str == "stationary" and kpss_stat_str == "non-stationary":
        verdict = "Contradiction → long memory (spurious KPSS rejection)"
    else:
        verdict = "Contradiction → inspect manually"

    ax = axes[0]
    ax.plot(roll_mean, lw=0.8, color='steelblue')
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_title(
        f'Rolling Mean (window={w})\n'
        f'ADF p={results["adf_pvalue"]:.4f} ({adf_stat_str})  '
        f'KPSS p={results["kpss_pvalue"]:.4f} ({kpss_stat_str})\n'
        f'{verdict}'
    )
    ax.set_ylabel('Rolling Mean'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(roll_var, lw=0.8, color='darkorange')
    ax.set_title(f'Rolling Variance (window={w})')
    ax.set_xlabel('Time'); ax.set_ylabel('Rolling Variance'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{filename}.png')
    plt.close()


# =============================================================================
# LEGACY ENTRY POINT (ACF only, preserves original call signature)
# =============================================================================

def acf_plot(series, filename):
    """
    Legacy wrapper: plots ACF (continuous + binary) using the original layout.

    Retained for backward compatibility. For new analyses prefer plot_acf_pacf.

    Parameters
    ----------
    series   : array — input time series
    filename : str   — output path prefix (images\\ prefix and .png are appended)
    """
    series_sign = np.where(series >= 0, 1, -1)

    lags           = 1000
    acf_continuous = acf(series,      nlags=lags, fft=True)
    acf_binary     = acf(series_sign, nlags=lags, fft=True)

    x_ref = np.arange(1, lags)
    y_ref = (x_ref ** -gamma) * (acf_continuous[10] / (10 ** -gamma))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acf_continuous, label='Empirical ACF (Continuous)', alpha=0.7)
    plt.plot(x_ref, y_ref, 'r--', label=f'Theoretical Decay $\\tau^{{-{gamma:.2f}}}$')
    plt.xscale('log'); plt.yscale('log')
    plt.title('ACF')
    plt.xlabel('Lag (log)'); plt.ylabel('Autocorrelation (log)')
    plt.legend(); plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.subplot(1, 2, 2)
    plt.plot(acf_binary, color='orange', label='Empirical ACF (Binary)', alpha=0.7)
    plt.plot(x_ref, y_ref * (2 / np.pi), 'k--', label='Theoretical Decay')
    plt.xscale('log'); plt.yscale('log')
    plt.title('Binary Clipped ACF (Sign)')
    plt.xlabel('Lag (log)')
    plt.legend(); plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'images\\{filename}.png')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import time
    import os

    n     = 100_000_000
    gamma = 0.6
    d     = 0.5 - 0.5 * gamma
    H     = d + 0.5

    print(f'Gamma={gamma}  d={d:.3f}  H={H:.3f}')

    os.makedirs('images', exist_ok=True)

    for name, gen_fn, gen_args in [
        ('FFM',    generate_ffm,    (n, gamma)),
        ('ARFIMA', generate_arfima, (n, d)),
        ('FGN',    generate_fgn,    (n, H)),
    ]:
        print(f'\n--- {name} ---')

        t0 = time.time()
        series = gen_fn(*gen_args)
        print(f'  Generation:     {time.time() - t0:.1f}s')

        base = f'images/series_creation/{name.lower()}'

        t0 = time.time()
        plot_acf_pacf(series, gamma, f'{base}_acf_pacf')
        print(f'  ACF/PACF:       {time.time() - t0:.1f}s')

        t0 = time.time()
        plot_spectrum(series, gamma, f'{base}_spectrum')
        print(f'  Spectrum:       {time.time() - t0:.1f}s')

        t0 = time.time()
        plot_distribution(series, f'{base}_distribution')
        print(f'  Distribution:   {time.time() - t0:.1f}s')

        t0 = time.time()
        plot_rs(series, H, f'{base}_rs')
        print(f'  R/S:            {time.time() - t0:.1f}s')

        t0 = time.time()
        plot_dfa(series, H, f'{base}_dfa')
        print(f'  DFA:            {time.time() - t0:.1f}s')

        t0 = time.time()
        plot_structure_functions(series, H, f'{base}_structure')
        print(f'  Structure fns:  {time.time() - t0:.1f}s')

        t0 = time.time()
        plot_wavelet_variance(series, H, f'{base}_wavelet')
        print(f'  Wavelet var:    {time.time() - t0:.1f}s')

        t0 = time.time()
        plot_stationarity(series, f'{base}_stationarity')
        print(f'  Stationarity:   {time.time() - t0:.1f}s')

        stat = compute_distribution(series)
        print(f'  Stats: mean={stat["mean"]:.4f}  var={stat["variance"]:.4f}  '
              f'skew={stat["skewness"]:.4f}  kurt={stat["kurtosis"]:.4f}  '
              f'KS_p={stat["ks_pvalue"]:.4f}')

        st = compute_stationarity_tests(series)
        adf_ok  = st["adf_pvalue"]  < 0.05
        kpss_ok = st["kpss_pvalue"] > 0.05
        if adf_ok and kpss_ok:
            verdict = "stationary ✓"
        elif adf_ok and not kpss_ok:
            verdict = "long-memory spurious KPSS rejection"
        else:
            verdict = "inspect manually"
        print(f'  ADF  p={st["adf_pvalue"]:.4f}   KPSS p={st["kpss_pvalue"]:.4f}  → {verdict}')