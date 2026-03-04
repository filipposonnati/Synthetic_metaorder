import numpy as np
from scipy import signal
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

def generate_ffm(n_points, gamma):
    """
    Generates a long-memory continuous series using the Fractional Filtering Method (FFM).

    The idea is to shape white noise in the frequency domain so that the resulting
    signal has a power spectrum S(f) ~ f^(-beta), which produces an autocorrelation
    function that decays as C(tau) ~ tau^(-gamma).

    The relationship between the spectral exponent beta and the correlation exponent
    gamma is: beta = 1 - gamma.

    Zero-padding (doubling the series length before filtering) is used to reduce
    circular correlation artifacts that arise from the FFT assuming periodicity.

    Parameters
    ----------
    n_points : int   — desired length of the output series
    gamma    : float — target autocorrelation decay exponent (C(tau) ~ tau^-gamma)

    Returns
    -------
    continuous_series : array of length n_points
    """
    # Spectral exponent: relates the power spectrum shape to the correlation exponent
    beta = 1.0 - gamma

    # Double the length for zero-padding: reduces wrap-around artifacts from the FFT
    N = 2 * n_points

    # --- Step 1: Generate white Gaussian noise ---
    noise = np.random.normal(size=N)

    # --- Step 2: Move to frequency domain ---
    freq_dom = np.fft.rfft(noise)

    # Compute the corresponding frequency bins for the rfft output
    freqs = np.fft.rfftfreq(N)

    # Avoid division by zero at the DC component (f=0)
    freqs[0] = 1e-10

    # --- Step 3: Build the spectral filter ---
    # Multiplying by f^(-beta/2) in amplitude is equivalent to shaping the
    # power spectrum as S(f) ~ f^(-beta), since power = amplitude^2
    filter_array = freqs ** (-beta / 2.0)

    # Zero out the DC component to ensure the series has zero mean
    filter_array[0] = 0.0

    # --- Step 4: Apply filter and return to time domain ---
    correlated_gauss = np.fft.irfft(freq_dom * filter_array, n=N)

    # Keep only the first n_points to discard the zero-padded tail
    continuous_series = correlated_gauss[:n_points]

    return continuous_series


def generate_arfima(n, d, ar_params=None, ma_params=None, std_dev=1.0):
    """
    Generates a realization of an ARFIMA(p, d, q) process.

    ARFIMA (AutoRegressive Fractionally Integrated Moving Average) is a
    generalization of ARIMA where the differencing order d is allowed to be
    fractional. For |d| < 0.5 the process is stationary and exhibits long-range
    dependence, with autocorrelation decaying as C(tau) ~ tau^(2d-1).

    The generation proceeds in three stages:
      1. Fractional integration: convolve white noise with the infinite-order
         MA representation of (1-L)^{-d}, truncated to length n.
      2. Optionally apply an ARMA(p,q) filter on top.

    Parameters
    ----------
    n         : int   — number of samples to generate
    d         : float — fractional integration parameter (|d| < 0.5 for stationarity)
    ar_params : list  — AR coefficients [phi_1, phi_2, ...] (optional)
    ma_params : list  — MA coefficients [theta_1, theta_2, ...] (optional)
    std_dev   : float — standard deviation of the driving white noise (default: 1.0)

    Returns
    -------
    arfima_process : array of length n
    """
    # --- Step 1: Compute the MA(inf) weights for fractional integration (1-L)^{-d} ---
    # The weights follow the recursion: w_k = w_{k-1} * (k - 1 + d) / k
    # This is the binomial series expansion of (1-L)^{-d}, truncated at length n.
    M = n
    k = np.arange(1, M)
    w = np.concatenate(([1.0], np.cumprod((k - 1 + d) / k)))

    # --- Step 2: Generate white noise innovations ---
    # Extra M points are prepended to handle the transient (warm-up) of the convolution
    eps = np.random.normal(0, std_dev, size=n + M)

    # --- Step 3: Apply fractional integration via convolution ---
    # Convolving the noise with the weight vector w produces an ARFIMA(0, d, 0) series.
    # 'valid' mode discards the transient at the beginning, keeping exactly n points.
    x_fractional = np.convolve(eps, w, mode='valid')[:n]

    # --- Step 4: Apply the optional ARMA(p, q) filter ---
    # scipy's lfilter implements: A(z) * y = B(z) * x, where
    #   A(z) = 1 - phi_1*z^-1 - phi_2*z^-2 - ...  (AR polynomial)
    #   B(z) = 1 + theta_1*z^-1 + theta_2*z^-2 + ... (MA polynomial)
    ar_poly = np.r_[1, -np.array(ar_params)] if ar_params is not None else [1]
    ma_poly = np.r_[1, np.array(ma_params)]  if ma_params is not None else [1]

    arfima_process = signal.lfilter(ma_poly, ar_poly, x_fractional)

    return arfima_process


# =============================================================================
# MAIN — quick validation of both generators
# =============================================================================
if __name__ == "__main__":
    n = 1000000
    d = 0.3  # Fractional integration parameter (must satisfy |d| < 0.5 for stationarity)

    # The ACF of an ARFIMA(0,d,0) process decays as tau^(-(1-2d)) = tau^(-gamma)
    gamma = 1 - 2 * d
    print(f'Parameters: d = {d}, expected gamma (ACF decay) = {gamma}')

    # --- Generate series ---
    series = generate_arfima(n, d)

    # Binarize the continuous series by taking the sign: +1 if >= 0, -1 otherwise
    series_sign = np.where(series >= 0, 1, -1)

    # --- Compute autocorrelation functions ---
    lags = 1000
    acf_continuous = acf(series,      nlags=lags, fft=True)
    acf_binary     = acf(series_sign, nlags=lags, fft=True)

    # --- Build theoretical reference decay line: ACF(tau) ~ tau^-gamma ---
    # Start from lag 1 (lag 0 is always 1 and undefined in log scale)
    x_ref = np.arange(1, lags)

    # Rescale the theoretical line to match the empirical ACF at lag 10
    y_ref = (x_ref ** -gamma) * (acf_continuous[10] / (10 ** -gamma))

    # --- Plotting ---
    plt.figure(figsize=(12, 5))

    # Left panel: ACF of the continuous (Gaussian) ARFIMA series
    plt.subplot(1, 2, 1)
    plt.plot(acf_continuous, label='Empirical ACF (Continuous)', alpha=0.7)
    plt.plot(x_ref, y_ref, 'r--', label=f'Theoretical Decay $\\tau^{{-{gamma:.2f}}}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'ARFIMA(0, {d}, 0) ACF')
    plt.xlabel('Lag (log)')
    plt.ylabel('Autocorrelation (log)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Right panel: ACF of the binary (sign-clipped) version of the series.
    # By Rice's formula, clipping a Gaussian process to its sign preserves the
    # power-law exponent but scales the ACF by a factor of 2/pi.
    plt.subplot(1, 2, 2)
    plt.plot(acf_binary, color='orange', label='Empirical ACF (Binary)', alpha=0.7)
    plt.plot(x_ref, y_ref * (2 / np.pi), 'k--', label='Theoretical Decay')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Binary Clipped ACF (Sign)')
    plt.xlabel('Lag (log)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig('images\\series\\generated_series.png')
    plt.show()