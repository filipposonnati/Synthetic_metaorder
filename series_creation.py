import numpy as np
from scipy import signal
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

def generate_ffm(n_points, gamma):
    """Genera serie binaria con memoria lunga usando FFM e zero-padding."""
    beta = 1.0 - gamma
    N = 2 * n_points 
    
    noise = np.random.normal(size=N)
    freq_dom = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(N)
    freqs[0] = 1e-10 
    
    filter_array = freqs ** (-beta / 2.0)
    filter_array[0] = 0.0 
    
    correlated_gauss = np.fft.irfft(freq_dom * filter_array, n=N)
    continuous_series = correlated_gauss[:n_points]
    
    return continuous_series

def generate_arfima(n, d, ar_params=None, ma_params=None, std_dev=1.0):
    """
    Generates an ARFIMA(p, d, q) process.
    
    Parameters:
    - n: Number of samples
    - d: Fractional integration parameter (|d| < 0.5 for stationarity)
    - ar_params: List/array of AR coefficients [phi_1, phi_2, ...]
    - ma_params: List/array of MA coefficients [theta_1, theta_2, ...]
    - std_dev: Standard deviation of the white noise
    """
    # 1. Generate Weights for (1-L)^{-d}
    # Length M should be large to capture long memory; n is a safe bet
    M = n
    k = np.arange(1, M)
    # Correct recursion for (1-L)^-d is (k - 1 + d) / k
    w = np.concatenate(([1.0], np.cumprod((k - 1 + d) / k)))
    
    # 2. Generate Innovation Noise (with padding to handle transient)
    eps = np.random.normal(0, std_dev, size=n + M)
    
    # 3. Apply Fractional Integration via Convolution
    # This creates the ARFIMA(0, d, 0) component
    x_fractional = np.convolve(eps, w, mode='valid')[:n]
    
    # 4. Apply ARMA(p, q) filter
    # Statsmodels/Scipy use: polynomial A(z)x = B(z)eps
    # AR: [1, -phi_1, -phi_2...] | MA: [1, theta_1, theta_2...]
    
    ar_poly = np.r_[1, -np.array(ar_params)] if ar_params is not None else [1]
    ma_poly = np.r_[1, np.array(ma_params)] if ma_params is not None else [1]
    
    # Apply the linear filter
    arfima_process = signal.lfilter(ma_poly, ar_poly, x_fractional)
    
    return arfima_process

if __name__ == "__main__":
    n = 1000000
    d = 0.2  # Set to 0.3 to stay within stationary bounds (d < 0.5)
    gamma = 1 - 2*d
    print(f'Parameters: d = {d}, expected gamma (ACF decay) = {gamma}')

    # 1. Generate Series
    series = generate_arfima(n, d)
    series_sign = np.where(series >= 0, 1, -1)

    # 2. Compute ACFs
    lags = 1000 # Increased lags to see the long-term power law better
    acf_continuous = acf(series, nlags=lags, fft=True)
    acf_binary = acf(series_sign, nlags=lags, fft=True)

    # 3. Create Reference Power Law Line: ACF(tau) ~ tau^-gamma
    # We skip lag 0 (index 0) because log(0) is undefined
    x_ref = np.arange(1, lags) 
    # Scale the reference line to match the magnitude of the ACF at lag 10
    y_ref = (x_ref**-gamma) * (acf_continuous[10] / (10**-gamma))

    # 4. Plotting
    plt.figure(figsize=(12, 5))

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

    plt.subplot(1, 2, 2)
    plt.plot(acf_binary, color='orange', label='Empirical ACF (Binary)', alpha=0.7)
    # Binary ACF follows the same exponent but scaled by 2/pi (Rice's Formula)
    plt.plot(x_ref, y_ref * (2/np.pi), 'k--', label='Theoretical Decay (Scaled)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Binary Clipped ACF (Sign)')
    plt.xlabel('Lag (log)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig('images\\series\\generated_series.png')
    plt.show()

    