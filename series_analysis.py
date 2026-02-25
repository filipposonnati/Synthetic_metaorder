import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def linear_func(x, slope, intercept):
    """Standard linear function for curve fitting in log-log space."""
    return intercept + slope * x

def ensure_dir(path):
    """Ensures that the directory for saving plots exists."""
    if not os.path.exists(path):
        os.makedirs(path)

# =============================================================================
# CORE LOGIC
# =============================================================================

def generate_correlated_sign_series(n_points, gamma):
    """Generates a binary series with power-law correlation C(t) ~ t^-gamma."""
    white_noise = np.random.normal(size=n_points)
    freq_dom = np.fft.fft(white_noise)
    freqs = np.fft.fftfreq(n_points)
    
    beta = 1 - gamma
    
    with np.errstate(divide='ignore', invalid='ignore'):
        filter_array = np.power(np.abs(freqs), -(beta / 2.0))
    
    filter_array[0] = 0 
    correlated_gauss = np.real(np.fft.ifft(freq_dom * filter_array))
    return np.where(correlated_gauss >= 0, 1, -1)

def compute_runs(sign_series):
    """Computes the lengths of consecutive identical symbols."""
    changes = np.diff(sign_series) != 0
    change_indices = np.where(changes)[0]
    run_lengths = np.diff(np.concatenate([[-1], change_indices, [len(sign_series) - 1]]))
    return run_lengths

# =============================================================================
# DUAL ANALYSIS (ACF & RUNS)
# =============================================================================

def perform_analysis(gamma_target, n_total=10000000):
    # 1. Generation
    sign_series = generate_correlated_sign_series(2 * n_total, gamma_target)[:n_total]

    # 2. Data Preparation
    acf_values = acf(sign_series, nlags=1000, fft=True)
    runs = compute_runs(sign_series)
    bins = np.linspace(1.0, 1000.0, 1001)
    hist, _ = np.histogram(runs, bins=bins)
    
    x_axis = np.arange(1, len(acf_values) + 1)
    ensure_dir('images/series')

    # --- 3. ACF DUAL FITTING (Head vs Tail) ---
    # Head: lags 2 to 10 | Tail: lags 50 to 500
    mask_acf_head = (x_axis >= 2) & (x_axis <= 10)
    mask_acf_tail = (x_axis >= 50) & (x_axis <= 500)
    
    popt_acf_h, _ = curve_fit(linear_func, np.log(x_axis[mask_acf_head]), np.log(acf_values[mask_acf_head]))
    popt_acf_t, _ = curve_fit(linear_func, np.log(x_axis[mask_acf_tail]), np.log(acf_values[mask_acf_tail]))

    # --- 4. RUNS DUAL FITTING (Head vs Tail) ---
    x_hist = np.linspace(1.0, len(hist), len(hist))
    mask_run_head = (hist > 0) & (x_hist < 5)
    mask_run_tail = (hist > 0) & (x_hist > 20)
    
    popt_run_h, _ = curve_fit(linear_func, np.log(x_hist[mask_run_head]), np.log(hist[mask_run_head]))
    popt_run_t, _ = curve_fit(linear_func, np.log(x_hist[mask_run_tail]), np.log(hist[mask_run_tail]))

    # --- 5. VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot ACF
    ax1.loglog(x_axis, acf_values, 'o', markersize=3, alpha=0.4, label='ACF Data')
    x_fit = np.geomspace(1, 1000, 100)
    ax1.plot(x_fit, np.exp(popt_acf_h[1]) * (x_fit**popt_acf_h[0]), 'g--', label=f'Head $\gamma$: {-popt_acf_h[0]:.2f}')
    ax1.plot(x_fit, np.exp(popt_acf_t[1]) * (x_fit**popt_acf_t[0]), 'r-', label=f'Tail $\gamma$: {-popt_acf_t[0]:.2f}')
    ax1.set_title(f"ACF Analysis (Target $\gamma$={gamma_target})")
    ax1.legend()

    # Plot Runs
    ax2.loglog(x_hist, hist, 'o', markersize=3, alpha=0.4, label='Run Data')
    ax2.plot(x_fit, np.exp(popt_run_h[1]) * (x_fit**popt_run_h[0]), 'g--', label=f'Head $\mu$: {-popt_run_h[0]:.2f}')
    ax2.plot(x_fit, np.exp(popt_run_t[1]) * (x_fit**popt_run_t[0]), 'r-', label=f'Tail $\mu$: {-popt_run_t[0]:.2f}')
    ax2.set_title("Run-Length Distribution")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('images/series/dual_analysis.png')
    plt.close()

    # Return slopes for statistics (gamma = -slope)
    return -popt_acf_h[0], -popt_acf_t[0], -popt_run_h[0], -popt_run_t[0]

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ITERATIONS = 3
    gamma_in = 0.7
    
    results = [] # Store results as tuples
     
    for i in range(ITERATIONS):
        print(f"Iteration {i+1}...")
        res = perform_analysis(gamma_in)
        results.append(res)
    
    results = np.array(results)
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)

    labels = ["ACF Gamma (Head)", "ACF Gamma (Tail)", "Run Mu (Head)", "Run Mu (Tail)"]
    
    print("\n" + "="*40)
    print(f"{'Metric':<20} | {'Mean':<10} | {'Std':<10}")
    print("-"*40)
    for i in range(4):
        print(f"{labels[i]:<20} | {means[i]:<10.4f} | {stds[i]:<10.4f}")