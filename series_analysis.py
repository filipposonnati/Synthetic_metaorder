import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf
import powerlaw

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

    # --- 3. ACF DUAL FITTING ---
    mask_acf_tail = (x_axis >= 100)
    
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
    ax1.plot(x_fit, np.exp(popt_acf_t[1]) * (x_fit**popt_acf_t[0]), 'r-', label=f'Tail $\gamma$: {-popt_acf_t[0]:.2f}')
    ax1.set_title(f"ACF Analysis (Target $\gamma$={gamma_target})")
    ax1.legend()

    # --- All'interno di perform_analysis ---

    # 1. Fit automatico (Truncated Power Law)
    fit = powerlaw.Fit(runs, discrete=True)
    mu_auto = fit.truncated_power_law.alpha
    xmin_auto = fit.xmin

    # 2. Istogramma Lineare (Mantenendo linspace come richiesto)
    # Usiamo un numero di bin fisso o proporzionale al range
    bins_linear = np.linspace(min(runs), max(runs), 100) # Ridotto a 100 per leggibilità
    counts, bin_edges = np.histogram(runs, bins=bins_linear, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Filtra i bin vuoti per evitare log(0) nel plot
    mask = counts > 0
    bin_centers = bin_centers[mask]
    counts = counts[mask]

    # 3. Curva di Fit Lineare
    # La percentuale di dati nella coda serve a scalare la PDF teorica
    # affinché si integri correttamente nell'istogramma totale
    perc_coda = np.sum(runs >= xmin_auto) / len(runs)

    # IMPORTANTE: Usa linspace per x_fit se vuoi campionamento lineare
    x_fit = np.linspace(xmin_auto, max(runs), 500)
    y_fit = fit.truncated_power_law.pdf(x_fit) * perc_coda

    # 4. Plot
    ax2.loglog(bin_centers, counts, 'ob', markersize=4, alpha=0.4, label='Dati (Lin-bins)')
    ax2.loglog(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit Troncato ($\mu$={mu_auto:.2f})')
    ax2.axvline(x=xmin_auto, color='k', linestyle='--', alpha=0.3, label=f'xmin={xmin_auto:.1f}')

    ax2.set_title("Distribuzione Integrale (Testa + Coda)")
    ax2.set_xlabel("Lunghezza Run")
    ax2.set_ylabel("Densità di Probabilità")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('images/series/dual_plot.png')
    plt.close()

    # Return slopes for statistics (gamma = -slope)
    return -popt_acf_t[0], -popt_run_h[0], mu_auto

# =============================================================================
# MAIN
# =============================================================================

def avg_exponents(gamma_in, ITERATIONS = 10):
    results = [] # Store results as tuples
     
    for i in range(ITERATIONS):
        res = perform_analysis(gamma_in)
        results.append(res)
    
    results = np.array(results)
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)

    return means, stds

if __name__ == "__main__":
    x = np.linspace(0.1, 0.6, 6)

    gamma = []
    gamma_err = []
    mu = []
    mu_err = []

    for gamma_th in x:
        means, stds = avg_exponents(gamma_th)

        gamma.append(means[0])
        gamma_err.append(stds[0])

        mu.append(means[2])
        mu_err.append(stds[2])

    mu = np.array(mu)
    mu_err = np.array(mu_err)
    gamma = np.array(gamma)
    gamma_err = np.array(gamma_err)

    print(mu, mu_err, gamma, gamma_err)

    popt, pcov = curve_fit(linear_func, mu, gamma, sigma=gamma_err, absolute_sigma=True)

    err = np.sqrt(mu_err**2 + (popt[0] * gamma_err)**2)

    popt, pcov = curve_fit(linear_func, mu, gamma, sigma=err, absolute_sigma=True)

    print(popt)

    x = np.linspace(min(mu), max(mu), 2)

    fig = plt.plot()
    plt.errorbar(mu, gamma, yerr=gamma_err, xerr=mu_err, fmt='o', capsize=0, markersize=6, markeredgecolor='white')
    plt.plot(x, linear_func(x, *popt))
    plt.savefig('images\\series\\gamma_mu_graph.png')
    plt.grid()
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\gamma$')
    plt.show()