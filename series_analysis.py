import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf
import powerlaw
from series_creation import generate_ffm

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def linear_func(x, slope, intercept):
    """Standard linear function for curve fitting in log-log space."""
    return intercept + slope * x

# =============================================================================
# CORE LOGIC
# =============================================================================

def compute_runs(sign_series):
    """Computes the lengths of consecutive identical symbols."""
    changes = np.diff(sign_series) != 0
    change_indices = np.where(changes)[0]
    run_lengths = np.diff(np.concatenate([[-1], change_indices, [len(sign_series) - 1]]))
    return run_lengths

# =============================================================================
# DUAL ANALYSIS (ACF & RUNS)
# =============================================================================

def perform_analysis(gamma_target, n_total=10000000, lim = 1.0):
    # 1. Generation
    #sign_series = generate_correlated_sign_series(2 * n_total, gamma_target)[:n_total]
    #sign_series = generate_arfima_fast(n_total, gamma_target)
    series = generate_ffm(n_total, gamma_target)
    sign_series = np.where(series >= 0, 1, -1)

    """
    print('fit dfa')
    scales, F_n = bin_gen.compute_dfa(sign_series, min_scale=10, max_scale=10000, num_scales=40)
    
    # Fit della DFA per trovare l'esponente alpha (F(n) ~ n^alpha)
    # Fittiamo in scala logaritmica: log(F(n)) = alpha * log(n) + c
    log_scales = np.log(scales)
    log_Fn = np.log(F_n)
    alpha, intercept = np.polyfit(log_scales, log_Fn, 1)
    
    # Conversione da alpha a gamma
    gamma_stimato = 2.0 - 2.0 * alpha

    print(gamma_stimato)
    """

    # 2. Data Preparation
    acf_values = acf(sign_series, nlags=1000, fft=True)
    runs = compute_runs(sign_series)
    bins = np.linspace(1.0, 1000.0, 1000)
    hist, _ = np.histogram(runs, bins=bins)
    
    x_axis = np.arange(1, len(acf_values) + 1)

    # --- 3. ACF DUAL FITTING ---
    mask_acf_tail = (x_axis >= 10) & (acf_values > 0)
    
    popt_acf_t, _ = curve_fit(linear_func, np.log(x_axis[mask_acf_tail]), np.log(acf_values[mask_acf_tail]))
    gamma_fit = -popt_acf_t[0]

    # --- 5. VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot ACF
    ax1.loglog(x_axis, acf_values, 'o', markersize=3, alpha=0.4, label='ACF Data')
    x_fit = np.geomspace(1, 1000, 100)
    ax1.plot(x_fit, np.exp(popt_acf_t[1]) * (x_fit**popt_acf_t[0]), 'r-', label=f'Tail $\gamma$: {-popt_acf_t[0]:.2f}')
    ax1.set_title(f"ACF Analysis (Target $\gamma$={gamma_target:.2f})")
    ax1.grid()
    ax1.legend()

    # 2. Istogramma Lineare (Mantenendo linspace come richiesto)
    # Usiamo un numero di bin fisso o proporzionale al range
    bins_linear = np.linspace(min(runs), max(runs), max(runs) - min(runs) + 1)
    counts, bin_edges = np.histogram(runs, bins=bins_linear, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Filtra i bin vuoti per evitare log(0) nel plot
    mask = counts > 0
    bin_centers = bin_centers[mask]
    counts = counts[mask]

    # 1. Ordina i run
    sorted_runs = np.sort(runs)
    n_runs = len(sorted_runs)

    # 2. Calcola la probabilità per ogni elemento
    p_all = 1.0 - np.arange(n_runs) / n_runs

    valori_unici, indici = np.unique(sorted_runs, return_index=True)
    
    indices = np.searchsorted(sorted_runs, valori_unici, side='right') - 1
    
    x_ccdf_clean = valori_unici
    y_ccdf_clean = p_all[indices]

    # --- Ora esegui il fit solo su questi punti puliti ---
    mask_ccdf = (x_ccdf_clean >= lim) & (y_ccdf_clean > 0)
    x_fit_log = np.log(x_ccdf_clean[mask_ccdf])
    y_fit_log = np.log(y_ccdf_clean[mask_ccdf])

    popt_ccdf, _ = curve_fit(linear_func, x_fit_log, y_fit_log)
    slope_ccdf = popt_ccdf[0]
    mu_ccdf = -slope_ccdf + 1  # Ricaviamo mu dallo slope

    # Plot CCDF (Dati)
    ax2.loglog(x_ccdf_clean, y_ccdf_clean, markersize=3, alpha=1.0, label='CCDF Data')
    # Plot Fit della CCDF
    x_plot = np.geomspace(lim, max(runs), 100)
    y_plot = np.exp(popt_ccdf[1]) * (x_plot**slope_ccdf)
    ax2.plot(x_plot, y_plot, 'r-', linewidth=2, label=f'CCDF Fit ($\mu$={mu_ccdf:.2f})')

    ax2.set_xlabel("Lunghezza Run")
    ax2.set_ylabel("P(X > x)")
    ax2.set_title("Run Length Distribution")
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'images/series/dual_plot_{gamma_target:.2f}.png')
    plt.close()

    return gamma_fit, mu_ccdf

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    gamma_th = np.linspace(0.1, 0.6, 6)

    xlim = [300.0, 100.0, 50.0, 40.0, 30.0, 20.0]

    gamma_fit = []
    mu_fit = []

    for i in range(6):
        gamma = gamma_th[i]
        lim = xlim[i]
        print(f'{gamma:.2f}')
        g_fit, m_fit = perform_analysis(gamma, lim=lim)

        gamma_fit.append(g_fit)
        mu_fit.append(m_fit)

    mu_fit = np.array(mu_fit)
    gamma_fit = np.array(gamma_fit)

    print(mu_fit, gamma_fit)

    fig = plt.figure()

    popt, pcov = curve_fit(linear_func, mu_fit, gamma_fit)
    print(popt)

    x = np.linspace(min(mu_fit), max(mu_fit), 2)

    plt.plot(mu_fit, gamma_fit, marker='o', linestyle='', markeredgecolor='white')
    plt.plot(x, linear_func(x, *popt))

    popt, pcov = curve_fit(linear_func, mu_fit, gamma_th)
    print(popt)

    plt.plot(mu_fit, gamma_th, marker='x', linestyle='')
    plt.plot(x, linear_func(x, *popt), linestyle='--')

    plt.grid()
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\gamma$')
    plt.savefig('images\\series\\gamma_mu_graph.png')
    plt.show()