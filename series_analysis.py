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
    """
    Standard linear function y = slope * x + intercept.
    Used for fitting in log-log space, where power-law relationships
    F ~ x^alpha become straight lines: log F = alpha * log x + const.
    """
    return intercept + slope * x


# =============================================================================
# CORE LOGIC
# =============================================================================

def compute_runs(sign_series):
    """
    Computes the lengths of consecutive runs of identical symbols in a binary series.

    A "run" is a maximal sequence of consecutive identical values (e.g. [+1,+1,+1]
    is a run of length 3). Run-length statistics carry information about the
    correlation structure of the series: long-range correlated signals tend to
    produce heavy-tailed run-length distributions.

    Parameters
    ----------
    sign_series : array of {+1, -1} — the binary input series

    Returns
    -------
    run_lengths : array of int — length of each consecutive run
    """
    # Detect positions where the value changes (True = a change occurred)
    changes = np.diff(sign_series) != 0

    # Indices of change points (i.e. where each run ends)
    change_indices = np.where(changes)[0]

    # Prepend -1 and append the last index so that np.diff gives the full run lengths,
    # including the first and last run
    run_lengths = np.diff(np.concatenate([[-1], change_indices, [len(sign_series) - 1]]))

    return run_lengths


# =============================================================================
# DUAL ANALYSIS (ACF & RUNS)
# =============================================================================

def perform_analysis(gamma_target, n_total=10000000, lim=1.0):
    """
    Generates a long-memory binary series with target exponent gamma_target,
    then performs two complementary analyses:
      1. ACF analysis — fits a power law to the autocorrelation function tail
         to recover the correlation exponent gamma.
      2. Run-length CCDF analysis — fits a power law to the complementary
         cumulative distribution of run lengths to recover the run exponent mu.

    Both analyses produce a dual plot and return the fitted exponents.

    Parameters
    ----------
    gamma_target : float — target ACF decay exponent (C(tau) ~ tau^-gamma)
    n_total      : int   — length of the series to generate (default: 10 million)
    lim          : float — lower cutoff for the CCDF power-law fit (default: 1.0)

    Returns
    -------
    gamma_fit : float — ACF exponent estimated from the tail fit
    mu_ccdf   : float — run-length exponent estimated from the CCDF fit
    """

    # --- 1. Series generation ---
    # Generate a continuous long-memory series via FFM, then binarize by sign
    series = generate_ffm(n_total, gamma_target)
    sign_series = np.where(series >= 0, 1, -1)

    # (The DFA block below is commented out but left for reference.
    #  It was an alternative way to estimate gamma via Detrended Fluctuation Analysis.)
    """
    print('fit dfa')
    scales, F_n = bin_gen.compute_dfa(sign_series, min_scale=10, max_scale=10000, num_scales=40)
    log_scales = np.log(scales)
    log_Fn = np.log(F_n)
    alpha, intercept = np.polyfit(log_scales, log_Fn, 1)
    gamma_estimated = 2.0 - 2.0 * alpha
    print(gamma_estimated)
    """

    # --- 2. Data preparation ---

    # Compute ACF up to lag 1000 using FFT for efficiency
    acf_values = acf(sign_series, nlags=1000, fft=True)

    # Compute run lengths of the binary series
    runs = compute_runs(sign_series)

    # (Histogram bins — computed but not directly used in the final plot below)
    bins = np.linspace(1.0, 1000.0, 1000)
    hist, _ = np.histogram(runs, bins=bins)

    # Lag axis for the ACF (starts at lag 1 to align with the power-law fit)
    x_axis = np.arange(1, len(acf_values) + 1)

    # --- 3. ACF power-law fit ---
    # Fit only the tail of the ACF (lags >= 10) where the power-law behavior dominates.
    # We also restrict to positive ACF values to avoid log of non-positive numbers.
    mask_acf_tail = (x_axis >= 10) & (acf_values > 0)

    # Fit log(ACF) = slope * log(lag) + intercept in log-log space
    popt_acf_t, _ = curve_fit(
        linear_func,
        np.log(x_axis[mask_acf_tail]),
        np.log(acf_values[mask_acf_tail])
    )
    # The ACF exponent gamma is the negative of the slope (C(tau) ~ tau^-gamma)
    gamma_fit = -popt_acf_t[0]

    # --- 4. Run-length CCDF construction ---
    # We estimate the Complementary CDF (CCDF) empirically: P(X > x).
    # For a power-law distribution P(x) ~ x^(-mu), the CCDF scales as
    # P(X > x) ~ x^(-(mu-1)), so the slope of log CCDF vs log x gives -(mu-1).

    # Sort the run lengths in ascending order
    sorted_runs = np.sort(runs)
    n_runs = len(sorted_runs)

    # Empirical survival probabilities: the i-th smallest value has P(X > x) = 1 - i/n
    p_all = 1.0 - np.arange(n_runs) / n_runs

    # Keep only unique run-length values to avoid duplicates in the fit
    valori_unici, indici = np.unique(sorted_runs, return_index=True)

    # For each unique value, find its corresponding survival probability
    indices = np.searchsorted(sorted_runs, valori_unici, side='right') - 1
    x_ccdf_clean = valori_unici
    y_ccdf_clean = p_all[indices]

    # --- 5. CCDF power-law fit ---
    # Apply the lower cutoff lim and restrict to positive CCDF values
    mask_ccdf = (x_ccdf_clean >= lim) & (y_ccdf_clean > 0)
    x_fit_log = np.log(x_ccdf_clean[mask_ccdf])
    y_fit_log = np.log(y_ccdf_clean[mask_ccdf])

    popt_ccdf, _ = curve_fit(linear_func, x_fit_log, y_fit_log)
    slope_ccdf = popt_ccdf[0]

    # Recover the PDF exponent mu from the CCDF slope: slope = -(mu - 1) => mu = 1 - slope
    mu_ccdf = -slope_ccdf + 1

    # --- 6. Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left panel: ACF on a log-log scale with power-law fit
    ax1.loglog(x_axis, acf_values, 'o', markersize=3, alpha=0.4, label='ACF Data')
    x_fit = np.geomspace(1, 1000, 100)
    ax1.plot(
        x_fit,
        np.exp(popt_acf_t[1]) * (x_fit ** popt_acf_t[0]),
        'r-',
        label=f'Tail $\\gamma$: {-popt_acf_t[0]:.2f}'
    )
    ax1.set_title(f"ACF Analysis (Target $\\gamma$={gamma_target:.2f})")
    ax1.grid()
    ax1.legend()

    # Right panel: empirical CCDF of run lengths with power-law fit
    ax2.loglog(x_ccdf_clean, y_ccdf_clean, markersize=3, alpha=1.0, label='CCDF Data')
    x_plot = np.geomspace(lim, max(runs), 100)
    y_plot = np.exp(popt_ccdf[1]) * (x_plot ** slope_ccdf)
    ax2.plot(x_plot, y_plot, 'r-', linewidth=2, label=f'CCDF Fit ($\\mu$={mu_ccdf:.2f})')
    ax2.set_xlabel("Run Length")
    ax2.set_ylabel("P(X > x)")
    ax2.set_title("Run Length Distribution (CCDF)")
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'images/series/dual_plot_{gamma_target:.2f}.png')
    plt.close()

    return gamma_fit, mu_ccdf


# =============================================================================
# MAIN — sweep over gamma values and plot the gamma vs mu relationship
# =============================================================================
if __name__ == "__main__":
    # Target gamma values to sweep: six values from 0.1 to 0.6
    gamma_th = np.linspace(0.1, 0.6, 6)

    # Lower cutoff for the CCDF fit: larger gamma means shorter runs,
    # so the power-law tail starts earlier (smaller lim needed)
    xlim = [300.0, 100.0, 50.0, 40.0, 30.0, 20.0]

    gamma_fit = []
    mu_fit = []

    for i in range(6):
        gamma = gamma_th[i]
        lim   = xlim[i]
        print(f'Processing gamma = {gamma:.2f}')
        g_fit, m_fit = perform_analysis(gamma, lim=lim)
        gamma_fit.append(g_fit)
        mu_fit.append(m_fit)

    mu_fit    = np.array(mu_fit)
    gamma_fit = np.array(gamma_fit)

    print("Estimated mu values:    ", mu_fit)
    print("Estimated gamma values: ", gamma_fit)

    fig = plt.figure()

    # Fit a linear relationship between estimated mu and estimated gamma
    popt, pcov = curve_fit(linear_func, mu_fit, gamma_fit)
    print("Linear fit (mu -> gamma_fit):", popt)

    x = np.linspace(min(mu_fit), max(mu_fit), 2)

    # Plot estimated gamma vs estimated mu (circles)
    plt.plot(mu_fit, gamma_fit, marker='o', linestyle='', markeredgecolor='white',
             label='Estimated gamma')
    plt.plot(x, linear_func(x, *popt), label='Linear fit (estimated)')

    # Fit a linear relationship between estimated mu and theoretical gamma
    popt, pcov = curve_fit(linear_func, mu_fit, gamma_th)
    print("Linear fit (mu -> gamma_th):", popt)

    # Plot theoretical gamma vs estimated mu (crosses) — for comparison
    plt.plot(mu_fit, gamma_th, marker='x', linestyle='', label='Theoretical gamma')
    plt.plot(x, linear_func(x, *popt), linestyle='--', label='Linear fit (theoretical)')

    plt.grid()
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\gamma$')
    plt.legend()
    plt.savefig('images\\series\\gamma_mu_graph.png')
    plt.show()