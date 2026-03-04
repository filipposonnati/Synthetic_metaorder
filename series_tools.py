import numpy as np
import matplotlib.pyplot as plt
from series_creation import generate_ffm

# =============================================================================
# DETRENDED FLUCTUATION ANALYSIS (DFA)
# =============================================================================
def compute_dfa(series, min_scale=10, max_scale=None, num_scales=30):
    """
    Computes DFA of order 1 (linear detrending) on the input time series.
    
    DFA measures long-range correlations by analyzing how the root-mean-square
    fluctuation F(n) scales with the window size n. If the series has power-law
    correlations, F(n) ~ n^alpha, where alpha is the DFA scaling exponent.

    Parameters
    ----------
    series     : array-like, the input time series
    min_scale  : int, smallest window size to test (default: 10)
    max_scale  : int, largest window size to test (default: N // 10)
    num_scales : int, number of window sizes to test, log-spaced (default: 30)

    Returns
    -------
    scales : array of int   — the window sizes n that were tested
    F_n    : array of float — the DFA fluctuation F(n) for each scale
    """
    N = len(series)

    # Default max scale: a common rule of thumb is to not exceed 1/10th of the
    # total series length, ensuring enough windows exist for a reliable estimate
    if max_scale is None:
        max_scale = N // 10
        
    # --- Step 1: Integration ---
    # Compute the cumulative sum of the mean-centered series.
    # This converts the signal into a "profile" (analogous to a random walk),
    # which amplifies the correlation structure into large-scale geometric features.
    y = np.cumsum(series - np.mean(series))
    
    # --- Step 2: Build the scale ladder ---
    # Generate window sizes spaced evenly on a log scale (geomspace), so that
    # they appear evenly spaced on the log-log plot used to estimate alpha.
    # np.unique removes duplicates that arise after rounding to integers.
    scales = np.unique(np.geomspace(min_scale, max_scale, num_scales).astype(int))

    # Array to store the fluctuation value F(n) for each scale
    F_n = np.zeros(len(scales))
    
    for i, n in enumerate(scales):

        # --- Step 3: Segmentation ---
        # Divide the profile into non-overlapping windows of length n.
        # Any leftover points at the tail (N mod n) are discarded.
        n_windows = N // n
        windowed_y = y[:n_windows * n].reshape((n_windows, n))
        
        # Local x-axis used for polynomial fitting inside each window
        x = np.arange(n)
        
        rms_sum = 0.0
        for w in range(n_windows):

            # --- Step 4: Local detrending (the key DFA step) ---
            # Fit a degree-1 polynomial (straight line) to the profile within
            # this window. This is the "DFA order 1" detrending.
            # The fitted line captures the local trend (slow drift) in this segment.
            p = np.polyfit(x, windowed_y[w], 1)
            trend = np.polyval(p, x)
            
            # Subtract the local trend and accumulate the mean squared residual.
            # The residual represents genuine fluctuation at scale n,
            # stripped of any local polynomial trend.
            rms_sum += np.mean((windowed_y[w] - trend)**2)
            
        # --- Step 5: Fluctuation function F(n) ---
        # Average the residual variance across all windows, then take the square root.
        # F(n) is the typical (RMS) fluctuation of the detrended profile at scale n.
        F_n[i] = np.sqrt(rms_sum / n_windows)
        
    return scales, F_n

# =============================================================================
# TEST AND VISUALIZATION
# =============================================================================
if __name__ == "__main__":
    n_points = 1000000
    gamma_target = 0.3  # Target correlation exponent: C(t) ~ t^-0.3
    
    print(f"Generating binary series (n={n_points}, gamma={gamma_target})...")
    series = generate_ffm(n_points, gamma_target)
    
    print("Computing DFA...")
    scales, F_n = compute_dfa(series, min_scale=10, max_scale=10000, num_scales=40)
    
    # --- Estimate alpha from the log-log slope ---
    # If F(n) ~ n^alpha, then log F(n) = alpha * log(n) + const.
    # A linear fit in log-log space directly gives the scaling exponent alpha.
    log_scales = np.log(scales)
    log_Fn = np.log(F_n)
    alpha, intercept = np.polyfit(log_scales, log_Fn, 1)
    
    # --- Convert alpha to gamma ---
    # The DFA exponent alpha and the autocorrelation exponent gamma are related by:
    #   gamma = 2 - 2 * alpha
    # This lets us recover the decay rate of C(t) ~ t^-gamma from the DFA result.
    gamma_estimated = 2.0 - 2.0 * alpha
    
    # --- Plot ---
    plt.figure(figsize=(8, 6))

    # Raw DFA fluctuation values on a log-log scale
    plt.loglog(scales, F_n, 'bo', markersize=5, alpha=0.7, label='DFA data')
    
    # Overlay the fitted power law line F(n) = exp(intercept) * n^alpha
    fit_line = np.exp(intercept) * (scales ** alpha)
    plt.loglog(scales, fit_line, 'r-', linewidth=2,
               label=f'Linear fit ($\\alpha$={alpha:.3f})')
    
    plt.title(f"DFA Analysis — Target $\\gamma$={gamma_target}, Estimated $\\gamma$={gamma_estimated:.3f}")
    plt.xlabel("Scale $n$ (window size)")
    plt.ylabel("Detrended Fluctuation $F(n)$")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()