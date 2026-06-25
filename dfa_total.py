import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ══════════════════════════════════════════════════════════════════════════════
# BOUNDED GLOBAL CONCATENATED DFA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def estimate_global_hurst_bounded(concatenated_series, lower_bound=10, upper_bound=1000):
    """
    Computes the Hurst exponent (H) over a specified linear scaling regime 
    to avoid microstructure artifacts and finite-size saturation effects.
    """
    N = len(concatenated_series)
    Y = np.cumsum(concatenated_series - np.mean(concatenated_series))
    
    n_vals = np.unique(np.logspace(np.log10(10), np.log10(N // 4), num=60, dtype=int))
    F_n = []
    
    print(f"[INFO] Computing DFA fluctuations across {len(n_vals)} scales...")
    
    for n in n_vals:
        num_segments = N // n
        if num_segments == 0:
            continue
            
        squared_fluct = 0.0
        t = np.arange(n)
        
        for m in range(num_segments):
            start, end = m * n, m * n + n
            poly = np.polyfit(t, Y[start:end], 1)
            squared_fluct += np.sum((Y[start:end] - np.polyval(poly, t)) ** 2)
            
            start_b, end_b = N - (m + 1) * n, N - (m + 1) * n + n
            poly_b = np.polyfit(t, Y[start_b:end_b], 1)
            squared_fluct += np.sum((Y[start_b:end_b] - np.polyval(poly_b, t)) ** 2)
            
        F_n.append(np.sqrt(squared_fluct / (2 * num_segments * n)))
        
    n_vals = np.array(n_vals)
    F_n = np.array(F_n)
    
    fit_mask = (n_vals >= lower_bound) & (n_vals <= upper_bound)
    
    if np.sum(fit_mask) >= 3:
        H, intercept, r_value, p_value, std_err = linregress(
            np.log10(n_vals[fit_mask]), 
            np.log10(F_n[fit_mask])
        )
        r_sq = r_value**2
    else:
        raise ValueError("Not enough scaling points found within the specified fitting bounds.")
    
    return H, std_err, r_sq, n_vals, F_n, intercept

# ══════════════════════════════════════════════════════════════════════════════
# EFFICIENT GLOBAL ACF COMPUTATION & REGRESSION FIT
# ══════════════════════════════════════════════════════════════════════════════

def compute_global_acf(series, max_lag=2000):
    """
    Computes the autocorrelation function for a long series up to max_lag.
    Uses efficient numpy operations.
    """
    print(f"[INFO] Computing Global ACF up to lag {max_lag}...")
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    
    if var == 0:
        return np.ones(max_lag + 1)
        
    normalized_series = series - mean
    
    acf_vals = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            covariance = np.mean(normalized_series[:-lag] * normalized_series[lag:])
            acf_vals.append(covariance / var)
            
    return np.array(acf_vals)


def fit_acf_power_law(lags, acf_values, lag_min=20):
    """
    Fits a power-law decay C(tau) ~ tau^(-gamma) to the ACF for lags > lag_min.
    Filters out any negative ACF values that can appear due to statistical noise.
    """
    # Select lags strictly greater than the threshold
    mask = (lags > lag_min) & (acf_values > 0)
    
    if np.sum(mask) < 3:
        print(f"[WARNING] Not enough positive ACF points for lags > {lag_min} to fit power law.")
        return np.nan, np.nan, np.nan
        
    log_lags = np.log10(lags[mask])
    log_acf = np.log10(acf_values[mask])
    
    # Linear fit on log-log coordinates: log(C) = -gamma * log(tau) + intercept
    slope, intercept, r_value, p_value, std_err = linregress(log_lags, log_acf)
    
    # The decay exponent gamma is defined as the negative of the slope: C(tau) ~ tau^-gamma
    gamma_measured = -slope
    r_sq = r_value ** 2
    
    return gamma_measured, std_err, r_sq, intercept

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    data_dir = os.path.join('database', 'data')
    paths = sorted(os.listdir(data_dir))
    
    all_signs = []
    total_days = 0
    
    print("\n" + "═"*60)
    print("              LOADING & CLEANING DATA              ")
    print("═"*60)
    
    for path in paths:
        file_path = os.path.join(data_dir, path)
        trades = pd.read_csv(file_path, header=None)
        signs = trades[3].values.astype(float)
        
        if len(signs) < 100:
            continue
            
        # Normalize each day individually to strip out overnight shifts
        normalized_signs = signs - np.mean(signs)
        all_signs.append(normalized_signs)
        total_days += 1
        
    global_sequence = np.concatenate(all_signs)
    total_length = len(global_sequence)
    
    print(f"Successfully concatenated {total_days} trading days.")
    print(f"Total sequence length (N): {total_length} order signs.")
    print("─"*60)
    
    # 1. Run Bounded DFA Analysis
    DFA_L_BOUND, DFA_U_BOUND = 10, 1000
    H, h_err, dfa_r_sq, n_v, fn_v, dfa_intercept = estimate_global_hurst_bounded(
        global_sequence, lower_bound=DFA_L_BOUND, upper_bound=DFA_U_BOUND
    )
    
    # Calculate empirical gamma according to the paper equation: γ = 2(1 - H)
    gamma_from_dfa = 2 * (1 - H)
    gamma_dfa_err = 2 * h_err
    
    # 2. Run Global ACF Analysis
    MAX_LAG = 1000
    acf_values = compute_global_acf(global_sequence, max_lag=MAX_LAG)
    lags = np.arange(MAX_LAG + 1)
    
    # 3. Fit Power Law to ACF for Lags > 20
    ACF_MIN_LAG = 20
    gamma_from_acf, acf_err, acf_r_sq, acf_intercept = fit_acf_power_law(
        lags, acf_values, lag_min=ACF_MIN_LAG
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # CONSOLIDATED COMPARISON REPORT
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("         CONSOLIDATED METHOD COMPARISON REPORT      ")
    print("═"*60)
    print(f"METHOD 1: DFA INDIRECT ESTIMATION")
    print(f"  > Bounded Fit Regime:          n ∈ [{DFA_L_BOUND}, {DFA_U_BOUND}]")
    print(f"  > Global Hurst Exponent (H):   {H:.4f} ± {h_err:.4f}")
    print(f"  > Implied Gamma (2(1-H)):      {gamma_from_dfa:.4f} ± {gamma_dfa_err:.4f}")
    print(f"  > Goodness of Fit (R²):        {dfa_r_sq:.5f}")
    print("─"*60)
    print(f"METHOD 2: ACF DIRECT POWER-LAW FIT")
    print(f"  > Target Estimation Windows:   lags > {ACF_MIN_LAG}")
    if not np.isnan(gamma_from_acf):
        print(f"  > Measured Gamma (γ):          {gamma_from_acf:.4f} ± {acf_err:.4f}")
        print(f"  > Goodness of Fit (R²):        {acf_r_sq:.5f}")
        print("─"*60)
        discrepancy = abs(gamma_from_dfa - gamma_from_acf)
        print(f"Absolute Discrepancy |γ_dfa - γ_acf|: {discrepancy:.4f}")
    else:
        print("  > Measured Gamma (γ):          Fitting Failed (No clear positive power-law)")
    print("═"*60 + "\n")
    
    # ══════════════════════════════════════════════════════════════════════════
    # GENERATE DUAL ANALYSIS PLOTS WITH THE LAG > 20 FIT
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ─── LEFT SUBPLOT: DFA BOUNDED FIT ───
    ax_dfa = axes[0]
    ax_dfa.loglog(n_v, fn_v, 'ko', markersize=4, label='Empirical $F(n)$')
    
    n_fit = n_v[(n_v >= DFA_L_BOUND) & (n_v <= DFA_U_BOUND)]
    dfa_fit_line = (10**dfa_intercept) * (n_fit**H)
    ax_dfa.loglog(n_fit, dfa_fit_line, 'r-', linewidth=3, 
                  label=f'Bounded Fit (H = {H:.3f}, $R^2$ = {dfa_r_sq:.4f})')
    
    ax_dfa.axvline(DFA_L_BOUND, color='darkred', linestyle='--', alpha=0.7)
    ax_dfa.axvline(DFA_U_BOUND, color='darkred', linestyle='--', alpha=0.7)
    ax_dfa.axvspan(DFA_L_BOUND, DFA_U_BOUND, color='green', alpha=0.07, label='True Scaling Zone')
    ax_dfa.axvspan(DFA_U_BOUND, n_v[-1], color='red', alpha=0.05, label='Artifact Zone')
    
    ax_dfa.set_xlabel('Scale window $n$ (Log Scale)', fontsize=12)
    ax_dfa.set_ylabel('Fluctuation $F(n)$ (Log Scale)', fontsize=12)
    ax_dfa.set_title('Bounded Global DFA')
    ax_dfa.legend(loc='upper left')
    ax_dfa.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # ─── RIGHT SUBPLOT: LOG-LOG ACF DECAY WITH DIRECT COUPLING FIT ───
    ax_acf = axes[1]
    positive_lags = lags[1:]
    positive_acf = acf_values[1:]
    valid_mask = positive_acf > 0
    
    # Plot empirical points
    ax_acf.loglog(positive_lags[valid_mask], positive_acf[valid_mask], 'C0o', markersize=4, alpha=0.6, label='Empirical ACF')
    
    # Plot direct regression fit line for lags > 20
    if not np.isnan(gamma_from_acf):
        lags_fit = positive_lags[(positive_lags > ACF_MIN_LAG) & (positive_acf > 0)]
        # Reconstruct line using fitted parameters: C(tau) = 10^(intercept) * tau^(-gamma)
        acf_fit_line = (10**acf_intercept) * (lags_fit ** (-gamma_from_acf))
        ax_acf.loglog(lags_fit, acf_fit_line, 'm-', linewidth=3,
                      label=f'Direct Fit lags > {ACF_MIN_LAG} ($\\gamma$ = {gamma_from_acf:.3f}, $R^2$ = {acf_r_sq:.3f})')
    
    # Plot the implied theoretical projection mapping directly from DFA 
    if len(positive_lags[valid_mask]) > 10:
        base_idx = 10
        theoretical_decay = (positive_lags ** (-gamma_from_dfa)) * (positive_acf[base_idx] / (positive_lags[base_idx] ** (-gamma_from_dfa)))
        ax_acf.loglog(positive_lags, theoretical_decay, 'k--', linewidth=1.8, 
                      label=f'Implied DFA Projection ($\\gamma_{{dfa}}$ = {gamma_from_dfa:.3f})')
    
    # Highlight the lag = 20 threshold boundary
    ax_acf.axvline(ACF_MIN_LAG, color='purple', linestyle=':', linewidth=2, label=f'Fit Threshold ($\\tau$ = {ACF_MIN_LAG})')
    
    ax_acf.set_xlabel('Lag $\\tau$ (Log Scale)', fontsize=12)
    ax_acf.set_ylabel('Autocorrelation $C(\\tau)$ (Log Scale)', fontsize=12)
    ax_acf.set_title('Global ACF Decay Fit')
    ax_acf.legend(loc='lower left')
    ax_acf.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # Save output image
    output_dir = os.path.join('images', 'acf')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'global_dfa_and_acf_bounded_fit.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"[INFO] Analysis complete. Dual regression plots saved to:\n       {plot_path}\n")