import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import linregress

# ══════════════════════════════════════════════════════════════════════════════
# CORE METHODOLOGY: DETRENDED FLUCTUATION ANALYSIS (DFA)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_hurst_dfa(x):
    """
    Computes the Hurst exponent (H) using standard Detrended Fluctuation Analysis
    focused on capturing the true long-memory scaling behavior.
    """
    N = len(x)
    # 1. Integrate the time series (profile)
    Y = np.cumsum(x - np.mean(x))
    
    # 2. Set up scale windows (avoiding very small scales that distort the asymptote)
    n_vals = np.unique(np.logspace(np.log10(10), np.log10(N // 4), num=30, dtype=int))
    F_n = []
    
    for n in n_vals:
        num_segments = N // n
        if num_segments == 0:
            continue
            
        squared_fluct = 0.0
        t = np.arange(n)
        
        for m in range(num_segments):
            # Forward mapping
            start, end = m * n, m * n + n
            poly = np.polyfit(t, Y[start:end], 1)
            squared_fluct += np.sum((Y[start:end] - np.polyval(poly, t)) ** 2)
            
            # Backward mapping
            start_b, end_b = N - (m + 1) * n, N - (m + 1) * n + n
            poly_b = np.polyfit(t, Y[start_b:end_b], 1)
            squared_fluct += np.sum((Y[start_b:end_b] - np.polyval(poly_b, t)) ** 2)
            
        F_n.append(np.sqrt(squared_fluct / (2 * num_segments * n)))
        
    n_vals = np.array(n_vals)
    F_n = np.array(F_n)
    
    # 3. Calculate Hurst Exponent (H) via global linear fit on log-log scale
    H, intercept, _, _, _ = linregress(np.log10(n_vals), np.log10(F_n))
    return H, n_vals, F_n

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    data_dir = os.path.join('database', 'data')
    paths = sorted(listdir(data_dir))
    
    # Storage for across-dataset distribution analysis
    hurst_values = []
    gamma_values = []
    
    print("\n" + "═"*58)
    print(f"{'FILE':<25} | {'Hurst (H)':<12} | {'Actual γ (2-2H)':<16}")
    print("═"*58)
    
    for path in paths:
        file_path = os.path.join(data_dir, path)
        trades = pd.read_csv(file_path, header=None)
        
        # Order signs for scaling analysis (column index 3)
        signs = trades[3].values.astype(float)
        
        if len(signs) < 500:
            continue  # Long-memory checks require significant data lengths
            
        # Compute H and Empirical Gamma
        H, n_vals, F_n = estimate_hurst_dfa(signs)
        gamma_actual = 2 * (1 - H)
        
        # Collect calculated values for final distribution plotting
        hurst_values.append(H)
        gamma_values.append(gamma_actual)
        
        print(f"{path:<25} | {H:<12.4f} | {gamma_actual:<16.4f}")

    print("═"*58)

    # ══════════════════════════════════════════════════════════════════════════
    # GENERATE AND SAVE FINAL DISTRIBUTION PLOT ONLY (.png)
    # ══════════════════════════════════════════════════════════════════════════
    if len(hurst_values) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Hurst Exponent (H) Distribution
        axes[0].hist(hurst_values, bins=15, density=True, color='darkblue', alpha=0.6, edgecolor='black', label='Empirical Days')
        axes[0].axvline(0.5, color='gray', linestyle='--', linewidth=2, label='White Noise Benchmark ($H=0.5$)')
        if len(hurst_values) > 1:
            # Simple Kernel Density Estimate line overlay
            df_h = pd.Series(hurst_values)
            df_h.plot(kind='density', ax=axes[0], color='red', linewidth=2, label='KDE Trend')
        axes[0].set_xlabel('Hurst Exponent ($H$)')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'Distribution of Hurst Exponent\nMean: {np.mean(hurst_values):.3f} | Std: {np.std(hurst_values):.3f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Actual Measured Gamma Distribution
        axes[1].hist(gamma_values, bins=15, density=True, color='teal', alpha=0.6, edgecolor='black', label='Empirical Days')
        axes[1].axvline(1.0, color='gray', linestyle='--', linewidth=2, label='White Noise Benchmark ($\gamma=1.0$)')
        if len(gamma_values) > 1:
            df_g = pd.Series(gamma_values)
            df_g.plot(kind='density', ax=axes[1], color='red', linewidth=2, label='KDE Trend')
        axes[1].set_xlabel('Correlation Exponent ($\gamma$)')
        axes[1].set_ylabel('Density')
        axes[1].set_title(f'Distribution of Correlation Exponent $\gamma$\nMean: {np.mean(gamma_values):.3f} | Std: {np.std(gamma_values):.3f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Save structural image
        output_dir = os.path.join('images', 'acf')
        os.makedirs(output_dir, exist_ok=True)
        image_save_path = os.path.join(output_dir, 'dfa_final_distributions.png')
        
        plt.tight_layout()
        plt.savefig(image_save_path, dpi=300)
        plt.close()
        
        print(f"[INFO] Final distribution profile graphic generated successfully: {image_save_path}\n")
    else:
        print("[WARNING] Processing failed: No data metrics available to plot distribution configurations.\n")