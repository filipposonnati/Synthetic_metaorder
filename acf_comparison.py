import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from pathlib import Path
import os
import sys
from os import listdir
import pandas as pd
from acf import pooled_acf as pooled_acf_function

# ── local imports ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from lmf import simulate_lmf, simulate_lmf_lambda

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
MAX_LAG       = 5_000      
TOTAL_STEPS   = 10_000_000  

LMF_ALPHA     = 1.5

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def log_bin_acf(acf_vals, max_lag, num_bins=30):
    """
    Raggruppa l'ACF in bin logaritmici per ridurre il rumore ad alti lag.
    Ritorna i centri geometrici dei bin (lags) e la media dell'ACF in quel bin.
    """
    # Consideriamo solo i lag da 1 a max_lag
    y = acf_vals[1:max_lag + 1]
    x = np.arange(1, max_lag + 1)
    
    # Genera i bordi dei bin spaziati logaritmicamente da 1 a max_lag
    bin_edges = np.logspace(np.log10(1), np.log10(max_lag), num_bins + 1)
    
    binned_x = []
    binned_y = []
    
    for i in range(num_bins):
        # Maschera per trovare i lag che ricadono nel bin corrente
        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        if i == num_bins - 1:  # Includi l'estremo destro nell'ultimo bin
            mask = mask | (x == bin_edges[i+1])
            
        if np.any(mask):
            # Centro geometrico del bin per l'asse X
            binned_x.append(np.sqrt(bin_edges[i] * bin_edges[i+1]))
            # Media aritmetica dei valori ACF nel bin
            binned_y.append(np.mean(y[mask]))
            
    return np.array(binned_x), np.array(binned_y)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── 1. Load empirical ACF ─────────────────────────────────────────────────
    acf_binary_path = Path('database/acf_binary.npy')
    p_plus_path     = Path('database/p_plus.npy')

    if not acf_binary_path.is_file():
        raise FileNotFoundError("database/acf_binary.npy not found.")

    pooled_acf = np.load(acf_binary_path)
    p_plus     = float(np.load(p_plus_path)) if p_plus_path.is_file() else 0.5
    max_lag    = min(MAX_LAG, len(pooled_acf) - 1)
    lags       = np.arange(1, max_lag + 1)

    print(f"Loaded empirical ACF  (max_lag={max_lag},  p_plus={p_plus:.4f})")



    data_ar_dir = os.path.join('database', 'data_ar_1000')
    paths    = listdir(data_ar_dir)
    max_lag  = 5_000

    all_signs       = []
    all_daily_corrs = []

    for path in paths:
        trades = pd.read_csv(os.path.join(data_ar_dir, path), header=None)
        signs  = trades[3].values.astype(float)
        all_signs.append(signs)
        all_daily_corrs.append(acf(signs, nlags=max_lag, fft=True))

    cache_path = Path("database/acf_ar.npy")
    if cache_path.is_file():
        pooled_ar = np.load(cache_path)
        print("Loaded cached pooled ACF ar.")
    else:
        print("Computing pooled ACF …")
        pooled_ar = pooled_acf_function(all_signs, nlags=max_lag)
        np.save(cache_path, pooled_ar)
        print("Saved pooled ACF to cache.")

    theory  =  0.0550 * lags ** (-0.5)

    # ── 6. Applica Log-Binning ai dati simulati ed empirici ───────────────────
    NUM_BINS = 50

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Usiamo 'marker' (punti/quadrati) per i dati binnati, così si capisce che è una media
    # Empirical (real data)
    x_emp, y_emp     = log_bin_acf(pooled_acf, max_lag, num_bins=NUM_BINS)
    ax.loglog(x_emp, y_emp, marker='o', linestyle='-', color='black', lw=1.5, label='Empirical pooled ACF')


    # ── 3. Fixed-N LMF ───────────────────────────────────────────────────────
    print(f"Simulating fixed-N LMF ...")
    LMF_N_TRADERS = 100
    flow_fixed = simulate_lmf(LMF_ALPHA, LMF_N_TRADERS, TOTAL_STEPS)
    acf_fixed  = acf(flow_fixed, nlags=max_lag, fft=True)

    x_fix, y_fix     = log_bin_acf(acf_fixed, max_lag, num_bins=NUM_BINS)

    ax.loglog(x_fix, y_fix, marker='^', linestyle='-', color='tomato', lw=1.2, alpha=0.8,
              label=rf'LMF fixed-N  ($\alpha={LMF_ALPHA}$, $N={LMF_N_TRADERS}$)')



    LMF_N_TRADERS = 50
    flow_fixed = simulate_lmf(LMF_ALPHA, LMF_N_TRADERS, TOTAL_STEPS)
    acf_fixed  = acf(flow_fixed, nlags=max_lag, fft=True)

    x_fix, y_fix     = log_bin_acf(acf_fixed, max_lag, num_bins=NUM_BINS)

    ax.loglog(x_fix, y_fix, marker='^', linestyle='-', color='red', lw=1.2, alpha=0.8,
              label=rf'LMF fixed-N  ($\alpha={LMF_ALPHA}$, $N={LMF_N_TRADERS}$)')



    # ── 4. λ-model LMF ───────────────────────────────────────────────────────
    print(f"Simulating λ-model LMF ...")
    LMF_LAMBDA = 0.3
    lam_c = (LMF_LAMBDA - 1) / LMF_LAMBDA
    flow_lambda, _, _ = simulate_lmf_lambda(LMF_ALPHA, LMF_LAMBDA, TOTAL_STEPS)
    acf_lambda     = acf(flow_lambda, nlags=max_lag, fft=True)

    x_lam, y_lam = log_bin_acf(acf_lambda, max_lag, num_bins=NUM_BINS)

    ax.loglog(x_lam, np.abs(y_lam), marker='d', linestyle='-', color='seagreen', lw=1.2, alpha=0.8,
              label=rf'LMF $\lambda$-model  ($\alpha={LMF_LAMBDA}$, $\lambda={LMF_LAMBDA}$)')



    LMF_LAMBDA = 0.3
    p_random = 0.22
    lam_c = (LMF_LAMBDA - 1) / LMF_LAMBDA
    flow_lambda, _, _ = simulate_lmf_lambda(LMF_ALPHA, LMF_LAMBDA, TOTAL_STEPS, p_trade_random=p_random)
    acf_lambda     = acf(flow_lambda, nlags=max_lag, fft=True)

    x_lam, y_lam = log_bin_acf(acf_lambda, max_lag, num_bins=NUM_BINS)

    ax.loglog(x_lam, np.abs(y_lam), marker='d', linestyle='-', color='green', lw=1.2, alpha=0.8,
              label=rf'LMF $\lambda$-model  ($\alpha={LMF_LAMBDA}$, $\lambda={LMF_LAMBDA}$, $p_{{RANDOM}} = {p_random}$)')



    LMF_LAMBDA = 0.2
    lam_c = (LMF_LAMBDA - 1) / LMF_LAMBDA
    flow_lambda, _, _ = simulate_lmf_lambda(LMF_ALPHA, LMF_LAMBDA, TOTAL_STEPS, p_trade_random=p_random)
    acf_lambda     = acf(flow_lambda, nlags=max_lag, fft=True)

    x_lam, y_lam = log_bin_acf(acf_lambda, max_lag, num_bins=NUM_BINS)

    ax.loglog(x_lam, np.abs(y_lam), marker='d', linestyle='-', color='blue', lw=1.2, alpha=0.8,
              label=rf'LMF $\lambda$-model  ($\alpha={LMF_LAMBDA}$, $\lambda={LMF_LAMBDA}$)')


    
    # Teoria
    ax.loglog(lags, theory, color='black', lw=1.2, linestyle=':',
              label=rf'Theory $\tau^{{-0.5}}$')

    ax.set_xlabel(r'Lag $\tau$')
    ax.set_ylabel(r'ACF $C(\tau)$')
    ax.legend(loc='lower left')
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()

    os.makedirs(os.path.join('images', 'acf'), exist_ok=True)
    fig.savefig(os.path.join('images', 'acf', 'acf_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {os.path.join('images', 'acf', 'acf_comparison.png')}")
    plt.show()