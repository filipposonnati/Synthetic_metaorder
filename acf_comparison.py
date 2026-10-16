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
MAX_LAG       = 1_000      
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
    y = acf_vals[1:max_lag + 1]
    x = np.arange(1, max_lag + 1)
    
    bin_edges = np.logspace(np.log10(1), np.log10(max_lag), num_bins + 1)
    
    binned_x = []
    binned_y = []
    
    for i in range(num_bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        if i == num_bins - 1:
            mask = mask | (x == bin_edges[i+1])
            
        if np.any(mask):
            binned_x.append(np.sqrt(bin_edges[i] * bin_edges[i+1]))
            binned_y.append(np.mean(y[mask]))
            
    return np.array(binned_x), np.array(binned_y)


def get_or_compute_lmf_fixed_acf(alpha, n_traders, total_steps, max_lag, cache_dir='database'):
    """
    Carica l'ACF per LMF fixed-N dalla cache se esiste, altrimenti la simula e la salva.
    """
    cache_path = Path(cache_dir) / f"acf_lmf_fixed_a{alpha}_N{n_traders}.npy"
    if cache_path.is_file():
        print(f"Loaded cached LMF fixed-N ACF (N={n_traders}).")
        return np.load(cache_path)
    
    print(f"Simulating fixed-N LMF (N={n_traders}) ...")
    flow_fixed = simulate_lmf(alpha, n_traders, total_steps)
    acf_fixed = acf(flow_fixed, nlags=max_lag, fft=True)
    
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, acf_fixed)
    print(f"Saved LMF fixed-N ACF (N={n_traders}) to cache.")
    return acf_fixed


def get_or_compute_lmf_lambda_acf(alpha, lam, total_steps, max_lag, p_trade_random=0.0, cache_dir='database'):
    """
    Carica l'ACF per LMF lambda-model dalla cache se esiste, altrimenti la simula e la salva.
    """
    cache_path = Path(cache_dir) / f"acf_lmf_lambda_a{alpha}_lam{lam}_prand{p_trade_random}.npy"
    if cache_path.is_file():
        print(f"Loaded cached LMF lambda-model ACF (lambda={lam}, p_random={p_trade_random}).")
        return np.load(cache_path)
    
    print(f"Simulating λ-model LMF (lambda={lam}, p_random={p_trade_random}) ...")
    flow_lambda, _, _ = simulate_lmf_lambda(alpha, lam, total_steps, p_trade_random=p_trade_random)
    acf_lambda = acf(flow_lambda, nlags=max_lag, fft=True)
    
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, acf_lambda)
    print(f"Saved LMF lambda-model ACF (lambda={lam}, p_random={p_trade_random}) to cache.")
    return acf_lambda


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Load empirical ACF ─────────────────────────────────────────────────
    acf_binary_path = Path('database/acf_binary.npy')
    p_plus_path     = Path('database/p_plus.npy')

    if not acf_binary_path.is_file():
        raise FileNotFoundError("database/acf_binary.npy not found.")

    pooled_acf = np.load(acf_binary_path)
    p_plus     = float(np.load(p_plus_path)) if p_plus_path.is_file() else 0.5
    max_lag    = min(MAX_LAG, len(pooled_acf) - 1)
    lags       = np.arange(1, max_lag + 1)

    print(f"Loaded empirical ACF  (max_lag={max_lag},  p_plus={p_plus:.4f})")

    # ── Pooled AR ACF ──────────────────────────────────────────────────────
    cache_path = Path("database/acf_ar.npy")
    if cache_path.is_file():
        pooled_ar = np.load(cache_path)
        print("Loaded cached pooled ACF AR")
    else:
        data_ar_dir = os.path.join('database', 'data_ar_1000')
        paths    = listdir(data_ar_dir)
        all_signs = []
        for path in paths:
            trades = pd.read_csv(os.path.join(data_ar_dir, path), header=None)
            signs  = trades[3].values.astype(float)
            all_signs.append(signs)

        print("Computing pooled ACF …")
        pooled_ar = pooled_acf_function(all_signs, nlags=max_lag)
        np.save(cache_path, pooled_ar)
        print("Saved pooled ACF to cache.")

    # ── Pooled LMF ACF ──────────────────────────────────────────────────────
    cache_path = Path("database/acf_lmf_1.5_100.npy")
    if cache_path.is_file():
        pooled_lmf_100 = np.load(cache_path)
        print("Loaded cached pooled ACF LMF.")
    else:
        data_dir = os.path.join('database', 'data')
        paths    = listdir(data_dir)
        all_signs = []
        for path in paths:
            trades = pd.read_csv(os.path.join(data_dir, path), header=None)
            signs  = trades[3].values.astype(float)
            series_length = len(signs)

            signs_lmf = simulate_lmf(1.5, 100, series_length)

            all_signs.append(signs_lmf)

        print("Computing pooled ACF …")
        pooled_lmf_100 = pooled_acf_function(all_signs, nlags=max_lag)
        np.save(cache_path, pooled_lmf_100)
        print("Saved pooled ACF to cache.")

    # ── Pooled LMF Lambda ACF ──────────────────────────────────────────────────────
    cache_path = Path("database/acf_lmf_lambda_1.5_0.3.npy")
    if cache_path.is_file():
        pooled_lmf_lambda = np.load(cache_path)
        print("Loaded cached pooled ACF LMF Lambda.")
    else:
        data_dir = os.path.join('database', 'data')
        paths    = listdir(data_dir)
        all_signs = []
        for path in paths:
            trades = pd.read_csv(os.path.join(data_dir, path), header=None)
            signs  = trades[3].values.astype(float)
            series_length = len(signs)

            signs_lmf_lambda, _, _ = simulate_lmf_lambda(1.5, 0.3, series_length)

            all_signs.append(signs_lmf_lambda)

        print("Computing pooled ACF …")
        pooled_lmf_lambda = pooled_acf_function(all_signs, nlags=max_lag)
        np.save(cache_path, pooled_lmf_lambda)
        print("Saved pooled ACF to cache.")

    # ── Pooled LMF Lambda ACF ──────────────────────────────────────────────────────
    cache_path = Path("database/acf_lmf_lambda_1.5_0.3_p_rand_0.27.npy")
    if cache_path.is_file():
        pooled_lmf_lambda_rand = np.load(cache_path)
        print("Loaded cached pooled ACF LMF Lambda random.")
    else:
        data_dir = os.path.join('database', 'data')
        paths    = listdir(data_dir)
        all_signs = []
        for path in paths:
            trades = pd.read_csv(os.path.join(data_dir, path), header=None)
            signs  = trades[3].values.astype(float)
            series_length = len(signs)

            signs_lmf_lambda_rand, _, _ = simulate_lmf_lambda(1.5, 0.3, series_length, p_trade_random=0.27)

            all_signs.append(signs_lmf_lambda_rand)

        print("Computing pooled ACF …")
        pooled_lmf_lambda_rand = pooled_acf_function(all_signs, nlags=max_lag)
        np.save(cache_path, pooled_lmf_lambda_rand)
        print("Saved pooled ACF to cache.")

# ── Pooled LMF Lambda ACF ──────────────────────────────────────────────────────
    cache_path = Path("database/acf_nn.npy")
    if cache_path.is_file():
        pooled_nn = np.load(cache_path)
        print("Loaded cached pooled ACF NN.")
    else:
        data_dir = os.path.join('database', 'data_nn')
        paths    = listdir(data_dir)
        all_signs = []
        for path in paths:
            trades = pd.read_csv(os.path.join(data_dir, path), header=None)
            signs_nn  = trades[3].values.astype(float)
            all_signs.append(signs_nn)

        print("Computing pooled ACF …")
        pooled_nn = pooled_acf_function(all_signs, nlags=max_lag)
        np.save(cache_path, pooled_nn)
        print("Saved pooled ACF to cache.")

    theory = 0.0550 * lags ** (-0.5)

    NUM_BINS = 50

    # ── 3. Plot Setup ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Empirical
    x_emp, y_emp = log_bin_acf(pooled_acf, max_lag, num_bins=NUM_BINS)
    ax.loglog(x_emp, y_emp, marker='o', linestyle='-', color='black', lw=1.2, label='Empirical')

    # AR
    x_ar, y_ar = log_bin_acf(pooled_ar, max_lag, num_bins=NUM_BINS)
    ax.loglog(x_ar, y_ar, marker='x', linestyle='-', color='red', lw=1.2, label='Volumes AR')

    # LMF
    x_lmf, y_lmf = log_bin_acf(pooled_lmf_100, max_lag, num_bins=NUM_BINS)
    ax.loglog(x_lmf, y_lmf, marker='+', linestyle='-', color='blue', lw=1.2, label=r'LMF $\alpha = 1.5$ $N = 100$')

    # LMF Lambda
    x_lmf_lambda, y_lmf_lambda = log_bin_acf(pooled_lmf_lambda, max_lag, num_bins=NUM_BINS)
    ax.loglog(x_lmf_lambda, y_lmf_lambda, marker='v', linestyle='-', color='green', lw=1.2, label=r'LMF $\alpha = 1.5$ $\lambda = 0.3$')

    # LMF Lambda random
    x_lmf_lambda_rand, y_lmf_lambda_rand = log_bin_acf(pooled_lmf_lambda_rand, max_lag, num_bins=NUM_BINS)
    ax.loglog(x_lmf_lambda_rand, y_lmf_lambda_rand, marker='^', linestyle='-', color='orange', lw=1.2, label=r'LMF $\alpha = 1.5$ $\lambda = 0.3$ $p_{RAND} = 0.27$')

    # Neural Network
    x_nn, y_nn = log_bin_acf(pooled_nn, max_lag, num_bins=NUM_BINS)
    ax.loglog(x_nn, y_nn, marker='s', linestyle='-', color='purple', lw=1.2, label='Neural Network')

    # ── 6. Teoria e Output ────────────────────────────────────────────────────
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