import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from pathlib import Path
import os
import sys

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

    theory  =  0.0550 * lags ** (-0.5)

    # ── 2. Applica Log-Binning ai dati simulati ed empirici ───────────────────
    NUM_BINS = 50

    # ── 3. Plot Setup ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    })

    fig, ax = plt.subplots(figsize=(12, 7))

    # Empirical (real data)
    x_emp, y_emp = log_bin_acf(pooled_acf, max_lag, num_bins=NUM_BINS)
    ax.loglog(x_emp, y_emp, marker='o', linestyle='-', color='black', lw=2.0, label='Empirical pooled ACF')

    # Configurazione test distribuzioni
    distributions_to_test = [
        # (label, dist_type, alpha_param, dist_kwargs, color, marker)
        ('Pareto (default)', 'pareto', LMF_ALPHA, {}, 'tomato', '^'),
        ('Zeta (Zipf)', 'zeta', LMF_ALPHA + 1.0, {}, 'chocolate', 's'),
        ('Yule-Simon', 'yule', LMF_ALPHA, {'rho': LMF_ALPHA}, 'darkorange', 'v'),
        ('Lomax', 'lomax', LMF_ALPHA, {'scale': 1.0}, 'mediumpurple', '<')
    ]

    LMF_N_TRADERS = 50

    # ── 4. Simulazioni Fixed-N con le differenti distribuzioni ────────────────
    for label, dist_type, alpha_val, kwargs, color, marker in distributions_to_test:
        print(f"Simulating fixed-N LMF with {dist_type}")
        flow = simulate_lmf(
            alpha=alpha_val, 
            n_traders=LMF_N_TRADERS, 
            total_steps=TOTAL_STEPS, 
            dist_type=dist_type, 
            dist_kwargs=kwargs
        )
        acf_val = acf(flow, nlags=max_lag, fft=True)
        x_bin, y_bin = log_bin_acf(acf_val, max_lag, num_bins=NUM_BINS)

        ax.loglog(x_bin, np.abs(y_bin), marker=marker, linestyle='-', color=color, lw=1.2, alpha=0.85,
                  label=f'{label} ($N={LMF_N_TRADERS}$)')

    # ── 5. Simulazione Lambda-model con distribuzione Zeta ────────────────────
    for label, dist_type, alpha_val, kwargs, color, marker in distributions_to_test:
        LMF_LAMBDA = 0.3
        print(f"Simulating λ-model LMF with {dist_type}")
        flow_lambda, _, _ = simulate_lmf_lambda(
            alpha=alpha_val, 
            lam=LMF_LAMBDA, 
            total_steps=TOTAL_STEPS, 
            dist_type=dist_type,
            dist_kwargs=kwargs
        )
        acf_lambda = acf(flow_lambda, nlags=max_lag, fft=True)
        x_lam, y_lam = log_bin_acf(acf_lambda, max_lag, num_bins=NUM_BINS)

        ax.loglog(x_lam, np.abs(y_lam), marker=marker, linestyle='-', color=color, lw=1.5, alpha=0.9,
                label=rf'LMF $\lambda$-model {label} ($\lambda={LMF_LAMBDA}$)')

    # Teoria
    ax.loglog(lags, theory, color='black', lw=1.5, linestyle=':', label=rf'Theory $\tau^{{-0.5}}$')

    ax.set_xlabel(r'Lag $\tau$')
    ax.set_ylabel(r'ACF $C(\tau)$')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()

    os.makedirs(os.path.join('images', 'acf'), exist_ok=True)
    fig.savefig(os.path.join('images', 'acf', 'acf_comparison_new.png'), dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {os.path.join('images', 'acf', 'acf_comparison_new.png')}")
    plt.show()