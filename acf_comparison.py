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
from series_gaussian import generate_binary_sequence

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
MAX_LAG       = 5_000      
TOTAL_STEPS   = 10_000_000  

LMF_ALPHA     = 1.5
LMF_N_TRADERS = 100

LMF_LAMBDA_ALPHA = 1.8
LMF_LAMBDA       = 0.3     

VV_N_REALIZATIONS = 50     
VV_SEED           = 42

SAVE_PATH = os.path.join('images', 'acf', 'acf_comparison.png')

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

    # ── 2. Van Vleck reconstruction ACF ───────────────────────────────────────
    print(f"Generating Van Vleck binary realisation")
    vv_binaries, _ = generate_binary_sequence(
        pooled_acf, p_plus=p_plus,
        N=TOTAL_STEPS, n_realizations=1, seed=42,
    )
    acf_vv = acf(vv_binaries[0], nlags=max_lag, fft=True)

    # ── 3. Fixed-N LMF ───────────────────────────────────────────────────────
    print(f"Simulating fixed-N LMF ...")
    flow_fixed = simulate_lmf(LMF_ALPHA, LMF_N_TRADERS, TOTAL_STEPS)
    acf_fixed  = acf(flow_fixed, nlags=max_lag, fft=True)

    # ── 4. λ-model LMF ───────────────────────────────────────────────────────
    lam_c = (LMF_LAMBDA_ALPHA - 1) / LMF_LAMBDA_ALPHA
    print(f"Simulating λ-model LMF ...")
    flow_lambda, _, _ = simulate_lmf_lambda(LMF_LAMBDA_ALPHA, LMF_LAMBDA, TOTAL_STEPS)
    acf_lambda     = acf(flow_lambda, nlags=max_lag, fft=True)

    # ── 5. Theoretical power-law references ───────────────────────────────────
    gamma_fixed  = LMF_ALPHA - 1
    gamma_lambda = LMF_LAMBDA_ALPHA - 1

    # Le linee teoriche continue non hanno bisogno di binning, usano i lag originali
    theory_fixed  = acf_fixed[1]  * lags ** (-gamma_fixed)
    theory_lambda = acf_lambda[1] * lags ** (-gamma_lambda)

    # ── 6. Applica Log-Binning ai dati simulati ed empirici ───────────────────
    # Puoi regolare NUM_BINS (es. 25 o 30) per avere più o meno dettaglio
    NUM_BINS = 30
    
    x_emp, y_emp     = log_bin_acf(pooled_acf, max_lag, num_bins=NUM_BINS)
    x_vv, y_vv       = log_bin_acf(acf_vv, max_lag, num_bins=NUM_BINS)
    x_fix, y_fix     = log_bin_acf(acf_fixed, max_lag, num_bins=NUM_BINS)
    x_lam, y_lam     = log_bin_acf(acf_lambda, max_lag, num_bins=NUM_BINS)

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Usiamo 'marker' (punti/quadrati) per i dati binnati, così si capisce che è una media
    # Empirical (real data)
    ax.loglog(x_emp, y_emp, marker='o', linestyle='-', color='black', lw=1.5, label='Empirical pooled ACF')

    # Van Vleck reconstruction
    ax.loglog(x_vv, y_vv, marker='s', linestyle='--', color='steelblue', lw=1.2, alpha=0.8, label=f'Van Vleck reconstruction')

    # Fixed-N LMF
    ax.loglog(x_fix, y_fix, marker='^', linestyle='-', color='tomato', lw=1.2, alpha=0.8,
              label=rf'LMF fixed-N  ($\alpha={LMF_ALPHA}$, $N={LMF_N_TRADERS}$)')
    # Teoria (linea pulita nativa)
    ax.loglog(lags, theory_fixed, color='tomato', lw=1.2, linestyle=':',
              label=rf'Theory fixed-N  $\tau^{{-{gamma_fixed:.2f}}}$')

    # λ-model LMF (usiamo np.abs sul binned se ci sono fluttuazioni negative)
    ax.loglog(x_lam, np.abs(y_lam), marker='d', linestyle='-', color='seagreen', lw=1.2, alpha=0.8,
              label=rf'LMF $\lambda$-model  ($\alpha={LMF_LAMBDA_ALPHA}$, $\lambda={LMF_LAMBDA}$)')
    # Teoria (linea pulita nativa)
    ax.loglog(lags, theory_lambda, color='seagreen', lw=1.2, linestyle=':',
              label=rf'Theory $\lambda$-model  $\tau^{{-{gamma_lambda:.2f}}}$')

    ax.set_xlabel(r'Lag $\tau$')
    ax.set_ylabel(r'ACF $C(\tau)$')
    ax.legend(loc='lower left')
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()

    os.makedirs(os.path.join('images', 'acf'), exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {SAVE_PATH}")
    plt.show()