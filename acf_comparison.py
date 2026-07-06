"""
part3_compare_acf.py
────────────────────
Loads the empirical pooled ACF (from part 1) and the van-Vleck-reconstructed
binary ACF (from part 2), then generates two synthetic sign series using the
LMF models (fixed-N and λ), computes their ACFs, and plots everything on a
single figure for comparison.

Dependencies
------------
- database/acf_binary.npy   (pooled binary ACF, produced by part1_acf_dfa.py)
- database/p_plus.npy        (empirical fraction of +1s)
- lmf.py                     (simulate_lmf, simulate_lmf_lambda)
- part2_reconstruct_binary.py (generate_binary_sequence)

Usage
-----
    python part3_compare_acf.py
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from pathlib import Path
import os
import sys

# ── local imports ─────────────────────────────────────────────────────────────
# Allow running from any working directory as long as the scripts are siblings
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from lmf import simulate_lmf, simulate_lmf_lambda
from series_gaussian import generate_binary_sequence

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS  (edit here to tune the comparison)
# ══════════════════════════════════════════════════════════════════════════════

MAX_LAG       = 1_000      # lags to compute/plot
TOTAL_STEPS   = 10_000_000  # length of LMF synthetic series

# Fixed-N LMF parameters
LMF_ALPHA     = 1.5
LMF_N_TRADERS = 100

# λ-model parameters
LMF_LAMBDA_ALPHA = 1.8
LMF_LAMBDA       = 0.3     # must be < λ_c = (alpha-1)/alpha

# Van Vleck reconstruction parameters (part 2)
VV_N_REALIZATIONS = 50     # number of binary realisations to average over
VV_SEED           = 42

# Plot styling
SAVE_PATH = os.path.join('images', 'acf', 'acf_comparison.png')

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def mean_acf(series_list, max_lag):
    """Average ACF over a list of 1-D arrays (suppresses estimator noise)."""
    return np.mean(
        [acf(s, nlags=max_lag, fft=True) for s in series_list], axis=0
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── 1. Load empirical ACF ─────────────────────────────────────────────────
    acf_binary_path = Path('database/acf_binary.npy')
    p_plus_path     = Path('database/p_plus.npy')

    if not acf_binary_path.is_file():
        raise FileNotFoundError(
            "database/acf_binary.npy not found. Run part1_acf_dfa.py first."
        )

    pooled_acf = np.load(acf_binary_path)
    p_plus     = float(np.load(p_plus_path)) if p_plus_path.is_file() else 0.5
    max_lag    = min(MAX_LAG, len(pooled_acf) - 1)
    lags       = np.arange(1, max_lag + 1)

    print(f"Loaded empirical ACF  (max_lag={max_lag},  p_plus={p_plus:.4f})")

    # ── 2. Van Vleck reconstruction ACF (part 2) ──────────────────────────────
    median_len_path = Path('database/median_len.npy')
    median_len      = int(np.load(median_len_path)) if median_len_path.is_file() else 1_000_000
    N_recon         = max(median_len, 1000 * max_lag)

    print(f"Generating Van Vleck binary realisation")
    vv_binaries, _ = generate_binary_sequence(
        pooled_acf, p_plus=p_plus,
        N=TOTAL_STEPS, n_realizations=1, seed=42,
    )
    acf_vv = acf(vv_binaries[0], nlags=max_lag, fft=True)

    # ── 3. Fixed-N LMF ───────────────────────────────────────────────────────
    print(f"Simulating fixed-N LMF  (alpha={LMF_ALPHA}, "
          f"N={LMF_N_TRADERS}, steps={TOTAL_STEPS}) …")
    flow_fixed = simulate_lmf(LMF_ALPHA, LMF_N_TRADERS, TOTAL_STEPS)
    acf_fixed  = acf(flow_fixed, nlags=max_lag, fft=True)
    print("  done.")

    # ── 4. λ-model LMF ───────────────────────────────────────────────────────
    lam_c = (LMF_LAMBDA_ALPHA - 1) / LMF_LAMBDA_ALPHA
    print(f"Simulating λ-model LMF  (alpha={LMF_LAMBDA_ALPHA}, "
          f"lambda={LMF_LAMBDA}, lambda_c={lam_c:.3f}, steps={TOTAL_STEPS}) …")
    flow_lambda, _, _ = simulate_lmf_lambda(LMF_LAMBDA_ALPHA, LMF_LAMBDA, TOTAL_STEPS)
    acf_lambda     = acf(flow_lambda, nlags=max_lag, fft=True)
    print("  done.")

    # ── 5. Theoretical power-law references ───────────────────────────────────
    # γ = α - 1  →  ACF ∝ τ^{-γ}
    gamma_fixed  = LMF_ALPHA - 1
    gamma_lambda = LMF_LAMBDA_ALPHA - 1

    # Anchor each theory line at lag 1 of the corresponding simulated ACF
    theory_fixed  = acf_fixed[1]  * lags ** (-gamma_fixed)
    theory_lambda = acf_lambda[1] * lags ** (-gamma_lambda)

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Empirical (real data)
    ax.loglog(lags, pooled_acf[1:max_lag + 1],
              color='black',      lw=1.8,
              label='Empirical pooled ACF (real data)')

    # Van Vleck reconstruction
    ax.loglog(lags, acf_vv[1:],
              color='steelblue',  lw=1.4, linestyle='--',
              label=f'Van Vleck reconstruction (avg {VV_N_REALIZATIONS} realisations)')

    # Fixed-N LMF
    ax.loglog(lags, acf_fixed[1:],
              color='tomato',     lw=1.4,
              label=rf'LMF fixed-N  ($\alpha={LMF_ALPHA}$, $N={LMF_N_TRADERS}$)')
    ax.loglog(lags, theory_fixed,
              color='tomato',     lw=1.0, linestyle=':',
              label=rf'Theory fixed-N  $\tau^{{-{gamma_fixed:.2f}}}$')

    # λ-model LMF
    ax.loglog(lags, np.abs(acf_lambda[1:]),
              color='seagreen',   lw=1.4,
              label=rf'LMF $\lambda$-model  ($\alpha={LMF_LAMBDA_ALPHA}$, $\lambda={LMF_LAMBDA}$)')
    ax.loglog(lags, theory_lambda,
              color='seagreen',   lw=1.0, linestyle=':',
              label=rf'Theory $\lambda$-model  $\tau^{{-{gamma_lambda:.2f}}}$')

    ax.set_xlabel(r'Lag $\tau$')
    ax.set_ylabel(r'ACF $C(\tau)$')
    ax.set_title('ACF Comparison: empirical vs reconstructed vs LMF models')
    ax.legend(loc='lower left')
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()

    os.makedirs(os.path.join('images', 'acf'), exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {SAVE_PATH}")
    plt.show()