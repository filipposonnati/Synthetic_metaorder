"""
synthetic_market.py / lmf_tim_param.py
======================================
Simulatore di mercato sintetico a tre strati.
L'output e' un CSV per giorno nel formato IDENTICO ai dati reali,
leggibile direttamente da methods.generate() e methods.generate_slim().

PIPELINE
--------
  1. SEGNI   — Long Memory Flow (LMF)
  2. VOLUMI  — Modello del volume (importato da volumes_generation)
  3. PREZZI  — Transient Impact Model (TIM), calibrato sui dati reali.
"""

import numpy as np
import pandas as pd
from os import listdir
import sys
from pathlib import Path

# Ottiene la cartella genitore (1 livello sopra)
parent_dir = Path(__file__).resolve().parent.parent

# Aggiunge il percorso a sys.path
sys.path.append(str(parent_dir))

# Importazione dal modulo lmf esterno
from lmf import simulate_lmf, simulate_lmf_lambda

# Importazione dei metodi per la gestione dei volumi
from volumes_generation import (
    fit_ar_log_volume,
    simulate_ar_log_volume,
    fit_mem_acd,
    simulate_mem_acd,
    sample_empirical_volumes
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def open_real_data(fname, data_dir=r"..\database\data"):
    """
    Legge un file reale:
        col0=timestamp_sec, col1=prezzo, col2=volume_modulo, col3=segno
    """
    trades = pd.read_csv(Path(data_dir) / fname, header=None)
    prices = np.array(trades[1], dtype=float)
    volumes = np.array(trades[2], dtype=float)
    signs = np.array(trades[3], dtype=float)
    return prices, volumes, signs


def save_simulated_data(out_path, prices, volumes, signs, n_trades):
    """
    Salva nel formato identico ai dati reali:
        timestamp_sec, prezzo, volume_in_modulo, segno_volume
    """
    fp = Path(out_path)
    if fp.exists():
        fp.unlink()

    timestamps = np.linspace(36_000.0, 55_797.0, n_trades, endpoint=True)

    pd.DataFrame({
        0: timestamps,
        1: prices,                          # prezzo assoluto P_t
        2: np.abs(volumes),
        3: np.sign(signs).astype(int),
    }).to_csv(out_path, index=False, header=False)


# ---------------------------------------------------------------------------
# LAYER 3 — PREZZI  (Transient Impact Model)
# ---------------------------------------------------------------------------

def _build_impact_signal(signs, volumes, beta, delta, kernel_L):
    T = len(signs)
    kernel = (np.arange(kernel_L, dtype=float) + 1.0) ** (-beta)
    raw = signs.astype(float) * (volumes ** delta)
    return np.convolve(raw, kernel, mode='full')[:T]


def calibrate_tim(data_dir, beta, delta, kernel_L):
    all_r, all_X = [], []
    for fname in sorted(listdir(data_dir)):
        prices, volumes, signs = open_real_data(fname, data_dir)
        if len(prices) < 2: 
            continue
        log_r = np.diff(np.log(np.maximum(prices, 1e-12)))
        X = _build_impact_signal(signs[:-1], volumes[:-1], beta, delta, min(kernel_L, len(signs) - 1))
        all_r.append(log_r)
        all_X.append(X)

    r_all = np.concatenate(all_r)
    X_all = np.concatenate(all_X)
    sigma_f = float(np.dot(X_all, r_all) / np.dot(X_all, X_all))
    sigma_eta = float(np.std(r_all - sigma_f * X_all))
    return sigma_f, sigma_eta


def simulate_tim(signs, volumes, beta, delta, sigma_f, sigma_eta, kernel_L, P0, rng):
    T = len(signs)
    X = _build_impact_signal(signs, volumes, beta, delta, kernel_L)
    log_ret = sigma_f * X + rng.normal(0.0, sigma_eta, T)
    prices = np.empty(T, dtype=float)
    prices[0] = P0
    if T > 1:
        prices[1:] = P0 * np.exp(np.cumsum(log_ret[:-1]))
    return prices


# ---------------------------------------------------------------------------
# PIPELINE GIORNALIERA
# ---------------------------------------------------------------------------

def simulate_day(n_trades, volumes_real, volume_model, alpha, n_traders, lambda_lmf, beta,
                 delta, sigma_f, sigma_eta, kernel_L, P0, rng, mem_p=1, mem_q=1, 
                 mem_dist='burr12', ar_order=100):
    
    # 1. SEGNI (LMF)
    if n_traders is not None:
        signs = simulate_lmf(alpha, n_traders, n_trades)
    elif lambda_lmf is not None:
        signs, _, _ = simulate_lmf_lambda(alpha, lambda_lmf, n_trades)
    else:
        sys.exit("Fornire n_traders o lambda_lmf per LMF.")

    # 2. VOLUMI (Chiamata Diretta Day-by-Day)
    if volume_model in ('empirical', 'sample', 'real'):
        volumes = sample_empirical_volumes(volumes_real, n_trades, rng)

    elif volume_model in ('mem_acd', 'mem'):
        vol_params = fit_mem_acd([volumes_real], p=mem_p, q=mem_q, dist=mem_dist, sample_days=1, rng=rng)
        volumes = simulate_mem_acd(vol_params, n_trades, rng)

    elif volume_model == 'log_ar':
        p_eff = ar_order if len(volumes_real) > ar_order else max(1, len(volumes_real) // 10)
        vol_params = fit_ar_log_volume([volumes_real], p=p_eff, sample_days=1, rng=rng)
        volumes = simulate_ar_log_volume(vol_params, n_trades, rng)

    else:
        raise ValueError(f"volume_model sconosciuto: {volume_model!r}")

    # 3. PREZZI (TIM)
    prices = simulate_tim(signs, volumes, beta=beta, delta=delta,
                          sigma_f=sigma_f, sigma_eta=sigma_eta,
                          kernel_L=min(kernel_L, n_trades), P0=P0, rng=rng)
    return prices, volumes, signs


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run(data_dir = r"..\database\data",
        out_dir = r"..\database\data_synthetic",
        alpha = 1.5,
        n_traders = None,
        lambda_lmf = None,
        volume_model = 'empirical',  # 'empirical' | 'mem_acd' | 'log_ar'
        mem_dist = 'burr12',          # 'inverse_gaussian' | 'lognormal' | 'burr12'
        mem_p = 1,
        mem_q = 1,
        ar_order = 100,
        beta = 0.25,
        delta = 0.5,
        kernel_L = 500,
        seed = 42):
    
    rng = np.random.default_rng(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sigma_f, sigma_eta = calibrate_tim(data_dir, beta=beta, delta=delta, kernel_L=kernel_L)
    paths = sorted(listdir(data_dir))

    for fname in paths:
        prices_real, volumes_real, _ = open_real_data(fname, data_dir)
        P0 = float(prices_real[0])
        n_trades = len(prices_real)

        print(f"-> {fname}  |  n_trades={n_trades}  |  Volume Model={volume_model}")

        prices, volumes, signs = simulate_day(
            n_trades=n_trades,
            volumes_real=volumes_real,
            volume_model=volume_model,
            alpha=alpha,
            n_traders=n_traders,
            lambda_lmf=lambda_lmf,
            beta=beta,
            delta=delta,
            sigma_f=sigma_f,
            sigma_eta=sigma_eta,
            kernel_L=kernel_L,
            P0=P0,
            rng=rng,
            mem_p=mem_p,
            mem_q=mem_q,
            mem_dist=mem_dist,
            ar_order=ar_order
        )

        save_simulated_data(str(out_path / fname), prices, volumes, signs, n_trades)

    print("\nSimulazione completata.")


if __name__ == '__main__':
    run(
        data_dir     = r"..\database\data",
        out_dir      = r"..\database\data_lmf_1.5_0.3_real_tim_lin",
        alpha        = 1.5,
        n_traders    = None,
        lambda_lmf   = 0.3,
        volume_model = 'empirical',  # 'empirical' | 'mem_acd' | 'log_ar'
        mem_dist     = 'burr12',      # 'inverse_gaussian' | 'lognormal' | 'burr12'
        mem_p        = 1,
        mem_q        = 1,
        beta         = 0.25,
        delta        = 1.0,
        kernel_L     = 500,
        seed         = 42,
    )