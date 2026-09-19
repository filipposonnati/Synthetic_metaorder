"""
real_data_price_generator.py
=============================
Generatore di prezzi sintetici basato ESCLUSIVAMENTE su SEGNI E VOLUMI REALI
letti sequenzialmente (senza campionamento o simulazione stocastica di volumi/segni).

Supporta due modelli di generazione del prezzo:
  1. 'ar_regression' : Regressione OLS con p parametri liberi (simile a no_param)
  2. 'tim'           : Transient Impact Model con kernel a legge di potenza (simile a lmf_tim)
"""

import numpy as np
import pandas as pd
from os import listdir
from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def open_real_data(fname, data_dir=r"..\database\data"):
    trades = pd.read_csv(Path(data_dir) / fname, header=None)
    prices = np.array(trades[1], dtype=float)
    volumes = np.array(trades[2], dtype=float)
    signs = np.array(trades[3], dtype=float)
    return prices, volumes, signs


def save_simulated_data(out_path, prices, volumes, signs, n_trades):
    fp = Path(out_path)
    if fp.exists():
        fp.unlink()

    timestamps = np.linspace(36_000.0, 55_797.0, n_trades, endpoint=True)

    pd.DataFrame({
        0: timestamps,
        1: prices,
        2: np.abs(volumes),
        3: np.sign(signs).astype(int),
    }).to_csv(out_path, index=False, header=False)


# ---------------------------------------------------------------------------
# MODELLO 1 — REGRESSIONE OLS (p parametri liberi)
# ---------------------------------------------------------------------------

def calibrate_and_simulate_ols(prices_real, volumes_real, signs_real, P0, p, rng):
    if len(prices_real) <= p + 1:
        return prices_real

    log_prices = np.log(np.maximum(prices_real, 1e-12))
    returns = np.diff(log_prices)
    signed_v = volumes_real[:-1] * signs_real[:-1]
    n = len(returns)

    if n <= p:
        return prices_real

    # Matrice delle covariate sui lag dei volumi firmati reali
    X_day = np.column_stack([signed_v[p:]] + [signed_v[p - i : n - i] for i in range(1, p + 1)])
    y_day = returns[p:]

    # Stima OLS dei parametri
    params, residuals, _, _ = np.linalg.lstsq(X_day, y_day, rcond=None)
    
    if len(residuals) > 0:
        sigma = np.sqrt(residuals[0] / len(y_day))
    else:
        sigma = np.std(y_day - X_day @ params)

    # Simulazione prezzi usando volumi e segni reali
    n_trades = len(prices_real)
    signed_v_full = volumes_real * signs_real

    noise = rng.normal(0.0, sigma, size=n_trades)
    conv_res = np.convolve(signed_v_full, params, mode='full')[:n_trades]

    sim_r = conv_res + noise
    sim_r[:p] = rng.normal(0.0, sigma, p)

    prices = np.empty(n_trades, dtype=float)
    prices[0] = P0
    if n_trades > 1:
        prices[1:] = P0 * np.exp(np.cumsum(sim_r[1:]))

    return prices


# ---------------------------------------------------------------------------
# MODELLO 2 — TRANSIENT IMPACT MODEL (TIM con Kernel)
# ---------------------------------------------------------------------------

def _build_impact_signal(signs, volumes, beta, delta, kernel_L):
    T = len(signs)
    kernel = (np.arange(kernel_L, dtype=float) + 1.0) ** (-beta)
    raw = signs.astype(float) * (volumes ** delta)
    return np.convolve(raw, kernel, mode='full')[:T]


def calibrate_and_simulate_tim(prices_real, volumes_real, signs_real, P0, beta, delta, kernel_L, rng):
    if len(prices_real) < 2:
        return prices_real

    log_r = np.diff(np.log(np.maximum(prices_real, 1e-12)))
    X = _build_impact_signal(signs_real[:-1], volumes_real[:-1], beta, delta, min(kernel_L, len(signs_real) - 1))

    # Calibrazione TIM della giornata
    denom = np.dot(X, X)
    sigma_f = float(np.dot(X, log_r) / denom) if denom > 1e-12 else 0.0
    sigma_eta = float(np.std(log_r - sigma_f * X))

    # Simulazione
    T = len(signs_real)
    X_full = _build_impact_signal(signs_real, volumes_real, beta, delta, min(kernel_L, T))
    log_ret_sim = sigma_f * X_full + rng.normal(0.0, sigma_eta, T)

    prices = np.empty(T, dtype=float)
    prices[0] = P0
    if T > 1:
        prices[1:] = P0 * np.exp(np.cumsum(log_ret_sim[:-1]))

    return prices


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run(data_dir = r"..\database\data",
        out_dir = r"..\database\data_real_signs_vols_synthetic_prices",
        price_model = 'ar_regression',  # 'ar_regression' oppure 'tim'
        vol_lags_for_ret = 1000,         # Parametro p per OLS
        beta = 0.25,                     # Parametri per TIM
        delta = 1.0,
        kernel_L = 500,
        seed = 42):

    rng = np.random.default_rng(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths = sorted(listdir(data_dir))
    print(f"Inizio generazione prezzi da segni e volumi reali [Modello: {price_model}]...")

    for fname in paths:
        prices_real, volumes_real, signs_real = open_real_data(fname, data_dir)
        P0 = float(prices_real[0])
        n_trades = len(prices_real)

        if price_model == 'regression':
            prices_sim = calibrate_and_simulate_ols(
                prices_real, volumes_real, signs_real, P0=P0, p=vol_lags_for_ret, rng=rng
            )
        elif price_model == 'tim':
            prices_sim = calibrate_and_simulate_tim(
                prices_real, volumes_real, signs_real, P0=P0, beta=beta, delta=delta, kernel_L=kernel_L, rng=rng
            )
        else:
            raise ValueError(f"price_model sconosciuto: {price_model!r}")

        # Salva mantendendo volumi e segni REALI immutati
        save_simulated_data(str(out_path / fname), prices_sim, volumes_real, signs_real, n_trades)
        print(f"  -> Giorno {fname} completato.")

    print("\nElaborazione completata con successo.")


if __name__ == '__main__':
    run(
        data_dir         = r"..\database\data",
        out_dir          = r"..\database\data_real_tim_lin",
        price_model      = 'tim',  # 'regression' | 'tim'
        vol_lags_for_ret = 1000,
        beta             = 0.25,
        delta            = 1.0,
        kernel_L         = 500,
        seed             = 42,
    )