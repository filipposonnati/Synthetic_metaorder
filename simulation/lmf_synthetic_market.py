"""
lmf_synthetic_market.py
=======================
Simulatore di mercato sintetico (sostituisce lmf_tim.py e lmf_tim_no_param.py).

PIPELINE (giorno per giorno)
----------------------------
  1. SEGNI   - Long Memory Flow (LMF)
  2. VOLUMI  - 'empirical' | 'mem_acd' | 'log_ar'  (modulo volumes_generation)
  3. PREZZI  - modello calibrato sul giorno reale con le funzioni di
               real_data_price_gen e applicato a segni/volumi sintetici:
                 'regression' : OLS su p+1 lag di sign*|v|^delta
                 'tim'        : Transient Impact Model (kernel differenziato)

I timestamp dell'output sono quelli reali del giorno; segni, volumi e prezzi
sono sintetici. Le funzioni di I/O, fit e simulazione dei prezzi (con la
convenzione `impact_lag`) vengono tutte da real_data_price_gen.py.
"""

import sys
from pathlib import Path

import numpy as np

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from lmf import simulate_lmf, simulate_lmf_lambda
from volumes_generation import (
    fit_ar_log_volume,
    simulate_ar_log_volume,
    fit_mem_acd,
    simulate_mem_acd,
    sample_empirical_volumes,
)
from real_data_price_gen import (
    DEFAULT_DATA_DIR,
    open_real_data,
    save_simulated_data,
    fit_ols,
    simulate_ols,
    fit_tim,
    simulate_tim,
)

DEFAULT_OUT_DIR = Path("..") / "database" / "data_synthetic"


# ---------------------------------------------------------------------------
# LAYER 1 - SEGNI
# ---------------------------------------------------------------------------

def generate_signs(n_trades, alpha, n_traders=None, lambda_lmf=None):
    if n_traders is not None:
        signs = simulate_lmf(alpha, n_traders, n_trades)
    elif lambda_lmf is not None:
        signs, _, _ = simulate_lmf_lambda(alpha, lambda_lmf, n_trades)
    else:
        raise ValueError("Fornire n_traders o lambda_lmf per LMF.")
    return np.asarray(signs, dtype=float)


# ---------------------------------------------------------------------------
# LAYER 2 - VOLUMI
# ---------------------------------------------------------------------------

def generate_volumes(n_trades, volumes_real, volume_model, rng,
                     mem_p=1, mem_q=1, mem_dist="burr12", ar_order=100):
    if volume_model in ("empirical", "sample", "real"):
        return sample_empirical_volumes(volumes_real, n_trades, rng)

    if volume_model in ("mem_acd", "mem"):
        params = fit_mem_acd([volumes_real], p=mem_p, q=mem_q, dist=mem_dist,
                             sample_days=1, rng=rng)
        return simulate_mem_acd(params, n_trades, rng)

    if volume_model == "log_ar":
        p_eff = ar_order if len(volumes_real) > ar_order else max(1, len(volumes_real) // 10)
        params = fit_ar_log_volume([volumes_real], p=p_eff, sample_days=1, rng=rng)
        return simulate_ar_log_volume(params, n_trades, rng)

    raise ValueError(f"volume_model sconosciuto: {volume_model!r}")


# ---------------------------------------------------------------------------
# LAYER 3 - PREZZI (calibrazione sul giorno reale, simulazione su segni/volumi sintetici)
# ---------------------------------------------------------------------------

def generate_prices(price_model, prices_real, volumes_real, signs_real,
                    volumes_sim, signs_sim, P0, rng,
                    p=1000, beta=0.25, delta=1.0, kernel_L=500,
                    impact_lag=1, tick_size=None):
    if price_model == "regression":
        const, theta, sigma = fit_ols(prices_real, volumes_real, signs_real, p,
                                      delta=delta, impact_lag=impact_lag)
        return simulate_ols(P0, volumes_sim, signs_sim, const, theta, sigma, rng,
                            delta=delta, impact_lag=impact_lag, tick_size=tick_size)

    if price_model == "tim":
        L = min(kernel_L, len(prices_real) - 1)
        const, sigma_f, sigma_eta = fit_tim(prices_real, volumes_real, signs_real,
                                            beta, delta, L, impact_lag)
        return simulate_tim(P0, volumes_sim, signs_sim, const, sigma_f, sigma_eta,
                            beta, delta, L, rng, impact_lag, tick_size)

    raise ValueError(f"price_model sconosciuto: {price_model!r}")


# ---------------------------------------------------------------------------
# PIPELINE GIORNALIERA
# ---------------------------------------------------------------------------

def simulate_day(prices_real, volumes_real, signs_real, rng, *,
                 alpha, n_traders, lambda_lmf,
                 volume_model, mem_p, mem_q, mem_dist, ar_order,
                 price_model, vol_lags_for_ret, beta, delta, kernel_L,
                 impact_lag, tick_size):
    n_trades = len(prices_real)
    P0 = float(prices_real[0])

    signs = generate_signs(n_trades, alpha, n_traders, lambda_lmf)
    volumes = generate_volumes(n_trades, volumes_real, volume_model, rng,
                               mem_p, mem_q, mem_dist, ar_order)
    prices = generate_prices(price_model, prices_real, volumes_real, signs_real,
                             volumes, signs, P0, rng,
                             p=vol_lags_for_ret, beta=beta, delta=delta,
                             kernel_L=kernel_L, impact_lag=impact_lag,
                             tick_size=tick_size)
    return prices, volumes, signs


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run(data_dir=DEFAULT_DATA_DIR,
        out_dir=DEFAULT_OUT_DIR,
        # segni
        alpha=1.5,
        n_traders=None,
        lambda_lmf=0.3,
        # volumi
        volume_model="empirical",     # 'empirical' | 'mem_acd' | 'log_ar'
        mem_dist="burr12",            # 'inverse_gaussian' | 'lognormal' | 'burr12'
        mem_p=1,
        mem_q=1,
        ar_order=100,
        # prezzi
        price_model="tim",            # 'regression' | 'tim'
        vol_lags_for_ret=1000,        # p per OLS
        beta=0.25,                    # solo TIM
        delta=1.0,                    # sign*|v|^delta (OLS e TIM)
        kernel_L=500,                 # solo TIM
        impact_lag=1,                 # 1: prezzo pre-trade; 0: prezzo di esecuzione
        tick_size=None,
        pattern="*.csv",
        seed=42):

    if price_model not in ("regression", "tim"):
        raise ValueError(f"price_model sconosciuto: {price_model!r}")

    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in data_dir.glob(pattern) if f.is_file())
    if not files:
        raise FileNotFoundError(f"nessun file '{pattern}' in {data_dir}")

    print(f"Simulazione giorno per giorno | segni=LMF | volumi={volume_model} | prezzi={price_model}")

    n_done = 0
    for f in files:
        timestamps, prices_real, volumes_real, signs_real = open_real_data(f.name, data_dir)

        try:
            prices, volumes, signs = simulate_day(
                prices_real, volumes_real, signs_real, rng,
                alpha=alpha, n_traders=n_traders, lambda_lmf=lambda_lmf,
                volume_model=volume_model, mem_p=mem_p, mem_q=mem_q,
                mem_dist=mem_dist, ar_order=ar_order,
                price_model=price_model, vol_lags_for_ret=vol_lags_for_ret,
                beta=beta, delta=delta, kernel_L=kernel_L,
                impact_lag=impact_lag, tick_size=tick_size)
        except ValueError as e:
            print(f"  !! Giorno {f.name} saltato: {e}")
            continue

        save_simulated_data(out_dir / f.name, timestamps, prices, volumes, signs)
        n_done += 1
        print(f"  -> Giorno {f.name} completato.")

    print(f"\nSimulazione completata: {n_done}/{len(files)} giorni generati.")


if __name__ == "__main__":
    run(
        data_dir=DEFAULT_DATA_DIR,
        out_dir=Path("..") / "database" / "data_lmf_1.5_0.3_real_reg_lin",
        alpha=1.5,
        n_traders=None,
        lambda_lmf=0.3,
        volume_model="empirical",   # 'empirical' | 'mem_acd' | 'log_ar'
        price_model="regression",          # 'regression' | 'tim'
        vol_lags_for_ret=1000,
        beta=0.25,
        delta=1.0,
        kernel_L=500,
        impact_lag=1,
        tick_size=None,
        seed=42,
    )