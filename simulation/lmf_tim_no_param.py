"""
Simulatore di mercato sintetico indipendente con FIT E SIMULAZIONE GIORNO PER GIORNO.
"""

import numpy as np
import pandas as pd
from os import listdir
from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from lmf import simulate_lmf, simulate_lmf_lambda
from volumes_generation import fit_ar_log_volume, simulate_ar_log_volume, fit_mem_acd, simulate_mem_acd, sample_empirical_volumes


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
# CALIBRAZIONE SINGOLA GIORNATA (RENDIMENTI)
# ---------------------------------------------------------------------------

def calibrate_return_volume_single_day(prices, volumes, signs, p=10):
    """Calibra il modello OLS sui rendimenti esclusivamente per la singola giornata fornita."""
    if len(prices) <= p + 1:
        return None

    log_prices = np.log(np.maximum(prices, 1e-12))
    returns = np.diff(log_prices)
    signed_v = volumes[:-1] * signs[:-1]
    n = len(returns)

    if n <= p:
        return None

    X_day = np.column_stack([signed_v[p:]] + [signed_v[p - i : n - i] for i in range(1, p + 1)])
    y_day = returns[p:]

    params_day, residuals, rank, s = np.linalg.lstsq(X_day, y_day, rcond=None)
    
    if len(residuals) > 0:
        sigma_day = np.sqrt(residuals[0] / len(y_day))
    else:
        sigma_day = np.std(y_day - X_day @ params_day)

    return {
        "p": p,
        "params": params_day,
        "sigma": sigma_day
    }


def simulate_returns_from_volumes(signs, volumes, ret_model_params, P0, rng):
    p = ret_model_params['p']
    r_params = ret_model_params['params']
    r_sigma = ret_model_params['sigma']

    n_trades = len(signs)
    signed_v = volumes * signs

    noise = rng.normal(0.0, r_sigma, size=n_trades)
    conv_res = np.convolve(signed_v, r_params, mode='full')[:n_trades]

    sim_r = conv_res + noise
    sim_r[:p] = rng.normal(0.0, r_sigma, p)

    prices = np.empty(n_trades, dtype=float)
    prices[0] = P0
    if n_trades > 1:
        prices[1:] = P0 * np.exp(np.cumsum(sim_r[1:]))

    return prices, sim_r

def simulate_day(n_trades, volumes_real, volume_model, alpha, n_traders, lambda_lmf,
                 ret_model_params, P0, rng, mem_p=1, mem_q=1, mem_dist='burr12', ar_order_vol=10):
    
    # 1. Generazione Segni (LMF)
    if n_traders is not None:
        signs = simulate_lmf(alpha, n_traders, n_trades)
    elif lambda_lmf is not None:
        signs, _, _ = simulate_lmf_lambda(alpha, lambda_lmf, n_trades)
    else:
        sys.exit("Né n_traders né lambda_lmf sono stati forniti.")

    # 2. Generazione Volumi (Chiamate Dirette)
    if volume_model in ('empirical', 'sample', 'real'):
        volumes = sample_empirical_volumes(volumes_real, n_trades, rng)

    elif volume_model in ('mem_acd', 'mem'):
        vol_params = fit_mem_acd([volumes_real], p=mem_p, q=mem_q, dist=mem_dist, sample_days=1, rng=rng)
        volumes = simulate_mem_acd(vol_params, n_trades, rng)

    elif volume_model == 'log_ar':
        vol_params = fit_ar_log_volume([volumes_real], p=ar_order_vol, sample_days=1, rng=rng)
        volumes = simulate_ar_log_volume(vol_params, n_trades, rng)

    else:
        raise ValueError(f"volume_model sconosciuto: {volume_model!r}")

    # 3. Generazione Prezzi
    prices, _ = simulate_returns_from_volumes(signs, volumes, ret_model_params, P0=P0, rng=rng)

    return prices, volumes, signs

# ---------------------------------------------------------------------------
# ENTRY POINT CON PIPELINE GIORNO PER GIORNO
# ---------------------------------------------------------------------------

def run(data_dir = r"..\database\data",
        out_dir = r"..\database\data_generated",
        alpha = 1.8,
        n_traders = None,
        lambda_lmf = 0.3,
        volume_model = 'empirical',      # 'mem' | 'log_ar'
        mem_dist = 'burr12',
        mem_p = 1,
        mem_q = 1,
        ar_order_vol = 10,
        vol_lags_for_ret = 1000,
        seed = 42):

    rng = np.random.default_rng(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths = sorted(listdir(data_dir))
    
    print("Inizio elaborazione, fit e simulazione giorno per giorno...")
    
    for fname in paths:
        prices_real, volumes_real, signs_real = open_real_data(fname, data_dir)
        P0 = float(prices_real[0])
        n_trades = len(prices_real)

        # 1. Fit Rendimenti per il SINGOLO giorno reale
        ret_model_params = calibrate_return_volume_single_day(
            prices_real, volumes_real, signs_real, p=vol_lags_for_ret
        )
        if ret_model_params is None:
            continue

        # 2. Fit Volumi per il SINGOLO giorno reale
        if volume_model in ('mem_acd', 'mem'):
            vol_params = fit_mem_acd([volumes_real], p=mem_p, q=mem_q, dist=mem_dist, sample_days=1, rng=rng)
        elif volume_model == 'log_ar':
            vol_params = fit_ar_log_volume([volumes_real], p=ar_order_vol, sample_days=1, rng=rng)
        else:
            vol_params = None

        # 3. Simulazione del giorno usando i parametri specifici di quel giorno
        prices, volumes, signs = simulate_day(
            n_trades=n_trades,
            volumes_real=volumes_real,
            volume_model=volume_model,
            alpha=alpha,
            n_traders=n_traders,
            lambda_lmf=lambda_lmf,
            ret_model_params=ret_model_params,
            P0=P0,
            rng=rng,
            mem_p=mem_p,
            mem_q=mem_q,
            mem_dist=mem_dist,
            ar_order_vol=ar_order_vol
        )

        save_simulated_data(str(out_path / fname), prices, volumes, signs, n_trades)
        print(f"  -> Giorno {fname} completato.")

    print("\nSimulazione completata con successo.")


if __name__ == '__main__':
    run(
        data_dir         = r"..\database\data",
        out_dir          = r"..\database\data_lmf_1.8_0.3_real_no_param_returns",
        alpha            = 1.8,
        n_traders        = None,
        lambda_lmf       = 0.3,
        volume_model     = 'empirical',
        ar_order_vol     = 10,
        vol_lags_for_ret = 1000,
        seed             = 42,
    )