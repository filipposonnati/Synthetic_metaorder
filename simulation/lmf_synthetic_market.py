"""
lmf_synthetic_market.py
=======================
Simulatore di mercato sintetico, giorno per giorno.

PIPELINE
--------
  1. SEGNI   - Long Memory Flow (lmf.simulate_lmf / simulate_lmf_lambda)
  2. VOLUMI  - senza segno, campionati con volumes_generation:
                 'empirical' : bootstrap dai volumi reali del giorno
                 'mem_acd'   : modello MEM/ACD gia' calibrato (models/mem_<dist>_<p>_<q>)
                 'log_ar'    : AR(p) sui log-volumi gia' calibrato (models/logar_<p>)
  3. PREZZI  - modello gia' calibrato e salvato, applicato ai volumi con segno
               v_t = sign_t * volume_t:
                 'tim'       : tim.py        (models/tim_<delta>_<p>), r semplici
                 'tim_parametric' : tim_parametric.py  (models/tim_parametric_<delta>_<p>), log-rendimenti

I modelli NON vengono ricalibrati: si caricano i model.npz prodotti dagli script di fit.
I timestamp dell'output sono quelli reali del giorno; segni, volumi e prezzi sono sintetici.
La lunghezza di ogni giorno e' quella del giorno reale. Per dare ai kernel di impatto
(p volumi precedenti) una storia coerente, si generano n_trades + p trade e i primi p
fanno da initial_v (poi scartati dall'output).
"""

import sys
from pathlib import Path

import numpy as np

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from lmf import simulate_lmf, simulate_lmf_lambda
import volumes_generation as vg
import tim
import tim_parametric

import pandas as pd

DEFAULT_DATA_DIR = Path("..") / "database" / "data"


# ---------------------------------------------------------------------------
# I/O (CSV senza header: tempo [s], prezzo, volume, segno; stesso formato di methods.py)
# ---------------------------------------------------------------------------

def open_real_data(filename, data_dir=DEFAULT_DATA_DIR):
    """Legge un giorno reale. Ritorna (timestamps, prices, volumes, signs)."""
    df = pd.read_csv(Path(data_dir) / filename, header=None)
    return (df[0].to_numpy(dtype=float), df[1].to_numpy(dtype=float),
            np.abs(df[2].to_numpy(dtype=float)), df[3].to_numpy(dtype=float))


def save_simulated_data(path, timestamps, prices, volumes, signs):
    """Salva un giorno sintetico nello stesso formato dei dati reali."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({0: np.asarray(timestamps, dtype=float),
                  1: np.asarray(prices, dtype=float),
                  2: np.asarray(volumes, dtype=float),
                  3: np.asarray(signs).astype(int)}
                 ).to_csv(path, header=False, index=False)


DEFAULT_OUT_DIR = Path("..") / "database" / "data_synthetic"
MODELS_DIR = Path("..") / "models"


# ---------------------------------------------------------------------------
# CARICAMENTO MODELLI SALVATI
# ---------------------------------------------------------------------------

def _load(path, loader, what):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"modello {what} non trovato: {path} "
                                f"(eseguire prima lo script di fit corrispondente)")
    return loader(path)


def load_volume_model(volume_model, models_dir, p_ar, mem_p, mem_q, mem_dist):
    """Ritorna i parametri del modello di volumi salvato (None per 'empirical')."""
    if volume_model == "empirical":
        return None
    if volume_model == "log_ar":
        name = f"logar_{p_ar}"
    elif volume_model == "mem_acd":
        name = f"mem_{mem_dist}_{mem_p}_{mem_q}"
    else:
        raise ValueError(f"volume_model sconosciuto: {volume_model!r}")
    return _load(Path(models_dir) / name / "model.npz", vg.load_volume_model, name)


def load_price_model(price_model, models_dir, delta, p_price):
    """Ritorna il modello dei prezzi salvato."""
    if price_model == "tim":
        name = f"tim_{delta}_{p_price}"
        return _load(Path(models_dir) / name / "model.npz", tim.load_impact_model, name)
    if price_model == "tim_parametric":
        name = f"tim_param_{delta}_{p_price}"
        return _load(Path(models_dir) / name / "model.npz", tim_parametric.load_tim_param_model, name)
    raise ValueError(f"price_model sconosciuto: {price_model!r}")


# ---------------------------------------------------------------------------
# LAYER 1 - SEGNI
# ---------------------------------------------------------------------------

def generate_signs(n_steps, alpha, n_traders=None, lambda_lmf=None):
    if n_traders is not None:
        signs = simulate_lmf(alpha, n_traders, n_steps)
    elif lambda_lmf is not None:
        signs, _, _ = simulate_lmf_lambda(alpha, lambda_lmf, n_steps)
    else:
        raise ValueError("Fornire n_traders o lambda_lmf per LMF.")
    return np.asarray(signs, dtype=float)


# ---------------------------------------------------------------------------
# LAYER 2 - VOLUMI (senza segno)
# ---------------------------------------------------------------------------

def generate_volumes(n_steps, volumes_real, volume_model, volume_params, rng):
    if volume_model == "empirical":
        return vg.sample_empirical_volumes(np.abs(volumes_real), n_steps, rng)
    if volume_model == "log_ar":
        return vg.simulate_ar_log_volume(volume_params, n_steps, rng)
    if volume_model == "mem_acd":
        return vg.simulate_mem_acd(volume_params, n_steps, rng)
    raise ValueError(f"volume_model sconosciuto: {volume_model!r}")


# ---------------------------------------------------------------------------
# LAYER 3 - PREZZI
# ---------------------------------------------------------------------------

def _tim_parametric_prices(rm, sim_v, initial_v, P0, rng, bootstrap):
    """Prezzi dal TIM parametrico dati i volumi con segno (stessa logica di
    tim_parametric.tim_parametric_simulate, ma con volumi esterni)."""
    p = rm["p"]
    n = len(sim_v)
    K = tim_parametric.return_kernel(rm["beta"], rm["kernel_L"])
    tv = tim_parametric.power_transform(np.concatenate([initial_v, sim_v]), rm["delta"])
    X = tim_parametric._conv(tv, K)[p:p + n]

    if bootstrap:
        shocks = rm["resid"][rng.integers(0, len(rm["resid"]), size=n)]
    else:
        shocks = rng.normal(0.0, np.sqrt(rm["sigma2"]), n)
    sim_r = rm["const"] + rm["sigma_f"] * X + shocks
    return P0 * np.exp(np.concatenate([[0.0], np.cumsum(sim_r[:-1])]))


def generate_prices(price_model, price_params, sim_v, initial_v, P0, rng, bootstrap=True):
    """sim_v: volumi con segno (n_trades), initial_v: p volumi con segno precedenti."""
    if price_model == "tim":
        rm = price_params["return_model"]
        prices, _ = tim.return_simulate(rm, sim_v, initial_v, initial_price=P0,
                                        seed=rng.integers(0, 2**31 - 1),
                                        bootstrap=bootstrap)
        return prices
    if price_model == "tim_parametric":
        return _tim_parametric_prices(price_params["return_model"], sim_v, initial_v,
                                 P0, rng, bootstrap)
    raise ValueError(f"price_model sconosciuto: {price_model!r}")


# ---------------------------------------------------------------------------
# PIPELINE GIORNALIERA
# ---------------------------------------------------------------------------

def simulate_day(prices_real, volumes_real, rng, *,
                 alpha, n_traders, lambda_lmf,
                 volume_model, volume_params,
                 price_model, price_params, bootstrap):
    n_trades = len(prices_real)
    P0 = float(prices_real[0])
    p = int(price_params["p"])
    n_tot = n_trades + p

    signs_all = generate_signs(n_tot, alpha, n_traders, lambda_lmf)
    vols_all = generate_volumes(n_tot, volumes_real, volume_model, volume_params, rng)
    signed_all = signs_all * vols_all

    initial_v, sim_v = signed_all[:p], signed_all[p:]
    prices = generate_prices(price_model, price_params, sim_v, initial_v, P0, rng,
                             bootstrap=bootstrap)
    return prices, vols_all[p:], signs_all[p:]


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run(data_dir=DEFAULT_DATA_DIR,
        out_dir=DEFAULT_OUT_DIR,
        models_dir=MODELS_DIR,
        # segni
        alpha=1.5,
        n_traders=None,
        lambda_lmf=0.3,
        # volumi
        volume_model="empirical",     # 'empirical' | 'mem_acd' | 'log_ar'
        mem_dist="burr12",
        mem_p=1,
        mem_q=1,
        ar_order=1000,                # p del modello logar_<p> salvato
        # prezzi
        price_model="tim",            # 'tim' | 'tim_parametric'
        delta=0.0,                    # delta usato nel fit (nome della cartella del modello)
        p_price=1000,                 # p usato nel fit (nome della cartella del modello)
        bootstrap=True,               # shock dei rendimenti: bootstrap dei residui o gaussiani
        pattern="*.csv",
        seed=42):

    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    volume_params = load_volume_model(volume_model, models_dir, ar_order,
                                      mem_p, mem_q, mem_dist)
    price_params = load_price_model(price_model, models_dir, delta, p_price)

    files = sorted(f for f in data_dir.glob(pattern) if f.is_file())
    if not files:
        raise FileNotFoundError(f"nessun file '{pattern}' in {data_dir}")

    print(f"Simulazione giorno per giorno | segni=LMF | volumi={volume_model} | "
          f"prezzi={price_model} (p={price_params['p']}, delta={price_params['delta']})")

    n_done = 0
    for f in files:
        timestamps, prices_real, volumes_real, signs_real = open_real_data(f.name, data_dir)

        try:
            prices, volumes, signs = simulate_day(
                prices_real, volumes_real, rng,
                alpha=alpha, n_traders=n_traders, lambda_lmf=lambda_lmf,
                volume_model=volume_model, volume_params=volume_params,
                price_model=price_model, price_params=price_params,
                bootstrap=bootstrap)
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
        out_dir=Path("..") / "database" / "data_lmf_1.5_0.3_empirical_tim_param_0.3_1000",
        alpha=1.5,
        n_traders=None,
        lambda_lmf=0.3,
        volume_model="empirical",   # 'empirical' | 'mem_acd' | 'log_ar'
        price_model="tim_parametric",          # 'tim' | 'tim_parametric'
        delta=0.3,
        p_price=1000,
        bootstrap=True,
        seed=42,
    )