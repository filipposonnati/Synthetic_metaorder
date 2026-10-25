"""
empirical_signed_market.py
==========================
Simulatore giorno per giorno:

  1. VOLUMI CON SEGNO : campionati (con reinserimento, i.i.d.) dalla distribuzione
                        empirica dei volumi con segno (volume * segno) del giorno reale.
  2. PREZZI           : modello gia' calibrato, scelto con `price_model`:
                          'tim'            -> tim.py            (models/tim_<delta>_<p>)
                          'tim_parametric' -> tim_parametric.py (models/tim_param_<delta>_<p>)

I modelli NON vengono ricalibrati: si caricano i model.npz prodotti dagli script di fit.
I timestamp in uscita sono quelli reali; volumi, segni e prezzi sono sintetici.
Per dare ai kernel di impatto la storia dei p volumi precedenti si campionano
n_trades + p volumi con segno: i primi p fanno da initial_v e vengono scartati dall'output.
"""

from pathlib import Path

import numpy as np

from lmf_synthetic_market import (
    DEFAULT_DATA_DIR, MODELS_DIR,
    open_real_data, save_simulated_data,
    load_price_model, generate_prices,
)


def sample_signed_volumes(signed_real, n_steps, rng):
    """Campiona n_steps volumi con segno dalla distribuzione empirica (con reinserimento)."""
    signed_real = np.asarray(signed_real, dtype=np.float64).ravel()
    if len(signed_real) == 0:
        raise ValueError("giorno senza trade: impossibile campionare i volumi")
    return rng.choice(signed_real, size=n_steps, replace=True)


def sample_gaussian_signed_volumes(signed_real, n_steps, rng):
    """Campiona n_steps volumi con segno da N(mu, sigma^2), con mu e sigma^2 pari
    a media e varianza dei volumi con segno reali del giorno."""
    signed_real = np.asarray(signed_real, dtype=np.float64).ravel()
    if len(signed_real) < 2:
        raise ValueError("servono almeno 2 trade per stimare media e varianza")
    return rng.normal(signed_real.mean(), signed_real.std(), size=n_steps)


def sample_volumes(signed_real, n_steps, rng, volume_sampling):
    if volume_sampling == "empirical":
        return sample_signed_volumes(signed_real, n_steps, rng)
    if volume_sampling == "gaussian":
        return sample_gaussian_signed_volumes(signed_real, n_steps, rng)
    raise ValueError(f"volume_sampling sconosciuto: {volume_sampling!r}")


def simulate_day(prices_real, volumes_real, signs_real, rng, *,
                 volume_sampling, price_model, price_params, bootstrap):
    n_trades = len(prices_real)
    P0 = float(prices_real[0])
    p = int(price_params["p"])

    signed_real = np.abs(volumes_real) * signs_real
    signed_all = sample_volumes(signed_real, n_trades + p, rng, volume_sampling)

    initial_v, sim_v = signed_all[:p], signed_all[p:]
    prices = generate_prices(price_model, price_params, sim_v, initial_v, P0, rng,
                             bootstrap=bootstrap)
    return prices, np.abs(sim_v), np.sign(sim_v)


def run(data_dir=DEFAULT_DATA_DIR,
        out_dir=None,
        models_dir=MODELS_DIR,
        volume_sampling="empirical",  # 'empirical' | 'gaussian'
        price_model="tim",            # 'tim' | 'tim_parametric'
        delta=0.0,                    # delta usato nel fit (nome della cartella del modello)
        p_price=1000,                 # p usato nel fit (nome della cartella del modello)
        bootstrap=True,               # shock dei rendimenti: bootstrap dei residui o gaussiani
        pattern="*.csv",
        seed=42):

    if out_dir is None:
        out_dir = Path("..") / "database" / f"data_iid_{volume_sampling}_{price_model}_{delta}_{p_price}"

    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    price_params = load_price_model(price_model, models_dir, delta, p_price)

    files = sorted(f for f in data_dir.glob(pattern) if f.is_file())
    if not files:
        raise FileNotFoundError(f"nessun file '{pattern}' in {data_dir}")

    print(f"Simulazione giorno per giorno | volumi={volume_sampling} con segno | "
          f"prezzi={price_model} (p={price_params['p']}, delta={price_params['delta']})")

    n_done = 0
    for f in files:
        timestamps, prices_real, volumes_real, signs_real = open_real_data(f.name, data_dir)

        try:
            prices, volumes, signs = simulate_day(
                prices_real, volumes_real, signs_real, rng,
                volume_sampling=volume_sampling, price_model=price_model, price_params=price_params,
                bootstrap=bootstrap)
        except ValueError as e:
            print(f"  !! Giorno {f.name} saltato: {e}")
            continue

        save_simulated_data(out_dir / f.name, timestamps, prices, volumes, signs)
        n_done += 1
        print(f"  -> Giorno {f.name} completato.")

    print(f"\nSimulazione completata: {n_done}/{len(files)} giorni generati.")


if __name__ == "__main__":
    volume_sampling = "gaussian"      # 'empirical' | 'gaussian'
    price_model = "tim"    # 'tim' | 'tim_parametric'
    delta = 0.0
    p_price = 1000

    run(
        volume_sampling=volume_sampling,
        price_model=price_model,
        delta=delta,
        p_price=p_price,
        bootstrap=True,
        seed=42,
    )