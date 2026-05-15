"""
synthetic_market.py
====================
Simulatore di mercato sintetico a tre strati.
L'output e' un CSV per giorno nel formato IDENTICO ai dati reali,
leggibile direttamente da methods.generate() e methods.generate_slim().

PIPELINE
--------
  1. SEGNI   — Long Memory Flow (LMF)
               Pool di N trader con metaordini di durata Pareto(alpha).
               Produce epsilon_t in {-1, +1}.

  2. VOLUMI  — AR(p) calibrato sui log-volumi reali del giorno.
               Produce v_t > 0.

  3. PREZZI  — Transient Impact Model (TIM), calibrato sui dati reali.

               r_t = ln(P_t / P_{t-1})
                   = sum_{l=0}^{L-1} G(l) * epsilon_{t-l} * f(v_{t-l}) + eta_t

               G(l) = (l+1)^{-beta}          kernel power-law
               f(v) = sigma_f * v^delta       square-root impact (delta=0.5)
               eta_t ~ N(0, sigma_eta)
               P_t = P_{t-1} * exp(r_t)      P_t e' il prezzo ASSOLUTO

FORMATO CSV OUTPUT  (senza header, una riga per transazione)
------------------------------------------------------------
  col 0 : timestamp_sec      secondi dall'inizio della sessione
  col 1 : prezzo             P_t assoluto dopo il trade t  (sempre > 0)
  col 2 : volume_in_modulo   |v_t|
  col 3 : segno_volume       epsilon_t in {-1, +1}

I timestamp sono equispaziati in [36000, 55797] (10:00-15:29).

CALIBRAZIONE TIM
----------------
  sigma_f e sigma_eta vengono stimati dai dati reali via OLS:

  Costruiamo il segnale di impatto convolvendo i segni reali con il kernel:
      X_t = sum_{l=0}^{L-1} G(l) * epsilon_{t-l} * v_{t-l}^delta

  Poi fittiamo:
      r_t = sigma_f * X_t + eta_t   (OLS, 1 solo coefficiente)

  sigma_f  = coefficiente OLS
  sigma_eta = std dei residui

  Il fit viene fatto sull'intero dataset (tutti i giorni concatenati)
  prima di avviare la simulazione, in modo da avere stime stabili.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from os import listdir


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def open_real_data(fname, data_dir=r"..\database\data"):
    """
    Legge un file reale:
        col0=timestamp_sec, col1=prezzo, col2=volume_modulo, col3=segno
    """
    trades  = pd.read_csv(Path(data_dir) / fname, header=None)
    prices  = np.array(trades[1], dtype=float)
    volumes = np.array(trades[2], dtype=float)
    signs   = np.array(trades[3], dtype=float)
    return prices, volumes, signs


def save_simulated_data(out_path, prices, volumes, signs, n_trades):
    """
    Salva nel formato identico ai dati reali:
        timestamp_sec, prezzo, volume_in_modulo, segno_volume

    I timestamp sono equispaziati in [36000, 55797].
    Il prezzo e' il prezzo ASSOLUTO post-trade (non il log-prezzo).
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
# LAYER 1 — SEGNI  (Long Memory Flow)
# ---------------------------------------------------------------------------

def simulate_lmf(alpha, n_traders, total_steps, rng):
    """
    Serie binaria epsilon_t in {-1, +1} a memoria lunga.
    ACF dei segni ~ tau^{-(alpha-1)}  per  1 < alpha < 2.
    """
    state = np.zeros((n_traders, 2), dtype=np.int64)  # [side, remaining]

    def _new():
        return rng.choice([1, -1]), int(rng.pareto(alpha) + 1)

    for i in range(n_traders):
        state[i, 0], state[i, 1] = _new()

    signs = np.empty(total_steps, dtype=np.int8)

    for t in range(total_steps):
        idx             = rng.integers(0, n_traders)
        side, remaining = state[idx]
        signs[t]        = side
        remaining      -= 1
        if remaining <= 0:
            state[idx, 0], state[idx, 1] = _new()
        else:
            state[idx, 1] = remaining

    return signs


# ---------------------------------------------------------------------------
# LAYER 2 — VOLUMI  (AR(p) sui log-volumi reali)
# ---------------------------------------------------------------------------

def fit_ar_log_volume(volumes, p):
    """
    Calibra AR(p) sui log-volumi reali via OLS senza intercetta:
        log(v_t) = phi @ [log(v_{t-1}), ..., log(v_{t-p})] + eps_t
    """
    log_v = np.log(np.maximum(volumes, 1e-12))
    n     = len(log_v)
    X     = np.column_stack([log_v[p - i : n - i] for i in range(1, p + 1)])
    y     = log_v[p:]
    phi, *_ = np.linalg.lstsq(X, y, rcond=None)
    sigma   = float(np.std(y - X @ phi))
    return {'p': p, 'phi': phi, 'sigma': sigma, 'log_vol_seed': log_v[:p]}


def simulate_ar_log_volume(params, n_steps, rng):
    """
    Genera n_steps volumi > 0 tramite AR(p) in spazio log.
    """
    p, phi, sigma = params['p'], params['phi'], params['sigma']
    buf     = np.empty(p + n_steps)
    buf[:p] = params['log_vol_seed']
    noise   = rng.normal(0.0, sigma, n_steps)
    for t in range(n_steps):
        buf[p + t] = buf[t : t + p][::-1] @ phi + noise[t]
    return np.exp(buf[p:])


# ---------------------------------------------------------------------------
# LAYER 3 — PREZZI  (Transient Impact Model)
# ---------------------------------------------------------------------------

def _build_impact_signal(signs, volumes, beta, delta, kernel_L):
    """
    Calcola il segnale di impatto convolvendo segni e volumi col kernel:
        X_t = sum_{l=0}^{L-1} G(l) * epsilon_{t-l} * v_{t-l}^delta

    Usato sia per la calibrazione che per la simulazione.
    """
    T      = len(signs)
    kernel = (np.arange(kernel_L, dtype=float) + 1.0) ** (-beta)
    raw    = signs.astype(float) * (volumes ** delta)
    return np.convolve(raw, kernel, mode='full')[:T]


def calibrate_tim(data_dir, beta, delta, kernel_L):
    """
    Stima sigma_f e sigma_eta dai dati reali usando tutti i file in data_dir.

    Modello:
        r_t = sigma_f * X_t + eta_t
        X_t = sum_{l=0}^{L-1} (l+1)^{-beta} * epsilon_{t-l} * v_{t-l}^delta

    Procedura:
        1. Per ogni giorno, calcola i log-rendimenti reali r_t e il segnale X_t
        2. Concatena tutto e fitta sigma_f via OLS (un solo coefficiente, no intercetta)
        3. sigma_eta = std dei residui

    Parameters
    ----------
    data_dir : cartella dei file reali
    beta     : esponente kernel (deve coincidere con quello usato in simulazione)
    delta    : esponente impact function
    kernel_L : troncamento del kernel

    Returns
    -------
    sigma_f   : float — scala dell'impatto stimata
    sigma_eta : float — std del rumore idiosincratico stimata
    """
    print("Calibrazione TIM sui dati reali...")
    all_r = []
    all_X = []

    for fname in sorted(listdir(data_dir)):
        prices, volumes, signs = open_real_data(fname, data_dir)

        if len(prices) < 2:
            continue

        # Log-rendimenti reali: r_t = log(P_t / P_{t-1})
        log_r = np.diff(np.log(np.maximum(prices, 1e-12)))

        # Segnale di impatto sui trade t=0..T-2 (allineato con log_r)
        X = _build_impact_signal(signs[:-1], volumes[:-1], beta, delta,
                                 min(kernel_L, len(signs) - 1))

        all_r.append(log_r)
        all_X.append(X)

    r_all = np.concatenate(all_r)
    X_all = np.concatenate(all_X)

    # OLS: r = sigma_f * X  (senza intercetta)
    sigma_f   = float(np.dot(X_all, r_all) / np.dot(X_all, X_all))
    resid     = r_all - sigma_f * X_all
    sigma_eta = float(np.std(resid))

    print(f"  sigma_f   = {sigma_f:.6e}")
    print(f"  sigma_eta = {sigma_eta:.6e}")
    return sigma_f, sigma_eta


def simulate_tim(signs, volumes, beta, delta, sigma_f, sigma_eta,
                 kernel_L, P0, rng):
    """
    Genera i prezzi PRE-TRADE P_t tramite il Transient Impact Model.
    P_t e' il prezzo un momento PRIMA che la transazione t venga eseguita.

    Semantica:
        r_t = log(P_{t+1} / P_t) = effetto del trade t sul prezzo
            = sigma_f * X_t + eta_t
        X_t = sum_{l=0}^{L-1} G(l) * epsilon_{t-l} * v_{t-l}^delta

    Quindi:
        prices[0] = P0                                (prima del trade 0)
        prices[t] = P0 * exp(sum_{s=0}^{t-1} r_s)    (prima del trade t)

    Coerenza con calibrate_tim:
        La calibrazione fitta r_t = log(P[t+1]/P[t]) ~ sigma_f * X_t
        dove X_t e' costruito con lag-0 = trade t (stessa _build_impact_signal).
        Qui usiamo la stessa X_t per r_t, poi shiftiamo di 1:
            prices[t+1] = prices[t] * exp(r_t)
    """
    T = len(signs)

    # r_t: log-rendimento causato dal trade t  (lag-0 = trade t, coerente con calibrazione)
    X       = _build_impact_signal(signs, volumes, beta, delta, kernel_L)
    log_ret = sigma_f * X + rng.normal(0.0, sigma_eta, T)   # r_0, ..., r_{T-1}

    # prices[t] = prezzo PRIMA del trade t
    #   prices[0] = P0
    #   prices[1] = P0 * exp(r_0)
    #   prices[t] = P0 * exp(r_0 + ... + r_{t-1})
    prices    = np.empty(T, dtype=float)
    prices[0] = P0
    if T > 1:
        prices[1:] = P0 * np.exp(np.cumsum(log_ret[:-1]))

    return prices


# ---------------------------------------------------------------------------
# PIPELINE GIORNALIERA
# ---------------------------------------------------------------------------

def simulate_day(n_trades, ar_params, alpha, n_traders, beta, delta,
                 sigma_f, sigma_eta, kernel_L, P0, rng):
    """
    Simula un singolo giorno con n_trades transazioni.
    Returns: prices, volumes, signs
    """
    signs   = simulate_lmf(alpha, n_traders, n_trades, rng)
    volumes = simulate_ar_log_volume(ar_params, n_trades, rng)
    prices  = simulate_tim(signs, volumes, beta=beta, delta=delta,
                           sigma_f=sigma_f, sigma_eta=sigma_eta,
                           kernel_L=min(kernel_L, n_trades),
                           P0=P0, rng=rng)
    return prices, volumes, signs


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run(data_dir  = r"..\database\data",
        out_dir   = r"..\database\data_synthetic",
        alpha     = 1.5,
        n_traders = 10,
        ar_order  = 1000,
        beta      = 0.3,
        delta     = 0.5,
        kernel_L  = 500,
        seed      = 42):
    """
    Per ogni file reale in data_dir:
      - calibra AR(p) sui volumi reali del giorno
      - usa P0 = primo prezzo reale (prezzo assoluto, non log)
      - simula lo stesso numero di transazioni del giorno reale
      - salva il CSV con timestamp equispaziati in [36000, 55797]

    sigma_f e sigma_eta vengono stimati automaticamente dai dati reali
    tramite OLS sul Transient Impact Model prima di avviare la simulazione.

    Parametri
    ---------
    alpha     : esponente Pareto LMF  (1 < alpha < 2)
    n_traders : pool di trader LMF
    ar_order  : ordine AR log-volume (ridotto auto se dati insufficienti)
    beta      : esponente decadimento kernel TIM
    delta     : esponente impact function (0.5 = square-root)
    kernel_L  : troncamento memoria kernel (in numero di trade)
    seed      : seme per riproducibilita'
    """
    rng      = np.random.default_rng(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Calibrazione automatica di sigma_f e sigma_eta ---
    sigma_f, sigma_eta = calibrate_tim(data_dir, beta=beta, delta=delta,
                                       kernel_L=kernel_L)

    paths = sorted(listdir(data_dir))
    print(f"\nFile trovati: {len(paths)}\n")

    for fname in paths:
        prices_real, volumes_real, _ = open_real_data(fname, data_dir)
        P0       = float(prices_real[0])   # prezzo assoluto, NON log-prezzo
        n_trades = len(prices_real)

        print(f"-> {fname}  |  n_trades={n_trades}  |  P0={P0:.4f}")

        p_eff = ar_order if len(volumes_real) > ar_order else max(1, len(volumes_real) // 10)
        if p_eff < ar_order:
            print(f"   WARNING: dati reali insufficienti, AR order {ar_order} -> {p_eff}")

        ar_params = fit_ar_log_volume(volumes_real, p=p_eff)

        prices, volumes, signs = simulate_day(
            n_trades  = n_trades,
            ar_params = ar_params,
            alpha     = alpha,
            n_traders = n_traders,
            beta      = beta,
            delta     = delta,
            sigma_f   = sigma_f,
            sigma_eta = sigma_eta,
            kernel_L  = kernel_L,
            P0        = P0,
            rng       = rng,
        )

        save_simulated_data(
            out_path = str(out_path / fname),
            prices   = prices,
            volumes  = volumes,
            signs    = signs,
            n_trades = n_trades,
        )
        print(f"   OK -> {out_path / fname}")

    print("\nSimulazione completata.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run(
        data_dir  = r"..\database\data",
        out_dir   = r"..\database\data_lmf_tim_lin",

        # LMF
        alpha     = 1.5,    # esponente Pareto (1 < alpha < 2 -> memoria lunga)
        n_traders = 50,     # pool di trader

        # AR log-volumi
        ar_order  = 1000,   # ridotto auto se dati reali insufficienti

        # TIM
        beta      = 0.25,    # esponente decadimento kernel power-law
        delta     = 1.0,    # esponente impact (0.5 = square-root)
        # sigma_f e sigma_eta: calibrati automaticamente dai dati reali

        kernel_L  = 500,    # memoria massima kernel (in trade)
        seed      = 42,
    )