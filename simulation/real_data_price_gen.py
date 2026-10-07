"""
real_data_price_gen.py
======================
Generatore di prezzi sintetici basato ESCLUSIVAMENTE su SEGNI E VOLUMI REALI
letti sequenzialmente (nessun campionamento o simulazione di volumi/segni).
Timestamp, volumi e segni dell'output sono quelli reali; cambiano solo i prezzi.

Modelli di generazione del prezzo:
  1. 'regression' : regressione OLS dei rendimenti logaritmici su p+1 lag di sign*|v|^delta
                    (delta=1: volumi firmati; costante inclusa).
  2. 'tim'        : Transient Impact Model. Il propagatore G(l) = (l+1)^(-beta)
                    agisce sul LIVELLO del log-prezzo; il regressore dei
                    rendimenti e' quindi il kernel differenziato
                    K(l) = G(l) - G(l-1) (con G(-1) = 0 e G(L) = 0, troncato).

Convenzione di allineamento (parametro `impact_lag`)
----------------------------------------------------
  impact_lag = 0 : prices[t] e' il prezzo di esecuzione del trade t, quindi
                   l'impatto del trade t compare in r_t = log P_t - log P_{t-1}.
  impact_lag = 1 : prices[t] e' un prezzo (mid) PRECEDENTE al trade t, quindi
                   il trade t compare in r_t = log P_{t+1} - log P_t.
In entrambi i casi calibrazione e simulazione usano lo stesso allineamento.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path("..") / "database" / "data"
DEFAULT_OUT_DIR = Path("..") / "database" / "data_real_reg_sqrt"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def open_real_data(fname, data_dir=DEFAULT_DATA_DIR):
    """Legge un file giornaliero (senza header): colonne 0=timestamp, 1=prezzo,
    2=volume, 3=segno."""
    trades = pd.read_csv(Path(data_dir) / fname, header=None)
    timestamps = trades[0].to_numpy(dtype=float)
    prices = trades[1].to_numpy(dtype=float)
    volumes = trades[2].to_numpy(dtype=float)
    signs = trades[3].to_numpy(dtype=float)
    return timestamps, prices, volumes, signs


def save_simulated_data(out_path, timestamps, prices, volumes, signs):
    """Scrive il file di output mantenendo timestamp, volumi e segni reali."""
    fp = Path(out_path)
    if fp.exists():
        fp.unlink()

    pd.DataFrame({
        0: timestamps,
        1: prices,
        2: np.abs(volumes),
        3: signs.astype(int),
    }).to_csv(fp, index=False, header=False)


# ---------------------------------------------------------------------------
# Utilità comuni
# ---------------------------------------------------------------------------

def _validate_inputs(prices, volumes, signs):
    if not (len(prices) == len(volumes) == len(signs)):
        raise ValueError("prezzi, volumi e segni hanno lunghezze diverse")
    if len(prices) < 3:
        raise ValueError("serie troppo corta")
    if not (np.all(np.isfinite(prices)) and np.all(np.isfinite(volumes))
            and np.all(np.isfinite(signs))):
        raise ValueError("valori non finiti in prezzi/volumi/segni")
    if np.any(prices <= 0):
        raise ValueError("prezzi non positivi: impossibile calcolare i log-rendimenti")
    if np.any(signs == 0):
        warnings.warn("segni pari a 0: quei trade non producono impatto")


def _log_returns(prices):
    """r[j] = log P[j+1] - log P[j], lunghezza N-1."""
    return np.diff(np.log(prices))


def _align(x, impact_lag):
    """Allinea una serie per-trade (lunghezza N) ai rendimenti (lunghezza N-1).
    impact_lag=0: r[j] e' guidato dal trade j+1 -> x[1:]
    impact_lag=1: r[j] e' guidato dal trade j   -> x[:-1]"""
    if impact_lag not in (0, 1):
        raise ValueError("impact_lag deve essere 0 oppure 1")
    N = len(x)
    return x[1 - impact_lag: N - impact_lag]


def _signed_impact(signs, volumes, delta):
    """Segnale per-trade sign_s * |v_s|^delta (delta=1 -> volume firmato)."""
    return signs.astype(float) * np.abs(volumes) ** delta


def _ols_fit(X, y):
    """OLS con stima di sigma corretta per i gradi di libertà."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - X.shape[1]
    if dof <= 0:
        raise ValueError("troppo pochi dati per il numero di parametri stimati")
    sigma = float(np.sqrt(resid @ resid / dof))
    return coef, sigma


def _rebuild_prices(P0, sim_r, tick_size=None):
    """Ricostruisce i prezzi da log-rendimenti simulati (len N-1 -> len N)."""
    prices = np.empty(len(sim_r) + 1, dtype=float)
    prices[0] = P0
    prices[1:] = P0 * np.exp(np.cumsum(sim_r))
    if tick_size:
        prices = np.round(prices / tick_size) * tick_size
    return prices


# ---------------------------------------------------------------------------
# MODELLO 1 — REGRESSIONE OLS (p+1 coefficienti liberi + costante)
# ---------------------------------------------------------------------------

def fit_ols(prices_real, volumes_real, signs_real, p, delta=1.0, impact_lag=1):
    """r_t = c + sum_{k=0..p} theta_k * (s |v|^delta)_{t-k} + eps_t.
    delta=1 -> regressori = volumi firmati. Ritorna (c, theta[0..p], sigma)."""
    _validate_inputs(prices_real, volumes_real, signs_real)

    y_all = _log_returns(prices_real)
    d = _align(_signed_impact(signs_real, volumes_real, delta), impact_lag)
    n = len(y_all)

    n_obs = n - p
    if n_obs <= p + 2:
        raise ValueError(f"serie troppo corta per p={p}: {n} rendimenti")
    if n_obs < 10 * (p + 2):
        warnings.warn(f"p={p} è alto rispetto a {n_obs} osservazioni: rischio di overfitting")

    # Colonne: [1, d_t, d_{t-1}, ..., d_{t-p}] per t = p..n-1
    X = np.column_stack([np.ones(n_obs)] + [d[p - k: n - k] for k in range(p + 1)])
    coef, sigma = _ols_fit(X, y_all[p:])
    return float(coef[0]), coef[1:], sigma


def simulate_ols(P0, volumes_real, signs_real, const, theta, sigma, rng,
                 delta=1.0, impact_lag=1, tick_size=None):
    N = len(volumes_real)
    d = _align(_signed_impact(signs_real, volumes_real, delta), impact_lag)
    # np.convolve ribalta già il kernel: theta[k] moltiplica d[t-k].
    # Nei primi p passi 'full' usa somme parziali (transitorio corretto).
    conv = np.convolve(d, theta, mode="full")[:N - 1]
    sim_r = const + conv + rng.normal(0.0, sigma, size=N - 1)
    return _rebuild_prices(P0, sim_r, tick_size)


def calibrate_and_simulate_ols(prices_real, volumes_real, signs_real, P0, p, rng,
                               delta=1.0, impact_lag=1, tick_size=None):
    const, theta, sigma = fit_ols(prices_real, volumes_real, signs_real, p,
                                  delta=delta, impact_lag=impact_lag)
    return simulate_ols(P0, volumes_real, signs_real, const, theta, sigma, rng,
                        delta=delta, impact_lag=impact_lag, tick_size=tick_size)


# ---------------------------------------------------------------------------
# MODELLO 2 — TRANSIENT IMPACT MODEL
# ---------------------------------------------------------------------------

def _return_kernel(beta, kernel_L):
    """Kernel dei rendimenti K = diff del propagatore G(l) = (l+1)^(-beta),
    l = 0..L-1, troncato (G = 0 per l >= L). Lunghezza L+1."""
    G = (np.arange(kernel_L, dtype=float) + 1.0) ** (-beta)
    return np.diff(np.concatenate(([0.0], G, [0.0])))


def _build_impact_signal(signs, volumes, beta, delta, kernel_L, impact_lag):
    """Regressore dei rendimenti X_t = sum_s K(t-s) * sign_s * |v_s|^delta."""
    d = _align(_signed_impact(signs, volumes, delta), impact_lag)
    K = _return_kernel(beta, kernel_L)
    return np.convolve(d, K, mode="full")[:len(d)]


def fit_tim(prices_real, volumes_real, signs_real, beta, delta, kernel_L, impact_lag=1):
    """r_t = c + sigma_f * X_t + eta_t. Ritorna (c, sigma_f, sigma_eta)."""
    _validate_inputs(prices_real, volumes_real, signs_real)
    y = _log_returns(prices_real)
    X = _build_impact_signal(signs_real, volumes_real, beta, delta, kernel_L, impact_lag)
    coef, sigma_eta = _ols_fit(np.column_stack([np.ones(len(y)), X]), y)
    return float(coef[0]), float(coef[1]), sigma_eta


def simulate_tim(P0, volumes_real, signs_real, const, sigma_f, sigma_eta, beta, delta,
                 kernel_L, rng, impact_lag=1, tick_size=None):
    X = _build_impact_signal(signs_real, volumes_real, beta, delta, kernel_L, impact_lag)
    sim_r = const + sigma_f * X + rng.normal(0.0, sigma_eta, size=len(X))
    return _rebuild_prices(P0, sim_r, tick_size)


def calibrate_and_simulate_tim(prices_real, volumes_real, signs_real, P0, beta, delta,
                               kernel_L, rng, impact_lag=1, tick_size=None):
    const, sigma_f, sigma_eta = fit_tim(prices_real, volumes_real, signs_real,
                                        beta, delta, kernel_L, impact_lag)
    return simulate_tim(P0, volumes_real, signs_real, const, sigma_f, sigma_eta,
                        beta, delta, kernel_L, rng, impact_lag, tick_size)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run(data_dir=DEFAULT_DATA_DIR,
        out_dir=DEFAULT_OUT_DIR,
        price_model="regression",   # 'regression' | 'tim'
        vol_lags_for_ret=1000,      # p per OLS
        beta=0.25,                  # solo TIM
        delta=1.0,                  # esponente sul volume: sign*|v|^delta (OLS e TIM)
        kernel_L=500,               # solo TIM
        impact_lag=1,               # 1: prezzo PRE-trade (default); 0: prezzo di esecuzione
        tick_size=None,             # es. 0.01 per riportare i prezzi su griglia di tick
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

    print(f"Inizio generazione prezzi da segni e volumi reali [Modello: {price_model}]...")

    n_done = 0
    for f in files:
        timestamps, prices_real, volumes_real, signs_real = open_real_data(f.name, data_dir)
        P0 = float(prices_real[0])

        try:
            if price_model == "regression":
                prices_sim = calibrate_and_simulate_ols(
                    prices_real, volumes_real, signs_real, P0=P0, p=vol_lags_for_ret,
                    rng=rng, delta=delta, impact_lag=impact_lag, tick_size=tick_size)
            else:
                prices_sim = calibrate_and_simulate_tim(
                    prices_real, volumes_real, signs_real, P0=P0, beta=beta, delta=delta,
                    kernel_L=kernel_L, rng=rng, impact_lag=impact_lag, tick_size=tick_size)
        except ValueError as e:
            # Nessun fallback ai prezzi reali: il giorno viene saltato.
            print(f"  !! Giorno {f.name} saltato: {e}")
            continue

        save_simulated_data(out_dir / f.name, timestamps, prices_sim, volumes_real, signs_real)
        n_done += 1
        print(f"  -> Giorno {f.name} completato.")

    print(f"\nElaborazione completata: {n_done}/{len(files)} giorni generati.")


if __name__ == "__main__":
    run(
        data_dir=DEFAULT_DATA_DIR,
        out_dir=DEFAULT_OUT_DIR,
        price_model="regression",          # 'regression' | 'tim'
        vol_lags_for_ret=1000,
        beta=0.25,
        delta=0.5,
        kernel_L=500,
        impact_lag=1,
        tick_size=None,
        seed=42,
    )