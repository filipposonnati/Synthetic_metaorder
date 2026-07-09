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

  2. VOLUMI  — Modello del volume, due alternative selezionabili
               (parametro volume_model):
                 - 'mem_acd' (default): Multiplicative Error Model /
                   ACD(p,q) alla Engle & Russell, v_t = mu_t * z_t con
                   z_t > 0 i.i.d. media 1 (Gamma) e mu_t dinamica
                   GARCH-like. Positivo per costruzione, persistenza
                   sulla MEDIA condizionata (non sul log-livello).
                 - 'log_ar' (legacy): AR(p) sui log-volumi.
               Entrambi producono v_t > 0.

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
from scipy.optimize import minimize


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

def _ar_max_pole_modulus(phi):
    """
    Calcola il modulo massimo delle radici del polinomio caratteristico
    dell'AR(p), ossia il modulo massimo degli autovalori della companion
    matrix. Se >= 1 il processo NON e' stazionario (esplode).
    """
    p = len(phi)
    if p == 1:
        return float(np.abs(phi[0]))
    companion = np.zeros((p, p))
    companion[0, :] = phi
    companion[1:, :-1] = np.eye(p - 1)
    eigvals = np.linalg.eigvals(companion)
    return float(np.max(np.abs(eigvals)))


def _fit_ar_yule_walker(log_v, p):
    """
    Stima AR(p) tramite equazioni di Yule-Walker (Levinson-Durbin).
    A differenza dell'OLS diretto, questa stima e' GARANTITA stazionaria
    (i poli hanno sempre modulo < 1), perche' usa solo l'autocovarianza
    campionaria, che e' una sequenza definita non-negativa per costruzione.
    """
    x  = log_v - log_v.mean()
    n  = len(x)
    acov = np.array([np.dot(x[:n - k], x[k:]) / n for k in range(p + 1)])

    # Levinson-Durbin recursion
    phi = np.zeros(p)
    prev_phi = np.zeros(p)
    err = acov[0]
    for k in range(p):
        acc = acov[k + 1] - np.dot(prev_phi[:k], acov[k:0:-1])
        reflection = acc / err if err > 1e-300 else 0.0
        new_phi = np.zeros(p)
        new_phi[k] = reflection
        if k > 0:
            new_phi[:k] = prev_phi[:k] - reflection * prev_phi[k - 1::-1]
        phi = new_phi
        err *= (1.0 - reflection ** 2)
        prev_phi = phi.copy()

    sigma = float(np.sqrt(max(err, 1e-300)))
    return phi, sigma


def fit_ar_log_volume(volumes, p, force_stationary=True, pole_threshold=0.98):
    """
    Calibra AR(p) sui log-volumi reali:
        log(v_t) = phi @ [log(v_{t-1}), ..., log(v_{t-p})] + eps_t

    Di default il fit e' fatto via OLS senza intercetta. Con p grande
    (es. 1000) stimato su un solo giorno di dati, l'OLS puo' produrre
    coefficienti instabili (poli del polinomio caratteristico fuori dal
    cerchio unitario): il volume simulato esplode esponenzialmente e
    inietta rumore enorme nel prezzo (via f(v)=v^delta).

    Se force_stationary=True (default), dopo il fit OLS controlliamo il
    modulo massimo dei poli. Se >= pole_threshold, rifittiamo con
    Yule-Walker/Levinson-Durbin, che e' sempre stazionario per
    costruzione.

    Parameters
    ----------
    force_stationary : bool  — attiva il check + fallback stazionario
    pole_threshold    : float — soglia sul modulo massimo dei poli
                        (< 1.0, tipicamente 0.95-0.99) oltre la quale
                        si considera l'OLS instabile
    """
    log_v = np.log(np.maximum(volumes, 1e-12))
    n     = len(log_v)
    X     = np.column_stack([log_v[p - i : n - i] for i in range(1, p + 1)])
    y     = log_v[p:]
    phi, *_ = np.linalg.lstsq(X, y, rcond=None)

    method = 'ols'
    max_pole = _ar_max_pole_modulus(phi)

    if force_stationary and max_pole >= pole_threshold:
        print(f"   WARNING: AR({p}) OLS instabile (max |polo| = {max_pole:.4f})"
              f" -> fallback a Yule-Walker (stazionario garantito)")
        phi, _ = _fit_ar_yule_walker(log_v, p)
        method = 'yule_walker'
        max_pole = _ar_max_pole_modulus(phi)
        print(f"            Yule-Walker: max |polo| = {max_pole:.4f}")

    sigma = float(np.std(y - X @ phi))
    return {
        'p': p, 'phi': phi, 'sigma': sigma, 'log_vol_seed': log_v[:p],
        'method': method, 'max_pole_modulus': max_pole,
    }


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
# LAYER 2bis — VOLUMI  (Multiplicative Error Model / ACD(p,q))
# ---------------------------------------------------------------------------
#
# Alternativa al log-AR: v_t = mu_t * z_t
#
#   mu_t = omega + sum_{i=1}^p alpha_i * v_{t-i} + sum_{j=1}^q beta_j * mu_{t-j}
#   z_t ~ i.i.d. > 0, E[z_t] = 1   (qui: Gamma(shape=k, scale=1/k))
#
# Positivo per costruzione (mu_t > 0 per vincoli sui parametri, z_t > 0
# perche' Gamma). A differenza del log-AR, la persistenza (alpha+beta)
# agisce sulla MEDIA condizionata del volume, non sul suo log — e' il
# modello standard in microstruttura per volumi/durate (Engle & Russell
# 1998, "Autoregressive Conditional Duration").
#
# Stima: Quasi-Maximum-Likelihood assumendo z_t esponenziale (garantisce
# consistenza degli stimatori di omega/alpha/beta anche se la vera
# distribuzione non e' esponenziale, purche' mu_t sia specificato
# correttamente). Il parametro di forma k della Gamma usata poi in
# SIMULAZIONE viene stimato a parte sui residui via metodo dei momenti.
# ---------------------------------------------------------------------------

def _mem_build_mu(v, omega, alpha, beta, m):
    """
    Ricostruisce la media condizionata mu_t sull'intera serie osservata v,
    dato un set di parametri. Usato sia in stima (in-sample) che per
    diagnostica.
    """
    T  = len(v)
    p  = len(alpha)
    q  = len(beta)
    mu = np.empty(T)
    mu[:m] = v[:m].mean() if m > 0 else v.mean()
    for t in range(m, T):
        ar_term = 0.0
        for i in range(p):
            ar_term += alpha[i] * v[t - 1 - i]
        ma_term = 0.0
        for j in range(q):
            ma_term += beta[j] * mu[t - 1 - j]
        mu[t] = omega + ar_term + ma_term
    return mu


def _mem_negloglik(theta, v, p, q, m):
    """
    Negative log-likelihood QML assumendo z_t ~ Exponential(mean=1):
        f(v_t | mu_t) = (1/mu_t) * exp(-v_t/mu_t)
        NLL = sum_t [ log(mu_t) + v_t/mu_t ]

    Ritorna una penalita' grande (ma finita, per non rompere l'ottimizzatore)
    se i parametri violano positivita' o stazionarieta' (alpha+beta < 1).
    """
    omega = theta[0]
    alpha = theta[1:1 + p]
    beta  = theta[1 + p:1 + p + q]

    if omega <= 1e-12 or np.any(alpha < 0.0) or np.any(beta < 0.0):
        return 1e10

    persistence = float(alpha.sum() + beta.sum())
    if persistence >= 0.999:
        return 1e10 * (1.0 + persistence)

    try:
        with np.errstate(over='raise', invalid='raise'):
            mu = _mem_build_mu(v, omega, alpha, beta, m)
    except FloatingPointError:
        return 1e10

    mu_eff = mu[m:]
    v_eff  = v[m:]
    if np.any(mu_eff <= 0.0) or not np.all(np.isfinite(mu_eff)):
        return 1e10

    nll = float(np.sum(np.log(mu_eff) + v_eff / mu_eff))
    return nll if np.isfinite(nll) else 1e10


def fit_mem_acd(volumes, p=1, q=1):
    """
    Calibra un MEM/ACD(p,q) sui volumi reali (spazio livelli, non log).

        mu_t = omega + sum_i alpha_i * v_{t-i} + sum_j beta_j * mu_{t-j}
        v_t  = mu_t * z_t ,   z_t ~ Gamma(shape=k, scale=1/k)   (media 1)

    Stima omega, alpha, beta via QML-esponenziale (Nelder-Mead, robusto
    a un profilo di verosimiglianza non liscio per via dei vincoli).
    Il parametro di forma k della Gamma (usato solo in simulazione, per
    generare z_t con la giusta dispersione) e' stimato a posteriori sui
    residui standardizzati z_hat_t = v_t / mu_t via metodo dei momenti:
        Var(z) = 1/k  =>  k_hat = 1 / Var(z_hat)

    Returns
    -------
    dict con: p, q, omega, alpha, beta, k_shape, persistence,
              mu_seed, v_seed, converged, nll
    """
    v = np.maximum(np.asarray(volumes, dtype=float), 1e-12)
    T = len(v)
    m = max(p, q)
    if T <= m + 5:
        raise ValueError(f"Serie troppo corta per MEM/ACD({p},{q}): T={T}")

    v_mean = float(v.mean())
    theta0 = np.concatenate([
        [v_mean * 0.05],          # omega
        np.full(p, 0.05),         # alpha_i
        np.full(q, 0.90),         # beta_j
    ])

    res = minimize(
        _mem_negloglik, theta0, args=(v, p, q, m),
        method='Nelder-Mead',
        options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True},
    )

    omega = float(res.x[0])
    alpha = np.maximum(res.x[1:1 + p], 0.0)
    beta  = np.maximum(res.x[1 + p:1 + p + q], 0.0)
    persistence = float(alpha.sum() + beta.sum())

    if not res.success:
        print(f"   WARNING: MEM/ACD({p},{q}) ottimizzazione non converge "
              f"pulita (status: {res.message})")
    if persistence >= 0.999:
        print(f"   WARNING: MEM/ACD({p},{q}) persistenza (alpha+beta) = "
              f"{persistence:.4f} >= 1 -> non stazionario, risultati inaffidabili")

    mu = _mem_build_mu(v, omega, alpha, beta, m)
    z_hat = v[m:] / mu[m:]
    z_var = float(np.var(z_hat))
    k_shape = 1.0 / z_var if z_var > 1e-8 else 1e4  # cap: dispersione ~ 0
    k_shape = float(np.clip(k_shape, 0.05, 1e4))

    return {
        'p': p, 'q': q, 'omega': omega, 'alpha': alpha, 'beta': beta,
        'k_shape': k_shape, 'persistence': persistence,
        'mu_seed': mu[:m], 'v_seed': v[:m],
        'converged': bool(res.success), 'nll': float(res.fun),
    }


def simulate_mem_acd(params, n_steps, rng):
    """
    Genera n_steps volumi > 0 tramite MEM/ACD(p,q):
        mu_t = omega + sum_i alpha_i * v_{t-i} + sum_j beta_j * mu_{t-j}
        v_t  = mu_t * z_t ,   z_t ~ Gamma(shape=k, scale=1/k)
    """
    omega, alpha, beta = params['omega'], params['alpha'], params['beta']
    p, q, k = params['p'], params['q'], params['k_shape']
    m = max(p, q)

    v_buf  = np.empty(m + n_steps)
    mu_buf = np.empty(m + n_steps)
    v_buf[:m]  = params['v_seed']
    mu_buf[:m] = params['mu_seed']

    z = rng.gamma(shape=k, scale=1.0 / k, size=n_steps)

    for t in range(n_steps):
        idx = m + t
        ar_term = 0.0
        for i in range(p):
            ar_term += alpha[i] * v_buf[idx - 1 - i]
        ma_term = 0.0
        for j in range(q):
            ma_term += beta[j] * mu_buf[idx - 1 - j]
        mu_buf[idx] = omega + ar_term + ma_term
        v_buf[idx]  = mu_buf[idx] * z[t]

    return v_buf[m:]


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

    # R^2: quanta varianza dei rendimenti reali e' spiegata dal segnale
    # di impatto X_t. Se e' basso (es. < 0.05), sigma_eta sta assorbendo
    # quasi tutta la varianza reale come "rumore idiosincratico": il
    # modello di impatto (beta, delta fissati) spiega poco, e la
    # simulazione sara' dominata dal termine gaussiano iid invece che
    # dalla struttura di long-memory.
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((r_all - r_all.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    print(f"  sigma_f   = {sigma_f:.6e}")
    print(f"  sigma_eta = {sigma_eta:.6e}")
    print(f"  R^2       = {r2:.4f}"
          f"  (frazione di varianza di r_t spiegata dal modello di impatto)")
    if r2 < 0.05:
        print("  WARNING: R^2 molto basso -> il modello di impatto (beta, "
              "delta correnti) spiega poco della varianza reale. La "
              "simulazione sara' dominata dal rumore gaussiano iid "
              "(sigma_eta) piuttosto che dalla struttura di impatto/"
              "long-memory. Considera di ricalibrare beta/delta o il kernel.")

    return sigma_f, sigma_eta, r2


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

def simulate_day(n_trades, vol_params, volume_model, alpha, n_traders, beta,
                 delta, sigma_f, sigma_eta, kernel_L, P0, rng):
    """
    Simula un singolo giorno con n_trades transazioni.

    volume_model : 'mem_acd' (default consigliato) o 'log_ar' (legacy)
    vol_params   : dict prodotto da fit_mem_acd() o fit_ar_log_volume(),
                   coerente con volume_model

    Returns: prices, volumes, signs
    """
    signs = simulate_lmf(alpha, n_traders, n_trades, rng)

    if volume_model == 'mem_acd':
        volumes = simulate_mem_acd(vol_params, n_trades, rng)
    elif volume_model == 'log_ar':
        volumes = simulate_ar_log_volume(vol_params, n_trades, rng)
    else:
        raise ValueError(f"volume_model sconosciuto: {volume_model!r} "
                          f"(atteso 'mem_acd' o 'log_ar')")

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
        volume_model = 'mem_acd',   # 'mem_acd' (consigliato) o 'log_ar' (legacy)
        mem_p     = 1,
        mem_q     = 1,
        ar_order  = 1000,
        beta      = 0.3,
        delta     = 0.5,
        kernel_L  = 500,
        seed      = 42,
        force_stationary_ar = True,
        ar_pole_threshold    = 0.98):
    """
    Per ogni file reale in data_dir:
      - calibra il modello di volume scelto sui volumi reali del giorno
      - usa P0 = primo prezzo reale (prezzo assoluto, non log)
      - simula lo stesso numero di transazioni del giorno reale
      - salva il CSV con timestamp equispaziati in [36000, 55797]

    sigma_f e sigma_eta vengono stimati automaticamente dai dati reali
    tramite OLS sul Transient Impact Model prima di avviare la simulazione.

    Parametri
    ---------
    alpha        : esponente Pareto LMF  (1 < alpha < 2)
    n_traders    : pool di trader LMF
    volume_model : 'mem_acd' (default) — Multiplicative Error Model /
                   ACD(p,q), positivo per costruzione, persistenza sulla
                   media condizionata del volume (Engle & Russell).
                   'log_ar' (legacy) — AR(p) sui log-volumi.
    mem_p, mem_q : ordini del MEM/ACD (default 1,1 — di solito sufficiente,
                   a differenza del log-AR non serve un ordine alto)
    ar_order     : ordine AR log-volume, usato solo se volume_model='log_ar'
                   (ridotto auto se dati insufficienti)
    beta         : esponente decadimento kernel TIM
    delta        : esponente impact function (0.5 = square-root)
    kernel_L     : troncamento memoria kernel (in numero di trade)
    seed         : seme per riproducibilita'
    force_stationary_ar : bool — solo per volume_model='log_ar': se True
                          (default), controlla che l'AR(p) sui log-volumi
                          sia stazionario e, se non lo e', rifitta
                          automaticamente con Yule-Walker
    ar_pole_threshold    : float — soglia sul modulo massimo dei poli AR
                          oltre la quale scatta il fallback Yule-Walker
    """
    rng      = np.random.default_rng(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Calibrazione automatica di sigma_f e sigma_eta ---
    sigma_f, sigma_eta, r2 = calibrate_tim(data_dir, beta=beta, delta=delta,
                                           kernel_L=kernel_L)

    paths = sorted(listdir(data_dir))
    print(f"\nFile trovati: {len(paths)}\n")

    for fname in paths:
        prices_real, volumes_real, _ = open_real_data(fname, data_dir)
        P0       = float(prices_real[0])   # prezzo assoluto, NON log-prezzo
        n_trades = len(prices_real)

        print(f"-> {fname}  |  n_trades={n_trades}  |  P0={P0:.4f}")

        if volume_model == 'mem_acd':
            vol_params = fit_mem_acd(volumes_real, p=mem_p, q=mem_q)
            print(f"   MEM/ACD({mem_p},{mem_q}): omega={vol_params['omega']:.4f} "
                  f"alpha={np.round(vol_params['alpha'], 4)} "
                  f"beta={np.round(vol_params['beta'], 4)} "
                  f"persistenza={vol_params['persistence']:.4f} "
                  f"k_shape={vol_params['k_shape']:.3f} "
                  f"converged={vol_params['converged']}")

        elif volume_model == 'log_ar':
            p_eff = ar_order if len(volumes_real) > ar_order else max(1, len(volumes_real) // 10)
            if p_eff < ar_order:
                print(f"   WARNING: dati reali insufficienti, AR order {ar_order} -> {p_eff}")

            vol_params = fit_ar_log_volume(
                volumes_real, p=p_eff,
                force_stationary=force_stationary_ar,
                pole_threshold=ar_pole_threshold,
            )
            if vol_params['method'] == 'yule_walker':
                print(f"   (AR log-volume fittato con Yule-Walker per stabilita')")
        else:
            raise ValueError(f"volume_model sconosciuto: {volume_model!r} "
                              f"(atteso 'mem_acd' o 'log_ar')")

        prices, volumes, signs = simulate_day(
            n_trades     = n_trades,
            vol_params   = vol_params,
            volume_model = volume_model,
            alpha        = alpha,
            n_traders    = n_traders,
            beta         = beta,
            delta        = delta,
            sigma_f      = sigma_f,
            sigma_eta    = sigma_eta,
            kernel_L     = kernel_L,
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
        out_dir   = r"..\database\data_lmf_tim_sqrt_mem_1.5_50",

        # LMF
        alpha     = 1.5,    # esponente Pareto (1 < alpha < 2 -> memoria lunga)
        n_traders = 50,     # pool di trader

        # Volumi
        volume_model = 'mem_acd',  # 'mem_acd' (consigliato) o 'log_ar' (legacy)
        mem_p     = 1,      # ordine AR di mu_t sui volumi passati
        mem_q     = 1,      # ordine MA di mu_t sulla media condizionata passata
        ar_order  = 100,   # usato solo se volume_model='log_ar' (ridotto auto se insufficiente)

        # TIM
        beta      = 0.25,    # esponente decadimento kernel power-law
        delta     = 0.5,    # esponente impact (0.5 = square-root)
        # sigma_f e sigma_eta: calibrati automaticamente dai dati reali

        kernel_L  = 500,    # memoria massima kernel (in trade)
        seed      = 42,
    )