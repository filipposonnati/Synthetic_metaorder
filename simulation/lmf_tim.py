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
                   z_t > 0 i.i.d. media 1 e mu_t dinamica GARCH-like.
                   Distribuzioni supportate per z_t:
                     * 'inverse_gaussian' (Wald)
                     * 'lognormal'
                     * 'burr12' (Burr Type XII)
                 - 'log_ar' (legacy): AR(p) sui log-volumi.
               Entrambi producono v_t > 0.

  3. PREZZI  — Transient Impact Model (TIM), calibrato sui dati reali.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from os import listdir
from scipy.optimize import minimize
import scipy.special as special


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
    p = len(phi)
    if p == 1:
        return float(np.abs(phi[0]))
    companion = np.zeros((p, p))
    companion[0, :] = phi
    companion[1:, :-1] = np.eye(p - 1)
    eigvals = np.linalg.eigvals(companion)
    return float(np.max(np.abs(eigvals)))


def _fit_ar_yule_walker(log_v, p):
    x  = log_v - log_v.mean()
    n  = len(x)
    acov = np.array([np.dot(x[:n - k], x[k:]) / n for k in range(p + 1)])

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
    log_v = np.log(np.maximum(volumes, 1e-12))
    n     = len(log_v)
    X     = np.column_stack([log_v[p - i : n - i] for i in range(1, p + 1)])
    y     = log_v[p:]
    phi, *_ = np.linalg.lstsq(X, y, rcond=None)

    method = 'ols'
    max_pole = _ar_max_pole_modulus(phi)

    if force_stationary and max_pole >= pole_threshold:
        print(f"   WARNING: AR({p}) OLS instabile (max |polo| = {max_pole:.4f})"
              f" -> fallback a Yule-Walker")
        phi, _ = _fit_ar_yule_walker(log_v, p)
        method = 'yule_walker'
        max_pole = _ar_max_pole_modulus(phi)

    sigma = float(np.std(y - X @ phi))
    return {
        'p': p, 'phi': phi, 'sigma': sigma, 'log_vol_seed': log_v[:p],
        'method': method, 'max_pole_modulus': max_pole,
    }


def simulate_ar_log_volume(params, n_steps, rng):
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

def _mem_build_mu(v, omega, alpha, beta, m):
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


def _fit_burr12_dist(z_hat):
    """
    Stima i parametri c e d della Burr XII imponendo E[z] = 1.
    Dato che E[z] = d * B(1 + 1/c, d - 1/c), fissiamo il fattore di scala d
    analiticamente e ottimizziamo rispetto a c (con c > 1 e d > 1/c).
    """
    def obj(c_val):
        if c_val <= 1.001: return 1e10
        # Calcolo del d teorico per avere media = 1 partendo dalla varianza campionaria
        # E[z^2] = d * B(1 + 2/c, d - 2/c)
        # Ottimizzazione semplificata accoppiando momento secondo e vincolo della media
        z2_mean = np.mean(z_hat**2)
        
        def inner_obj(d_val):
            if d_val <= 2.0 / c_val: return 1e10
            mean_theo = d_val * special.beta(1.0 + 1.0/c_val, d_val - 1.0/c_val)
            m2_theo = d_val * special.beta(1.0 + 2.0/c_val, d_val - 2.0/c_val)
            # Normalizziamo la scala affinché la media sia 1
            scale_adj = 1.0 / mean_theo
            m2_adj = m2_theo * (scale_adj**2)
            return (m2_adj - z2_mean)**2
            
        res_d = minimize(inner_obj, [2.0 * c_val + 1.0], method='Nelder-Mead', bounds=[(2.0/c_val + 0.01, None)])
        return float(res_d.fun)

    res_c = minimize(obj, [3.0], method='Nelder-Mead', bounds=[(1.01, None)])
    c = max(float(res_c.x[0]), 1.01)
    
    # Ricalcola il d ottimale finale
    def final_d_obj(d_val):
        if d_val <= 2.0 / c: return 1e10
        mean_theo = d_val * special.beta(1.0 + 1.0/c, d_val - 1.0/c)
        m2_theo = d_val * special.beta(1.0 + 2.0/c, d_val - 2.0/c)
        scale_adj = 1.0 / mean_theo
        return ((m2_theo * scale_adj**2) - np.mean(z_hat**2))**2
        
    res_d_final = minimize(final_d_obj, [2.0 * c + 1.0], method='Nelder-Mead')
    d = max(float(res_d_final.x[0]), 2.0 / c + 0.01)
    return c, d


def fit_mem_acd(volumes, p=1, q=1, dist='inverse_gaussian'):
    """
    Calibra un MEM/ACD(p,q) sui volumi reali.
    Supporta: 'inverse_gaussian', 'lognormal', 'burr12'
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

    mu = _mem_build_mu(v, omega, alpha, beta, m)
    z_hat = v[m:] / mu[m:]
    z_var = float(np.var(z_hat))

    dist_params = {}
    if dist == 'inverse_gaussian':
        ig_scale = 1.0 / z_var if z_var > 1e-8 else 1e4
        dist_params['ig_scale'] = float(np.clip(ig_scale, 0.05, 1e4))
    elif dist == 'lognormal':
        # E[z] = exp(mu_norm + sigma_norm^2 / 2) = 1 => mu_norm = -sigma_norm^2 / 2
        # Var(z) = exp(sigma_norm^2) - 1
        sigma2_ln = np.log(z_var + 1.0)
        dist_params['sigma_ln'] = float(np.sqrt(max(sigma2_ln, 1e-4)))
    elif dist == 'burr12':
        c, d = _fit_burr12_dist(z_hat)
        dist_params['burr_c'] = c
        dist_params['burr_d'] = d
    else:
        raise ValueError(f"Distribuzione non riconosciuta: {dist}")

    return {
        'p': p, 'q': q, 'omega': omega, 'alpha': alpha, 'beta': beta,
        'dist': dist, 'dist_params': dist_params, 'persistence': persistence,
        'mu_seed': mu[:m], 'v_seed': v[:m], 'converged': bool(res.success)
    }


def simulate_mem_acd(params, n_steps, rng):
    omega, alpha, beta = params['omega'], params['alpha'], params['beta']
    p, q, dist = params['p'], params['q'], params['dist']
    m = max(p, q)

    v_buf  = np.empty(m + n_steps)
    mu_buf = np.empty(m + n_steps)
    v_buf[:m]  = params['v_seed']
    mu_buf[:m] = params['mu_seed']

    # Generazione dei residui standardizzati z_t con E[z] = 1
    dp = params['dist_params']
    if dist == 'inverse_gaussian':
        z = rng.wald(mean=1.0, scale=dp['ig_scale'], size=n_steps)
    elif dist == 'lognormal':
        s = dp['sigma_ln']
        z = rng.lognormal(mean=-0.5 * (s**2), sigma=s, size=n_steps)
    elif dist == 'burr12':
        c, d = dp['burr_c'], dp['burr_d']
        u = rng.uniform(0.0, 1.0, size=n_steps)
        # Burr XII standard inv-CDF: x = ((1-u)**(-1/d) - 1)**(1/c)
        z_raw = ((1.0 - u)**(-1.0 / d) - 1.0)**(1.0 / c)
        # Forziamo la media esatta a 1 riscalando con la media teorica
        mean_theo = d * special.beta(1.0 + 1.0/c, d - 1.0/c)
        z = z_raw / mean_theo
    else:
        raise ValueError(f"Errore simulazione: {dist}")

    for t in range(n_steps):
        idx = m + t
        ar_term = sum(alpha[i] * v_buf[idx - 1 - i] for i in range(p))
        ma_term = sum(beta[j] * mu_buf[idx - 1 - j] for j in range(q))
        mu_buf[idx] = omega + ar_term + ma_term
        v_buf[idx]  = mu_buf[idx] * z[t]

    return v_buf[m:]


# ---------------------------------------------------------------------------
# LAYER 3 — PREZZI  (Transient Impact Model)
# ---------------------------------------------------------------------------

def _build_impact_signal(signs, volumes, beta, delta, kernel_L):
    T      = len(signs)
    kernel = (np.arange(kernel_L, dtype=float) + 1.0) ** (-beta)
    raw    = signs.astype(float) * (volumes ** delta)
    return np.convolve(raw, kernel, mode='full')[:T]


def calibrate_tim(data_dir, beta, delta, kernel_L):
    all_r, all_X = [], []
    for fname in sorted(listdir(data_dir)):
        prices, volumes, signs = open_real_data(fname, data_dir)
        if len(prices) < 2: continue
        log_r = np.diff(np.log(np.maximum(prices, 1e-12)))
        X = _build_impact_signal(signs[:-1], volumes[:-1], beta, delta, min(kernel_L, len(signs) - 1))
        all_r.append(log_r)
        all_X.append(X)

    r_all = np.concatenate(all_r)
    X_all = np.concatenate(all_X)
    sigma_f   = float(np.dot(X_all, r_all) / np.dot(X_all, X_all))
    sigma_eta = float(np.std(r_all - sigma_f * X_all))
    return sigma_f, sigma_eta


def simulate_tim(signs, volumes, beta, delta, sigma_f, sigma_eta, kernel_L, P0, rng):
    T = len(signs)
    X       = _build_impact_signal(signs, volumes, beta, delta, kernel_L)
    log_ret = sigma_f * X + rng.normal(0.0, sigma_eta, T)
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
    signs = simulate_lmf(alpha, n_traders, n_trades, rng)

    if volume_model == 'mem_acd':
        volumes = simulate_mem_acd(vol_params, n_trades, rng)
    elif volume_model == 'log_ar':
        volumes = simulate_ar_log_volume(vol_params, n_trades, rng)
    else:
        raise ValueError(f"volume_model sconosciuto: {volume_model!r}")

    prices  = simulate_tim(signs, volumes, beta=beta, delta=delta,
                           sigma_f=sigma_f, sigma_eta=sigma_eta,
                           kernel_L=min(kernel_L, n_trades), P0=P0, rng=rng)
    return prices, volumes, signs


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run(data_dir  = r"..\database\data",
        out_dir   = r"..\database\data_synthetic",
        alpha     = 1.5,
        n_traders = 10,
        volume_model = 'mem_acd',
        mem_dist  = 'inverse_gaussian',  # <--- NUOVO PARAMETRO CORRENTE: 'inverse_gaussian', 'lognormal', 'burr12'
        mem_p     = 1,
        mem_q     = 1,
        ar_order  = 100,
        beta      = 0.3,
        delta     = 0.5,
        kernel_L  = 500,
        seed      = 42):
    
    rng      = np.random.default_rng(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sigma_f, sigma_eta = calibrate_tim(data_dir, beta=beta, delta=delta, kernel_L=kernel_L)
    paths = sorted(listdir(data_dir))

    for fname in paths:
        prices_real, volumes_real, _ = open_real_data(fname, data_dir)
        P0       = float(prices_real[0])
        n_trades = len(prices_real)

        print(f"-> {fname}  |  n_trades={n_trades}  |  Distribuzione={mem_dist}")

        if volume_model == 'mem_acd':
            vol_params = fit_mem_acd(volumes_real, p=mem_p, q=mem_q, dist=mem_dist)
            print(f"   MEM/ACD({mem_p},{mem_q}): Parametri Distr={vol_params['dist_params']}")
        elif volume_model == 'log_ar':
            p_eff = ar_order if len(volumes_real) > ar_order else max(1, len(volumes_real) // 10)
            vol_params = fit_ar_log_volume(volumes_real, p=p_eff)
        else:
            raise ValueError(f"volume_model sconosciuto: {volume_model!r}")

        prices, volumes, signs = simulate_day(
            n_trades=n_trades, vol_params=vol_params, volume_model=volume_model,
            alpha=alpha, n_traders=n_traders, beta=beta, delta=delta,
            sigma_f=sigma_f, sigma_eta=sigma_eta, kernel_L=kernel_L, P0=P0, rng=rng
        )

        save_simulated_data(str(out_path / fname), prices, volumes, signs, n_trades)

    print("\nSimulazione completata.")


if __name__ == '__main__':
    run(
        data_dir     = r"..\database\data",
        out_dir      = r"..\database\data_lmf_tim_sqrt_mem_1.5_50",
        alpha        = 1.5,
        n_traders    = 50,
        volume_model = 'mem_acd',
        mem_dist     = 'lognormal',  # Cambia rapidamente qui: 'inverse_gaussian' | 'lognormal' | 'burr12'
        mem_p        = 1,
        mem_q        = 1,
        beta         = 0.25,
        delta        = 0.5,
        kernel_L     = 500,
        seed         = 42,
    )