"""
volumes_generation.py
======================
Modulo dedicato alla calibrazione e alla simulazione dei volumi di mercato.
Supporta:
  - Modello AR(p) sui log-volumi (fit giorno per giorno)
  - Modello MEM/ACD(p,q) con diverse distribuzioni (fit giorno per giorno)
  - Campionamento Empirico / Bootstrap direttamente dai dati reali
  - Fit su PIU' GIORNI in un unico modello (fit_ar_log_volume_multi, fit_mem_acd_multi),
    con R2 in-sample, valutazione out-of-sample, salvataggio/caricamento del modello
    (stesso schema di tim.py: lag interni al giorno, nessuna concatenazione dei giorni)

I volumi sono volumi SENZA segno (> 0); ogni giorno e' un array separato.
"""

import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize
from scipy.signal import lfilter, lfiltic
import scipy.special as special

# ---------------------------------------------------------------------------
# HELPER PER SELEZIONE CASUALE GIORNI E BOOTSTRAP
# ---------------------------------------------------------------------------

def _select_random_days(volumes_list, sample_days, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    n_days = len(volumes_list)
    if sample_days is not None and sample_days < n_days:
        idx = rng.choice(n_days, size=sample_days, replace=False)
        return [volumes_list[i] for i in idx]
    return volumes_list


def sample_empirical_volumes(volumes_list, n_steps, rng):
    """
    Campiona casualmente (con reinserimento) 'n_steps' volumi 
    dalla distribuzione empirica dei volumi reali forniti.
    """
    if isinstance(volumes_list, list):
        flat_volumes = np.concatenate(volumes_list)
    else:
        flat_volumes = np.asarray(volumes_list, dtype=float)

    return rng.choice(flat_volumes, size=n_steps, replace=True)


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


def _fit_ar_yule_walker_single(log_v, p):
    x = log_v - log_v.mean()
    n = len(x)
    if n <= p:
        return np.zeros(p), 1.0
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


def fit_ar_log_volume(volumes_list, p, sample_days=10, force_stationary=True, pole_threshold=0.98, rng=None):
    sampled_v_list = _select_random_days(volumes_list, sample_days, rng)
    
    phi_list = []
    sigma_list = []
    poles_list = []

    for v in sampled_v_list:
        log_v = np.log(np.maximum(v, 1e-12))
        n = len(log_v)
        if n <= p + 1:
            continue

        X_day = np.column_stack([log_v[p - i : n - i] for i in range(1, p + 1)])
        y_day = log_v[p:]

        phi_day, *_ = np.linalg.lstsq(X_day, y_day, rcond=None)
        max_pole = _ar_max_pole_modulus(phi_day)

        if force_stationary and max_pole >= pole_threshold:
            phi_day, _ = _fit_ar_yule_walker_single(log_v, p)
            max_pole = _ar_max_pole_modulus(phi_day)

        sigma_day = float(np.std(y_day - X_day @ phi_day))

        phi_list.append(phi_day)
        sigma_list.append(sigma_day)
        poles_list.append(max_pole)

    if not phi_list:
        raise ValueError("Nessuna giornata valida per calibrare il modello AR dei volumi.")

    first_log_v = np.log(np.maximum(sampled_v_list[0], 1e-12))

    return {
        'p': p, 
        'phi': np.mean(phi_list, axis=0), 
        'sigma': float(np.mean(sigma_list)), 
        'log_vol_seed': first_log_v[:p],
        'method': 'day_by_day_mean', 
        'max_pole_modulus': float(np.mean(poles_list)),
    }


def simulate_ar_log_volume(params, n_steps, rng):
    p, phi, sigma = params['p'], params['phi'], params['sigma']
    buf = np.empty(p + n_steps)
    buf[:p] = params['log_vol_seed']
    noise = rng.normal(0.0, sigma, n_steps)
    c = params.get('c', 0.0)   # intercept (presente solo nei modelli del fit multi-giorno)
    for t in range(n_steps):
        buf[p + t] = c + buf[t : t + p][::-1] @ phi + noise[t]
    return np.exp(buf[p:])


# ---------------------------------------------------------------------------
# LAYER 2bis — VOLUMI  (Multiplicative Error Model / ACD(p,q))
# ---------------------------------------------------------------------------

def _mem_build_mu(v, omega, alpha, beta, m):
    T = len(v)
    p = len(alpha)
    q = len(beta)
    mu = np.empty(T)
    mu[:m] = v[:m].mean() if m > 0 else v.mean()
    for t in range(m, T):
        ar_term = sum(alpha[i] * v[t - 1 - i] for i in range(p))
        ma_term = sum(beta[j] * mu[t - 1 - j] for j in range(q))
        mu[t] = omega + ar_term + ma_term
    return mu


def _mem_negloglik_single(theta, v, p, q, m):
    omega = theta[0]
    alpha = theta[1:1 + p]
    beta = theta[1 + p:1 + p + q]

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
    v_eff = v[m:]
    if np.any(mu_eff <= 0.0) or not np.all(np.isfinite(mu_eff)):
        return 1e10

    nll = float(np.sum(np.log(mu_eff) + v_eff / mu_eff))
    return nll if np.isfinite(nll) else 1e10


def _fit_burr12_dist(z_hat):
    z2_mean = float(np.mean(z_hat**2))

    def obj(c_val):
        c_scalar = float(np.squeeze(c_val))
        if c_scalar <= 1.001:
            return 1e10
        
        def inner_obj(d_val):
            d_scalar = float(np.squeeze(d_val))
            if d_scalar <= 2.0 / c_scalar:
                return 1e10
            try:
                mean_theo = d_scalar * special.beta(1.0 + 1.0 / c_scalar, d_scalar - 1.0 / c_scalar)
                m2_theo = d_scalar * special.beta(1.0 + 2.0 / c_scalar, d_scalar - 2.0 / c_scalar)
                if not (np.isfinite(mean_theo) and np.isfinite(m2_theo)) or mean_theo <= 0:
                    return 1e10
                scale_adj = 1.0 / mean_theo
                m2_adj = m2_theo * (scale_adj**2)
                return float((m2_adj - z2_mean)**2)
            except Exception:
                return 1e10
            
        x0_d = float(2.0 * c_scalar + 1.0)
        bounds_d = [(2.0 / c_scalar + 0.01, None)]
        res_d = minimize(inner_obj, [x0_d], method='Nelder-Mead', bounds=bounds_d)
        return float(res_d.fun)

    res_c = minimize(obj, [3.0], method='Nelder-Mead', bounds=[(1.01, None)])
    c = max(float(res_c.x[0]), 1.01)
    
    def final_d_obj(d_val):
        d_scalar = float(np.squeeze(d_val))
        if d_scalar <= 2.0 / c:
            return 1e10
        try:
            mean_theo = d_scalar * special.beta(1.0 + 1.0 / c, d_scalar - 1.0 / c)
            m2_theo = d_scalar * special.beta(1.0 + 2.0 / c, d_scalar - 2.0 / c)
            if not (np.isfinite(mean_theo) and np.isfinite(m2_theo)) or mean_theo <= 0:
                return 1e10
            scale_adj = 1.0 / mean_theo
            return float(((m2_theo * scale_adj**2) - z2_mean)**2)
        except Exception:
            return 1e10
        
    x0_final_d = float(2.0 * c + 1.0)
    res_d_final = minimize(final_d_obj, [x0_final_d], method='Nelder-Mead', bounds=[(2.0 / c + 0.01, None)])
    d = max(float(res_d_final.x[0]), 2.0 / c + 0.01)
    return c, d


def fit_mem_acd(volumes_list, p=1, q=1, dist='inverse_gaussian', sample_days=10, rng=None):
    sampled_v_list = _select_random_days(volumes_list, sample_days, rng)
    cleaned_v_list = [np.maximum(np.asarray(v, dtype=float), 1e-12) for v in sampled_v_list]
    
    m = max(p, q)

    omega_list, alpha_list, beta_list, dist_params_list = [], [], [], []

    for v in cleaned_v_list:
        if len(v) <= m + 1:
            continue

        v_mean = float(np.mean(v))
        theta0 = np.concatenate([[v_mean * 0.05], np.full(p, 0.05), np.full(q, 0.90)])

        res = minimize(
            _mem_negloglik_single, theta0, args=(v, p, q, m),
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True},
        )

        omega_day = float(res.x[0])
        alpha_day = np.maximum(res.x[1:1 + p], 0.0)
        beta_day = np.maximum(res.x[1 + p:1 + p + q], 0.0)

        mu_day = _mem_build_mu(v, omega_day, alpha_day, beta_day, m)
        z_hat_day = v[m:] / mu_day[m:]
        z_var_day = float(np.var(z_hat_day))

        dp_day = {}
        if dist == 'inverse_gaussian':
            ig_scale = 1.0 / z_var_day if z_var_day > 1e-8 else 1e4
            dp_day['ig_scale'] = float(np.clip(ig_scale, 0.05, 1e4))
        elif dist == 'lognormal':
            sigma2_ln = np.log(z_var_day + 1.0)
            dp_day['sigma_ln'] = float(np.sqrt(max(sigma2_ln, 1e-4)))
        elif dist == 'burr12':
            c_day, d_day = _fit_burr12_dist(z_hat_day)
            dp_day['burr_c'] = c_day
            dp_day['burr_d'] = d_day

        omega_list.append(omega_day)
        alpha_list.append(alpha_day)
        beta_list.append(beta_day)
        dist_params_list.append(dp_day)

    if not omega_list:
        raise ValueError("Nessuna giornata valida per il fit MEM/ACD.")

    mean_dist_params = {}
    if dist == 'inverse_gaussian':
        mean_dist_params['ig_scale'] = float(np.mean([dp['ig_scale'] for dp in dist_params_list]))
    elif dist == 'lognormal':
        mean_dist_params['sigma_ln'] = float(np.mean([dp['sigma_ln'] for dp in dist_params_list]))
    elif dist == 'burr12':
        mean_dist_params['burr_c'] = float(np.mean([dp['burr_c'] for dp in dist_params_list]))
        mean_dist_params['burr_d'] = float(np.mean([dp['burr_d'] for dp in dist_params_list]))

    first_v = cleaned_v_list[0]
    mean_omega, mean_alpha, mean_beta = float(np.mean(omega_list)), np.mean(alpha_list, axis=0), np.mean(beta_list, axis=0)
    first_mu = _mem_build_mu(first_v, mean_omega, mean_alpha, mean_beta, m)

    return {
        'p': p, 'q': q, 
        'omega': mean_omega, 
        'alpha': mean_alpha, 
        'beta': mean_beta,
        'dist': dist, 
        'dist_params': mean_dist_params, 
        'persistence': float(mean_alpha.sum() + mean_beta.sum()),
        'mu_seed': first_mu[:m], 
        'v_seed': first_v[:m], 
        'converged': True
    }


def simulate_mem_acd(params, n_steps, rng):
    omega, alpha, beta = params['omega'], params['alpha'], params['beta']
    p, q, dist = params['p'], params['q'], params['dist']
    m = max(p, q)

    v_buf = np.empty(m + n_steps)
    mu_buf = np.empty(m + n_steps)
    v_buf[:m] = params['v_seed']
    mu_buf[:m] = params['mu_seed']

    dp = params['dist_params']
    if dist == 'inverse_gaussian':
        z = rng.wald(mean=1.0, scale=dp['ig_scale'], size=n_steps)
    elif dist == 'lognormal':
        s = dp['sigma_ln']
        z = rng.lognormal(mean=-0.5 * (s**2), sigma=s, size=n_steps)
    elif dist == 'burr12':
        c, d = dp['burr_c'], dp['burr_d']
        u = rng.uniform(0.0, 1.0, size=n_steps)
        z_raw = ((1.0 - u)**(-1.0 / d) - 1.0)**(1.0 / c)
        mean_theo = d * special.beta(1.0 + 1.0/c, d - 1.0/c)
        z = z_raw / mean_theo
    else:
        raise ValueError(f"Errore simulazione: {dist}")

    for t in range(n_steps):
        idx = m + t
        ar_term = sum(alpha[i] * v_buf[idx - 1 - i] for i in range(p))
        ma_term = sum(beta[j] * mu_buf[idx - 1 - j] for j in range(q))
        mu_buf[idx] = omega + ar_term + ma_term
        v_buf[idx] = mu_buf[idx] * z[t]

    return v_buf[m:]


# ===========================================================================
#  FIT SU PIU' GIORNI (un unico modello, stile tim.py)
# ===========================================================================
# - i giorni NON vengono concatenati: i lag restano dentro il giorno
# - AR: equazioni normali accumulate giorno per giorno / blocco per blocco
#   (memoria O(chunk * p), equivalente all'OLS sulle righe impilate)
# - MEM: massima quasi-verosimiglianza (esponenziale) POOLED sui giorni
# - R2 calcolato attorno alla media del campione valutato (come in tim.py)
# ---------------------------------------------------------------------------

def _model_stats(ssr, n, sy, sy2, k):
    """R2 e R2 aggiustato attorno alla media del campione; k = n. di parametri."""
    sst = sy2 - sy ** 2 / n
    r2 = 1.0 - ssr / sst
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    return float(r2), float(r2_adj)


def _clean_days(volumes_list, sample_days, rng, min_len):
    """Seleziona (opzionalmente a caso) i giorni e scarta quelli troppo corti.
    Ritorna volumi float64 > 0 (clip a 1e-12 come nel resto del modulo)."""
    days = _select_random_days(list(volumes_list), sample_days, rng)
    days = [np.maximum(np.asarray(v, dtype=np.float64).ravel(), 1e-12) for v in days]
    days = [v for v in days if len(v) > min_len]
    if not days:
        raise ValueError("Nessun giorno con lunghezza sufficiente per il fit.")
    return days


# ---------------------------------------------------------------------------
# AR(p) sui log-volumi, multi-giorno
# ---------------------------------------------------------------------------

def _ar_blocks(x, p, chunk):
    """Blocchi (X, y) di al piu' `chunk` righe per un giorno.
    X = [1, x_{t-1}, ..., x_{t-p}], y = x_t (lag interni al giorno, nessuna copia totale)."""
    n = len(x)
    if n <= p:
        return
    W = sliding_window_view(x, p)[:n - p, ::-1]
    for s in range(0, n - p, chunk):
        e = min(s + chunk, n - p)
        X = np.empty((e - s, p + 1))
        X[:, 0] = 1.0
        X[:, 1:] = W[s:e]
        yield X, x[p + s:p + e]


def _levinson_durbin(acov, p):
    """Coefficienti AR(p) dalle autocovarianze acov[0..p] (Yule-Walker)."""
    phi = np.zeros(p)
    err = acov[0]
    for k in range(p):
        acc = acov[k + 1] - np.dot(phi[:k], acov[k:0:-1])
        refl = acc / err if err > 1e-300 else 0.0
        new = phi.copy()
        new[k] = refl
        if k > 0:
            new[:k] = phi[:k] - refl * phi[k - 1::-1]
        phi = new
        err *= (1.0 - refl ** 2)
    return phi, float(np.sqrt(max(err, 1e-300)))


def _pooled_yule_walker(log_days, p):
    """Yule-Walker con autocovarianze accumulate su tutti i giorni (media globale)."""
    N = sum(len(x) for x in log_days)
    mean = sum(x.sum() for x in log_days) / N
    acov = np.zeros(p + 1)
    for x in log_days:
        xc = x - mean
        n = len(xc)
        nfft = 1 << int(np.ceil(np.log2(2 * n)))
        f = np.fft.rfft(xc, nfft)
        acov += np.fft.irfft(f * np.conj(f), nfft)[:p + 1]
    acov /= N
    phi, _ = _levinson_durbin(acov, p)
    return phi, mean


def fit_ar_log_volume_multi(volumes_list, p=10, sample_days=None, force_stationary=True,
                            pole_threshold=0.98, chunk=10000, rng=None, keep_resid=True):
    """Un unico AR(p) con intercetta sui log-volumi, stimato su piu' giorni.

        log V_t = c + sum_{i=1..p} phi_i log V_{t-i} + eps_t

    volumes_list : lista di array di volumi (senza segno), uno per giorno
    sample_days  : se None usa tutti i giorni, altrimenti ne estrae `sample_days` a caso
    force_stationary : se il modulo massimo dei poli dell'OLS e' >= pole_threshold
                       ripiega su Yule-Walker pooled (sempre stazionario)
    keep_resid   : conserva i residui (servono per bootstrap e per l'RMSE in-sample)

    Ritorna un dict compatibile con simulate_ar_log_volume (chiavi p, phi, sigma,
    log_vol_seed, + c) con in piu' rsquared, rsquared_adj, n_obs, resid.
    """
    log_days = [np.log(v) for v in _clean_days(volumes_list, sample_days, rng, p + 1)]
    k = p + 1

    # Passata 1: equazioni normali
    XtX, Xty = np.zeros((k, k)), np.zeros(k)
    n, sy, sy2 = 0, 0.0, 0.0
    for x in log_days:
        for X, y in _ar_blocks(x, p, chunk):
            XtX += X.T @ X
            Xty += X.T @ y
            n += len(y)
            sy += y.sum()
            sy2 += y @ y
    beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    c, phi = float(beta[0]), beta[1:]
    method = 'pooled_ols'

    max_pole = _ar_max_pole_modulus(phi)
    if force_stationary and max_pole >= pole_threshold:
        phi, mean = _pooled_yule_walker(log_days, p)
        c = float(mean * (1.0 - phi.sum()))
        beta = np.concatenate([[c], phi])
        max_pole = _ar_max_pole_modulus(phi)
        method = 'pooled_yule_walker'

    # Passata 2: residui e R2
    ssr = 0.0
    resid = []
    for x in log_days:
        for X, y in _ar_blocks(x, p, chunk):
            e = y - X @ beta
            ssr += e @ e
            if keep_resid:
                resid.append(e)
    r2, r2_adj = _model_stats(ssr, n, sy, sy2, k)
    sigma = float(np.sqrt(ssr / (n - k)))

    print(f"AR({p}) log-volumes R2: {r2:.4f} (adj {r2_adj:.4f}) | giorni: {len(log_days)} | "
          f"n: {n} | max pole: {max_pole:.4f} [{method}]")

    return {
        'kind': 'ar',
        'p': p,
        'c': c,
        'phi': np.asarray(phi, dtype=float),
        'sigma': sigma,
        'log_vol_seed': log_days[0][:p].copy(),
        'method': method,
        'max_pole_modulus': float(max_pole),
        'n_obs': int(n),
        'n_days': len(log_days),
        'rsquared': r2,
        'rsquared_adj': r2_adj,
        'resid': np.concatenate(resid) if keep_resid else None,
    }


def evaluate_ar_log_volume(params, volumes_list, chunk=10000):
    """Valutazione out-of-sample (previsione one-step-ahead sui log-volumi reali di giorni
    non usati nel fit). Ritorna statistiche pooled e R2 di ciascun giorno."""
    p = params['p']
    beta = np.concatenate([[params.get('c', 0.0)], np.asarray(params['phi'])])
    k = p + 1
    tot = np.zeros(4)   # n, sum y, sum y^2, ssr
    per_day = []
    for v in volumes_list:
        x = np.log(np.maximum(np.asarray(v, dtype=np.float64).ravel(), 1e-12))
        day = np.zeros(4)
        for X, y in _ar_blocks(x, p, chunk):
            e = y - X @ beta
            day += np.array([len(y), y.sum(), y @ y, e @ e])
        tot += day
        per_day.append(_model_stats(day[3], day[0], day[1], day[2], k)[0] if day[0] > k else np.nan)
    n, sy, sy2, ssr = tot
    r2, r2_adj = _model_stats(ssr, n, sy, sy2, k)
    return {'n': int(n), 'rmse': float(np.sqrt(ssr / n)), 'rsquared': r2,
            'rsquared_adj': r2_adj, 'per_day_rsquared': per_day}


# ---------------------------------------------------------------------------
# MEM / ACD(p,q), multi-giorno
# ---------------------------------------------------------------------------

def _mem_mu_fast(v, omega, alpha, beta, m):
    """Stessa ricorsione di _mem_build_mu, ma vettorizzata con scipy.signal.lfilter:
        mu_t = omega + sum_i alpha_i v_{t-1-i} + sum_j beta_j mu_{t-1-j},  mu[:m] = media(v[:m])"""
    T, p, q = len(v), len(alpha), len(beta)
    mu = np.empty(T)
    mu[:m] = v[:m].mean()
    u = np.full(T - m, omega)
    for i in range(p):
        u += alpha[i] * v[m - 1 - i:T - 1 - i]
    if q == 0:
        mu[m:] = u
    else:
        a = np.concatenate([[1.0], -np.asarray(beta)])
        zi = lfiltic([1.0], a, mu[m - q:m][::-1])
        mu[m:], _ = lfilter([1.0], a, u, zi=zi)
    return mu


def _mem_pooled_nll(theta, days, p, q, m):
    """Quasi-log-verosimiglianza negativa (esponenziale) sommata su tutti i giorni."""
    omega, alpha, beta = theta[0], theta[1:1 + p], theta[1 + p:1 + p + q]
    if omega <= 1e-12 or np.any(alpha < 0.0) or np.any(beta < 0.0):
        return 1e10
    persistence = float(alpha.sum() + beta.sum())
    if persistence >= 0.999:
        return 1e10 * (1.0 + persistence)

    nll = 0.0
    for v in days:
        mu = _mem_mu_fast(v, omega, alpha, beta, m)[m:]
        if not np.all(np.isfinite(mu)) or np.any(mu <= 0.0):
            return 1e10
        nll += np.sum(np.log(mu) + v[m:] / mu)
    return float(nll) if np.isfinite(nll) else 1e10


def fit_mem_acd_multi(volumes_list, p=1, q=1, dist='inverse_gaussian', sample_days=None,
                      rng=None, keep_resid=True):
    """Un unico MEM/ACD(p,q) stimato su piu' giorni (quasi-ML pooled).

    I parametri (omega, alpha, beta) sono comuni a tutti i giorni; la ricorsione riparte
    in ogni giorno (mu iniziale = media dei primi m volumi del giorno), quindi nessun
    legame tra la fine di un giorno e l'inizio del successivo.
    La distribuzione dell'innovazione z = V/mu (media 1) e' stimata dagli z pooled.

    R2 (sui volumi, previsione one-step-ahead V_hat_t = mu_t) e' attorno alla media del
    campione; essendo un modello moltiplicativo puo' essere basso o anche negativo.
    Ritorna un dict compatibile con simulate_mem_acd (+ rsquared, rsquared_adj, n_obs, z_hat).
    """
    if dist not in ('inverse_gaussian', 'lognormal', 'burr12'):
        raise ValueError(f"dist non supportata: {dist}")
    m = max(p, q)
    days = _clean_days(volumes_list, sample_days, rng, m + 1)

    # Lavora su volumi normalizzati (media globale 1): stesso ordine di grandezza per tutti i parametri
    scale = float(np.mean(np.concatenate(days)))
    days_n = [v / scale for v in days]

    theta0 = np.concatenate([[0.05], np.full(p, 0.05), np.full(q, 0.90)])
    res = minimize(
        _mem_pooled_nll, theta0, args=(days_n, p, q, m), method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True},
    )
    omega = float(res.x[0]) * scale
    alpha = np.maximum(res.x[1:1 + p], 0.0)
    beta = np.maximum(res.x[1 + p:1 + p + q], 0.0)

    # Residui moltiplicativi z e statistiche R2 (unita' originali)
    z_list = []
    n, sy, sy2, ssr = 0, 0.0, 0.0, 0.0
    for v in days:
        mu = _mem_mu_fast(v, omega, alpha, beta, m)
        y, mu_e = v[m:], mu[m:]
        z_list.append(y / mu_e)
        n += len(y)
        sy += y.sum()
        sy2 += y @ y
        ssr += np.sum((y - mu_e) ** 2)
    z_hat = np.concatenate(z_list)
    r2, r2_adj = _model_stats(ssr, n, sy, sy2, 1 + p + q)

    z_var = float(np.var(z_hat))
    if dist == 'inverse_gaussian':
        ig = 1.0 / z_var if z_var > 1e-8 else 1e4
        dist_params = {'ig_scale': float(np.clip(ig, 0.05, 1e4))}
    elif dist == 'lognormal':
        dist_params = {'sigma_ln': float(np.sqrt(max(np.log(z_var + 1.0), 1e-4)))}
    else:
        c_b, d_b = _fit_burr12_dist(z_hat)
        dist_params = {'burr_c': float(c_b), 'burr_d': float(d_b)}

    first_v = days[0]
    first_mu = _mem_mu_fast(first_v, omega, alpha, beta, m)
    persistence = float(alpha.sum() + beta.sum())

    print(f"MEM/ACD({p},{q}) [{dist}] volumes R2: {r2:.4f} (adj {r2_adj:.4f}) | "
          f"giorni: {len(days)} | n: {n} | persistence: {persistence:.4f} | "
          f"converged: {res.success}")

    return {
        'kind': 'mem',
        'p': p, 'q': q,
        'omega': omega, 'alpha': alpha, 'beta': beta,
        'dist': dist, 'dist_params': dist_params,
        'persistence': persistence,
        'mu_seed': first_mu[:m], 'v_seed': first_v[:m],
        'converged': bool(res.success),
        'n_obs': int(n), 'n_days': len(days),
        'rsquared': r2, 'rsquared_adj': r2_adj,
        'rmse': float(np.sqrt(ssr / n)),
        'z_hat': z_hat if keep_resid else None,
    }


def evaluate_mem_acd(params, volumes_list):
    """Valutazione out-of-sample del MEM (previsione one-step-ahead V_hat_t = mu_t con i
    volumi reali del giorno di test). Ritorna statistiche pooled e R2 per giorno."""
    p, q = params['p'], params['q']
    m = max(p, q)
    k = 1 + p + q
    tot = np.zeros(4)
    per_day = []
    for v in volumes_list:
        v = np.maximum(np.asarray(v, dtype=np.float64).ravel(), 1e-12)
        if len(v) <= m + 1:
            per_day.append(np.nan)
            continue
        mu = _mem_mu_fast(v, params['omega'], np.asarray(params['alpha']),
                          np.asarray(params['beta']), m)
        y, mu_e = v[m:], mu[m:]
        day = np.array([len(y), y.sum(), y @ y, np.sum((y - mu_e) ** 2)])
        tot += day
        per_day.append(_model_stats(day[3], day[0], day[1], day[2], k)[0] if day[0] > k else np.nan)
    n, sy, sy2, ssr = tot
    r2, r2_adj = _model_stats(ssr, n, sy, sy2, k)
    return {'n': int(n), 'rmse': float(np.sqrt(ssr / n)), 'rsquared': r2,
            'rsquared_adj': r2_adj, 'per_day_rsquared': per_day}


# ---------------------------------------------------------------------------
# Salvataggio / caricamento del modello finale (.npz, nessun pickle)
# ---------------------------------------------------------------------------

def save_volume_model(params, path):
    """Salva il modello (AR o MEM) prodotto da fit_*_multi in un .npz senza pickle.
    Contiene tutto cio' che serve a simulate_ar_log_volume / simulate_mem_acd,
    piu' R2 e (se presenti) i residui."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kind = params['kind']

    if kind == 'ar':
        meta = {k: params[k] for k in ('p', 'c', 'sigma', 'method', 'max_pole_modulus',
                                       'n_obs', 'n_days', 'rsquared', 'rsquared_adj')}
        arrays = {'phi': np.asarray(params['phi']),
                  'log_vol_seed': np.asarray(params['log_vol_seed'])}
        if params.get('resid') is not None:
            arrays['resid'] = np.asarray(params['resid'])
    elif kind == 'mem':
        meta = {k: params[k] for k in ('p', 'q', 'omega', 'dist', 'dist_params', 'persistence',
                                       'converged', 'n_obs', 'n_days', 'rsquared',
                                       'rsquared_adj', 'rmse')}
        arrays = {'alpha': np.asarray(params['alpha']), 'beta': np.asarray(params['beta']),
                  'mu_seed': np.asarray(params['mu_seed']), 'v_seed': np.asarray(params['v_seed'])}
        if params.get('z_hat') is not None:
            arrays['z_hat'] = np.asarray(params['z_hat'])
    else:
        raise ValueError(f"kind sconosciuto: {kind}")

    meta['kind'] = kind
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


def load_volume_model(path):
    """Ricarica un modello salvato con save_volume_model, nello stesso formato del fit."""
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f['meta']))
        arrays = {k: f[k] for k in f.files if k != 'meta'}

    if meta['kind'] == 'ar':
        meta['phi'] = arrays['phi']
        meta['log_vol_seed'] = arrays['log_vol_seed']
        meta['resid'] = arrays.get('resid')
    else:
        meta['alpha'], meta['beta'] = arrays['alpha'], arrays['beta']
        meta['mu_seed'], meta['v_seed'] = arrays['mu_seed'], arrays['v_seed']
        meta['z_hat'] = arrays.get('z_hat')
    return meta


def save_volume_oos_stats(params, oos, path, window_info=None):
    """Salva in JSON il confronto in-sample / out-of-sample e lo stampa."""
    summary = {
        'window_info': window_info or {},
        'in_sample': {'n': params['n_obs'], 'rsquared': params['rsquared'],
                      'rsquared_adj': params['rsquared_adj']},
        'out_of_sample': {k: (float(v) if not isinstance(v, list) else [float(x) for x in v])
                          for k, v in oos.items()},
    }
    print(f"{'sample':<16}{'n':>12}{'R2':>10}{'R2 adj':>10}")
    for label, r in (('in-sample', summary['in_sample']), ('out-of-sample', summary['out_of_sample'])):
        print(f"{label:<16}{int(r['n']):>12}{r['rsquared']:>10.4f}{r['rsquared_adj']:>10.4f}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)


def sample_empirical_volumes(volumes_list, n_steps, rng):
    """
    Campiona casualmente (con reinserimento) 'n_steps' volumi 
    dalla distribuzione empirica dei volumi reali forniti.
    """
    if isinstance(volumes_list, (list, tuple)):
        # Assegna una copia piatta (1D) convertendo tutti gli elementi in float
        flat_volumes = np.hstack([np.asarray(v, dtype=np.float64).ravel() for v in volumes_list])
    else:
        flat_volumes = np.asarray(volumes_list, dtype=np.float64).ravel()

    return rng.choice(flat_volumes, size=n_steps, replace=True)


# ---------------------------------------------------------------------------
# MAIN: calibrazione e salvataggio dei modelli log-AR e MEM (un modello per tutti i giorni)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from os import listdir
    from utils import open_data

    paths = np.array(sorted(listdir('..\\database\\data')))   # sorted -> chronological order

    p_ar = 1000                    # ordine AR sui log-volumi
    p_mem, q_mem = 1, 1           # ordini MEM/ACD
    mem_dist = 'burr12'  # 'inverse_gaussian' | 'lognormal' | 'burr12'
    fitting_window = 200          # numero di giorni usati per il fit
    trim_top = False               # come in tim.py: scarta i trade con |volume| nel top top_frac
    top_frac = 0.00
    refit = False                 # True: ricalibra anche se il modello esiste gia'

    def load_day(path):
        """Volumi senza segno di un giorno (stesso allineamento di tim.py: volumes[:-1])."""
        prices, volumes, signs = open_data(path)
        volumes = np.abs(np.asarray(volumes, dtype=float))
        if trim_top:
            thr = np.quantile(volumes, 1.0 - top_frac)
            volumes = volumes[volumes <= thr]
        return volumes[:-1]

    fit_paths = paths[:fitting_window]
    test_paths = paths[fitting_window:]

    models = {
        'logar': dict(name=f'logar_{p_ar}',
                      fit=lambda d: fit_ar_log_volume_multi(d, p=p_ar),
                      evaluate=evaluate_ar_log_volume),
        'mem': dict(name=f'mem_{mem_dist}_{p_mem}_{q_mem}',
                    fit=lambda d: fit_mem_acd_multi(d, p=p_mem, q=q_mem, dist=mem_dist),
                    evaluate=evaluate_mem_acd),
    }

    fit_days = test_days = None   # caricati solo se servono

    for key, cfg in models.items():
        model_dir = Path('..') / 'models' / cfg['name']
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'model.npz'

        if model_path.exists() and not refit:
            print(f'[{key}] Load Model')
            results = load_volume_model(model_path)
        else:
            if fit_days is None:
                fit_days = [load_day(path) for path in fit_paths]
            print(f'[{key}] Compute Model')
            results = cfg['fit'](fit_days)
            save_volume_model(results, model_path)
            print(f'[{key}] Saved to {model_path}')

        # ---- Valutazione out-of-sample sui giorni rimanenti ----
        if len(test_paths) > 0:
            if test_days is None:
                test_days = [load_day(path) for path in test_paths]
            oos = cfg['evaluate'](results, test_days)
            save_volume_oos_stats(results, oos, model_dir / 'oos_stats.json',
                                  window_info={'fitting_window': fitting_window,
                                               'n_fit_days': len(fit_paths),
                                               'n_test_days': len(test_paths),
                                               'trim_top': trim_top,
                                               'top_frac': top_frac})
        else:
            print(f'[{key}] No days left for out-of-sample evaluation')