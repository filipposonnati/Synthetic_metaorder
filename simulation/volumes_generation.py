"""
volumes_generation.py
======================
Modulo dedicato alla calibrazione e alla simulazione dei volumi di mercato.
Supporta:
  - Modello AR(p) sui log-volumi (fit giorno per giorno)
  - Modello MEM/ACD(p,q) con diverse distribuzioni (fit giorno per giorno)
  - Campionamento Empirico / Bootstrap direttamente dai dati reali
"""

import numpy as np
from scipy.optimize import minimize
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
    for t in range(n_steps):
        buf[p + t] = buf[t : t + p][::-1] @ phi + noise[t]
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