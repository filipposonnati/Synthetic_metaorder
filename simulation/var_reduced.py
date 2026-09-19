import numpy as np
from scipy.linalg import hankel
from numba import njit


def _build_lags(v, r, p):
    n = len(v)
    v_lags = hankel(v[:n-p], v[n-p-1:n-1])[:, ::-1]
    r_lags = hankel(r[:n-p], r[n-p-1:n-1])[:, ::-1]
    return np.hstack([v_lags, r_lags])


def var_reduced_fit(returns, volumes, p=10):
    n = len(returns)
    X = _build_lags(volumes, returns, p)

    y_vol = volumes[p:]
    y_ret = returns[p:]

    params_v, _, _, _ = np.linalg.lstsq(X, y_vol, rcond=None)
    params_r, _, _, _ = np.linalg.lstsq(X, y_ret, rcond=None)

    res_v = y_vol - X @ params_v
    res_r = y_ret - X @ params_r

    residuals = np.column_stack([res_v, res_r])
    sigma_cov = np.cov(residuals, rowvar=False)

    return {
        "p": p,
        "v_params": params_v,
        "r_params": params_r,
        "sigma_cov": sigma_cov
    }


# Funzione di simulazione compilata in C con Numba
@njit(fastmath=True)
def _simulate_loop_numba(n_steps, p, v_params, r_params, shocks, initial_v, initial_r):
    sim_v = np.empty(n_steps + p, dtype=np.float64)
    sim_r = np.empty(n_steps + p, dtype=np.float64)

    sim_v[:p] = initial_v
    sim_r[:p] = initial_r

    v_coef_v = v_params[:p]
    v_coef_r = v_params[p:]
    r_coef_v = r_params[:p]
    r_coef_r = r_params[p:]

    for idx in range(n_steps):
        t = idx + p

        dot_vv = 0.0
        dot_vr = 0.0
        dot_rv = 0.0
        dot_rr = 0.0

        for i in range(p):
            v_val = sim_v[t - 1 - i]
            r_val = sim_r[t - 1 - i]

            dot_vv += v_val * v_coef_v[i]
            dot_vr += r_val * v_coef_r[i]
            dot_rv += v_val * r_coef_v[i]
            dot_rr += r_val * r_coef_r[i]

        u_v = shocks[idx, 0]
        u_r = shocks[idx, 1]

        sim_v[t] = dot_vv + dot_vr + u_v
        sim_r[t] = dot_rv + dot_rr + u_r

    return sim_v[p:], sim_r[p:]


def simulate_var_reduced(results, n_steps, initial_v, initial_r, initial_price=100.0):
    p = results['p']

    # Estrazione congiunta degli shock correlati
    shocks = np.random.multivariate_normal([0, 0], results['sigma_cov'], size=n_steps)

    # Esecuzione del loop compilato JIT
    sim_v, sim_r = _simulate_loop_numba(
        n_steps,
        p,
        results['v_params'],
        results['r_params'],
        shocks,
        np.ascontiguousarray(initial_v, dtype=np.float64),
        np.ascontiguousarray(initial_r, dtype=np.float64)
    )

    prices = initial_price * np.cumprod(1 + sim_r)

    return prices, sim_v, sim_r