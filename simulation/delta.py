import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize


def power_transform(v, delta):
    """Applica la trasformazione sign(v) * |v|^delta."""
    return np.sign(v) * (np.abs(v) ** delta)


def delta_fit_fixed(returns, volumes, delta=0.5, p=10):
    n = len(returns)
    y_ret = returns[p:]

    # Trasformazione dei volumi con il delta scelto
    v_curr = volumes[p:]
    v_curr_transf = np.sign(v_curr) * (np.abs(v_curr) ** delta)

    v_lags = np.column_stack([volumes[p-i:n-i] for i in range(1, p+1)])
    v_lags_transf = np.sign(v_lags) * (np.abs(v_lags) ** delta)

    # Creazione della matrice X: [V_curr^delta, V_{t-1}^delta, ..., V_{t-p}^delta]
    X_ret = np.column_stack([v_curr_transf, v_lags_transf])

    # Fit lineare
    res_ret = sm.OLS(y_ret, X_ret).fit()

    # Fit volumi (rimane lineare sui volumi originali)
    X_vol_final = np.column_stack([volumes[p-i:n-i] for i in range(1, p+1)])
    res_vol = sm.OLS(volumes[p:], X_vol_final).fit()

    return {
        "p": p,
        "delta": delta,
        "volume_model": {
            "params": res_vol.params,
            "sigma2": np.var(res_vol.resid)
        },
        "return_model": {
            "params": res_ret.params,  # [gamma, beta_1...beta_p]
            "sigma2": np.var(res_ret.resid)
        }
    }


def simulate_delta_fixed(results, n_steps, initial_v, initial_r, initial_price=100.0):
    p = results['p']
    delta = results['delta']

    # Inizializziamo con i dati reali invece di zeri
    sim_v = np.zeros(n_steps + p)
    sim_r = np.zeros(n_steps + p)
    prices = np.zeros(n_steps + p)

    # Inseriamo i dati reali nei primi 'p' slot
    sim_v[:p] = initial_v
    sim_r[:p] = initial_r
    prices[p-1] = initial_price

    v_params = results['volume_model']['params']
    v_sigma = np.sqrt(results['volume_model']['sigma2'])
    r_params = results['return_model']['params']
    r_sigma = np.sqrt(results['return_model']['sigma2'])

    for t in range(p, n_steps + p):
        # 1. Simulazione Volume (AR lineare)
        v_lagged = sim_v[t-p:t][::-1]
        sim_v[t] = np.dot(v_lagged, v_params) + np.random.normal(0, v_sigma)

        # 2. Simulazione Return (Power Law Impact)
        v_curr = sim_v[t]
        v_curr_transf = np.sign(v_curr) * (np.abs(v_curr) ** delta)
        v_lags_transf = np.sign(v_lagged) * (np.abs(v_lagged) ** delta)

        exog_r = np.concatenate([[v_curr_transf], v_lags_transf])
        sim_r[t] = np.dot(exog_r, r_params) + np.random.normal(0, r_sigma)

        # 3. Aggiornamento Prezzo
        prices[t] = prices[t-1] * (1 + sim_r[t])

    # Restituiamo solo la parte simulata (tagliando i primi p valori reali)
    return prices[p:], sim_v[p:], sim_r[p:]