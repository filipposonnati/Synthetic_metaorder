import numpy as np
import statsmodels.api as sm


def var_fit(returns, volumes, p=10):
    n = len(returns)
    v_lags_f = np.column_stack([volumes[p-i:n-i] for i in range(1, p+1)])
    r_lags_f = np.column_stack([returns[p-i:n-i] for i in range(1, p+1)])
    y_vol_f = volumes[p:]; y_ret_f = returns[p:]; v_curr_f = volumes[p:]

    # No add_constant
    X_vol = np.hstack([v_lags_f, r_lags_f])
    res_vol = sm.OLS(y_vol_f, X_vol).fit()

    X_ret = np.column_stack([v_curr_f, v_lags_f, r_lags_f])
    res_ret = sm.OLS(y_ret_f, X_ret).fit()

    return {
        "p": p,
        "volume_model": {
            "params": res_vol.params,
            "sigma2": np.var(res_vol.resid),
            "labels": [f"V_l{i}" for i in range(1, p+1)] + [f"R_l{i}" for i in range(1, p+1)]
        },
        "return_model": {
            "params": res_ret.params,
            "sigma2": np.var(res_ret.resid),
            "labels": ["V_curr"] + [f"V_l{i}" for i in range(1, p+1)] + [f"R_l{i}" for i in range(1, p+1)]
        }
    }


def simulate_var(results, n_steps, initial_v, initial_r, initial_price=100.0):
    p = results['p']
    sim_v = np.zeros(n_steps + p)
    sim_r = np.zeros(n_steps + p)
    prices = np.zeros(n_steps + p)

    sim_v[:p] = initial_v
    sim_r[:p] = initial_r
    prices[p-1] = initial_price

    v_params = results['volume_model']['params']
    v_sigma = np.sqrt(results['volume_model']['sigma2'])
    r_params = results['return_model']['params']
    r_sigma = np.sqrt(results['return_model']['sigma2'])

    for t in range(p, n_steps + p):
        v_lagged = sim_v[t-p:t][::-1]
        r_lagged = sim_r[t-p:t][::-1]

        exog_v = np.concatenate([v_lagged, r_lagged])
        sim_v[t] = np.dot(exog_v, v_params) + np.random.normal(0, v_sigma)

        v_curr = sim_v[t]
        exog_r = np.concatenate([[v_curr], v_lagged, r_lagged])
        sim_r[t] = np.dot(exog_r, r_params) + np.random.normal(0, r_sigma)
        prices[t] = prices[t-1] * (1 + sim_r[t])

    return prices[p:], sim_v[p:], sim_r[p:]