import numpy as np
import statsmodels.api as sm


def var_fit(returns, volumes, p=10, constant=True):
    """Stima del VAR strutturale (ordinamento di Cholesky: volume -> rendimento).

    Equazione del volume:     V_t = c_v + sum_i a_i V_{t-i} + sum_i b_i R_{t-i} + e_v
    Equazione dei rendimenti: R_t = c_r + g V_t + sum_i c_i V_{t-i} + sum_i d_i R_{t-i} + e_r
    """
    returns = np.asarray(returns, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    n = len(returns)
    v_lags_f = np.column_stack([volumes[p-i:n-i] for i in range(1, p+1)])
    r_lags_f = np.column_stack([returns[p-i:n-i] for i in range(1, p+1)])
    y_vol_f = volumes[p:]
    y_ret_f = returns[p:]
    v_curr_f = volumes[p:]

    X_vol = np.hstack([v_lags_f, r_lags_f])
    X_ret = np.column_stack([v_curr_f, v_lags_f, r_lags_f])
    if constant:
        # La costante e' sempre la PRIMA colonna / il PRIMO coefficiente
        X_vol = sm.add_constant(X_vol, prepend=True, has_constant="add")
        X_ret = sm.add_constant(X_ret, prepend=True, has_constant="add")

    res_vol = sm.OLS(y_vol_f, X_vol).fit()
    res_ret = sm.OLS(y_ret_f, X_ret).fit()

    const_label = ["const"] if constant else []
    lags_v = [f"V_l{i}" for i in range(1, p+1)]
    lags_r = [f"R_l{i}" for i in range(1, p+1)]

    return {
        "p": p,
        "constant": constant,
        "volume_model": {
            "params": res_vol.params,
            "bse": res_vol.bse,
            # res.scale = SSR / (N - k): stimatore corretto per i gradi di liberta'
            "sigma2": res_vol.scale,
            "labels": const_label + lags_v + lags_r,
        },
        "return_model": {
            "params": res_ret.params,
            "bse": res_ret.bse,
            "sigma2": res_ret.scale,
            "labels": const_label + ["V_curr"] + lags_v + lags_r,
        },
    }


def simulate_var(results, n_steps, initial_v, initial_r, initial_price=100.0,
                 rng=None, log_returns=False):
    """Simula il VAR strutturale stimato da var_fit.

    initial_v, initial_r: lunghezza p, in ordine cronologico (il piu' vecchio per primo).
    log_returns: se True i rendimenti sono log-rendimenti e P_t = P_{t-1} * exp(R_t),
                 altrimenti rendimenti semplici e P_t = P_{t-1} * (1 + R_t).
    """
    if rng is None:
        rng = np.random.default_rng()

    p = results['p']
    constant = results.get('constant', False)
    initial_v = np.asarray(initial_v, dtype=float)
    initial_r = np.asarray(initial_r, dtype=float)
    if len(initial_v) != p or len(initial_r) != p:
        raise ValueError(f"initial_v e initial_r devono avere lunghezza p={p}")

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
    c = np.array([1.0]) if constant else np.array([])

    for t in range(p, n_steps + p):
        v_lagged = sim_v[t-p:t][::-1]
        r_lagged = sim_r[t-p:t][::-1]

        exog_v = np.concatenate([c, v_lagged, r_lagged])
        sim_v[t] = np.dot(exog_v, v_params) + rng.normal(0, v_sigma)

        exog_r = np.concatenate([c, [sim_v[t]], v_lagged, r_lagged])
        sim_r[t] = np.dot(exog_r, r_params) + rng.normal(0, r_sigma)

        if log_returns:
            prices[t] = prices[t-1] * np.exp(sim_r[t])
        else:
            prices[t] = prices[t-1] * (1 + sim_r[t])

    return prices[p:], sim_v[p:], sim_r[p:]