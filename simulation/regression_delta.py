import numpy as np
import statsmodels.api as sm


def power_transform(v, delta):
    """Funzione di impatto sign(v) * |v|^delta.

    delta = 1   -> impatto lineare
    delta = 0   -> solo il segno del volume
    0<delta<1   -> impatto power-law concavo
    """
    return np.sign(v) * np.abs(v) ** delta


def _lags(x, p):
    """Matrice [x_{t-1}, ..., x_{t-p}] per t = p, ..., n-1."""
    n = len(x)
    return np.column_stack([x[p - i:n - i] for i in range(1, p + 1)])


def impact_fit(returns, volumes, delta=1.0, p=10):
    returns = np.asarray(returns, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if len(returns) != len(volumes):
        raise ValueError("returns e volumes devono avere la stessa lunghezza")

    # Modello volume: AR(p) lineare sui volumi originali
    X_vol = sm.add_constant(_lags(volumes, p))
    res_vol = sm.OLS(volumes[p:], X_vol).fit()

    # Modello rendimenti: regressione sui volumi trasformati [V_t, V_{t-1}, ..., V_{t-p}]
    v_t = power_transform(volumes[p:], delta)
    v_lags = power_transform(_lags(volumes, p), delta)
    X_ret = sm.add_constant(np.column_stack([v_t, v_lags]))
    res_ret = sm.OLS(returns[p:], X_ret).fit()

    return {
        "p": p,
        "delta": delta,
        "volume_model": {"params": res_vol.params, "sigma2": res_vol.mse_resid},
        "return_model": {"params": res_ret.params, "sigma2": res_ret.mse_resid},
    }


def impact_simulate(results, n_steps, initial_v, initial_price=100.0, seed=None):
    """initial_v: array cronologico (dal più vecchio al più recente) di lunghezza p."""
    rng = np.random.default_rng(seed)
    p = results["p"]
    delta = results["delta"]

    initial_v = np.asarray(initial_v, dtype=float)
    if initial_v.shape != (p,):
        raise ValueError(f"initial_v deve avere lunghezza p={p}")

    sim_v = np.zeros(n_steps + p)
    sim_r = np.zeros(n_steps + p)
    prices = np.zeros(n_steps + p)
    sim_v[:p] = initial_v
    prices[p - 1] = initial_price

    v_params = results["volume_model"]["params"]
    v_sigma = np.sqrt(results["volume_model"]["sigma2"])
    r_params = results["return_model"]["params"]
    r_sigma = np.sqrt(results["return_model"]["sigma2"])

    for t in range(p, n_steps + p):
        v_lagged = sim_v[t - p:t][::-1]  # V_{t-1}, ..., V_{t-p}

        # 1. Volume
        sim_v[t] = np.dot(np.concatenate([[1.0], v_lagged]), v_params) + rng.normal(0, v_sigma)

        # 2. Rendimento
        exog_r = np.concatenate([[1.0, power_transform(sim_v[t], delta)],
                                 power_transform(v_lagged, delta)])
        sim_r[t] = np.dot(exog_r, r_params) + rng.normal(0, r_sigma)

        # 3. Prezzo
        prices[t] = prices[t - 1] * (1 + sim_r[t])

    return prices[p:], sim_v[p:], sim_r[p:]