import numpy as np
from statsmodels.tsa.api import VAR


def var_reduced_fit(returns, volumes, p=10):
    """Stima un VAR(p) in forma ridotta su [volumi, rendimenti], senza intercetta
    (come nella versione con lstsq)."""
    data = np.column_stack([volumes, returns])  # colonna 0 = volumi, colonna 1 = rendimenti
    fitted = VAR(data).fit(maxlags=p, trend="n")

    return {
        "p": p,
        "model": fitted,                 # oggetto VARResults di statsmodels
        "sigma_cov": fitted.sigma_u,     # covarianza dei residui
    }


def simulate_var_reduced(results, n_steps, initial_v, initial_r,
                         initial_price=100.0, rng=None):
    """Simula il VAR stimato. initial_v e initial_r devono contenere gli ultimi p
    valori in ordine cronologico (dal più vecchio al più recente)."""
    fitted = results["model"]

    initial_values = np.column_stack([initial_v, initial_r]).astype(np.float64)

    sim = fitted.simulate_var(
        steps=n_steps,
        initial_values=initial_values,
        rng=rng,
    )

    sim_v, sim_r = sim[:, 0], sim[:, 1]
    prices = initial_price * np.cumprod(1 + sim_r)

    return prices, sim_v, sim_r