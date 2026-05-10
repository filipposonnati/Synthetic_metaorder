import numpy as np

def gaussian_fit(returns, volumes):
    """
    Fits volumes and returns assuming both are pure Gaussian (white noise) processes.
    
    No AR structure is estimated — volumes and returns are modelled purely by their
    empirical mean and variance, so the 'fit' simply captures the first two moments
    of each series.

    Parameters
    ----------
    returns : array-like, shape (n,)
        Return series.
    volumes : array-like, shape (n,)
        Volume series.

    Returns
    -------
    dict with keys:
        p            – always 0 (no lag structure)
        volume_model – {"mean": float, "sigma2": float}
        return_model – {"mean": float, "sigma2": float}
    """
    return {
        "p": 0,
        "volume_model": {
            "mean":   float(np.mean(volumes)),
            "sigma2": float(np.var(volumes)),
        },
        "return_model": {
            "mean":   float(np.mean(returns)),
            "sigma2": float(np.var(returns)),
        },
    }


def simulate_gaussian(results, n_steps, initial_v, initial_r, initial_price=100.0):
    """
    Simulates volumes and prices by drawing i.i.d. Gaussian noise.

    Each step:
      V_t ~ N(mu_v, sigma2_v)
      R_t ~ N(mu_r, sigma2_r)
      P_t  = P_{t-1} * (1 + R_t)

    Parameters
    ----------
    results       : dict returned by gaussian_fit()
    n_steps       : int   – number of steps to simulate
    initial_v     : ignored (kept for API consistency with the other simulate functions)
    initial_r     : ignored (kept for API consistency)
    initial_price : float – starting price level

    Returns
    -------
    prices   : ndarray, shape (n_steps,)
    sim_v    : ndarray, shape (n_steps,)
    sim_r    : ndarray, shape (n_steps,)
    """
    mu_v    = results["volume_model"]["mean"]
    sigma_v = np.sqrt(results["volume_model"]["sigma2"])
    mu_r    = results["return_model"]["mean"]
    sigma_r = np.sqrt(results["return_model"]["sigma2"])

    sim_v = np.random.normal(mu_v, sigma_v, size=n_steps)
    sim_r = np.random.normal(mu_r, sigma_r, size=n_steps)

    prices = np.zeros(n_steps)
    prices[0] = initial_price * (1 + sim_r[0])
    for t in range(1, n_steps):
        prices[t] = prices[t - 1] * (1 + sim_r[t])

    return prices, sim_v, sim_r