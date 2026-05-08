import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit

def power_law(x, constant, alpha):
    return constant * x**alpha

def simulate_lillo_farmer_multi_trader(alpha, n_traders, total_steps):
    """
    Advanced simulation: pool of traders with overlapping metaorders.
    """
    trader_state = np.zeros((n_traders, 2), dtype=int)
    
    def get_new_metaorder():
        length = int(np.random.pareto(alpha) + 1)
        side = np.random.choice([1, -1])
        return side, length

    for i in range(n_traders):
        trader_state[i] = get_new_metaorder()

    order_flow = np.zeros(total_steps)

    for t in range(total_steps):
        idx = np.random.randint(0, n_traders)
        side, remaining = trader_state[idx]
        order_flow[t] = side
        remaining -= 1
        if remaining <= 0:
            trader_state[idx] = get_new_metaorder()
        else:
            trader_state[idx, 1] = remaining
            
    return order_flow


def study_alpha_traders(
    alphas,
    n_traders_list,
    total_steps=5_000_000,
    max_lag=1000,
    fit_start_lag=25,
    save_path=None,
):
    """
    Run a grid study over multiple alpha values and numbers of traders.

    For each alpha, one subplot is produced showing the simulated ACF,
    the theoretical power-law decay, and a fitted power-law for every
    value of n_traders.

    Parameters
    ----------
    alphas : list of float
        Pareto exponents to study (one subplot per value).
    n_traders_list : list of int
        Pool sizes to compare within each subplot.
    total_steps : int
        Length of the order-flow time series for each simulation.
    max_lag : int
        Maximum lag used for ACF computation.
    fit_start_lag : int
        First lag included in the power-law fit (to avoid short-lag noise).
    save_path : str or None
        If given, the figure is saved to this path.

    Returns
    -------
    results : dict
        Nested dict  results[alpha][n_traders] = {'acf': ..., 'popt': ..., 'pcov': ...}
    """
    n_alphas = len(alphas)
    lags = np.arange(1, max_lag + 1)
    fit_slice = slice(fit_start_lag - 1, None)      # lags[fit_slice] starts at fit_start_lag
    colors = ['blue', 'red', 'green']

    fig, axes = plt.subplots(1, n_alphas, figsize=(7 * n_alphas, 6), sharey=False)
    if n_alphas == 1:
        axes = [axes]

    results = {}

    for ax, alpha in zip(axes, alphas):
        results[alpha] = {}
        theoretical_gamma = alpha - 1

        for color, n_traders in zip(colors, n_traders_list):
            print(f"  Simulating alpha={alpha}, n_traders={n_traders} …")
            flow = simulate_lillo_farmer_multi_trader(alpha, n_traders, total_steps)
            auto_corr = acf(flow, nlags=max_lag, fft=True)   # index 0 = lag-0

            # Power-law fit on lags >= fit_start_lag
            try:
                popt, pcov = curve_fit(
                    power_law,
                    lags[fit_slice],
                    auto_corr[1:][fit_slice],
                    p0=[auto_corr[1], -theoretical_gamma],
                    maxfev=5000,
                )
            except RuntimeError:
                popt, pcov = [np.nan, np.nan], np.full((2, 2), np.nan)
                print(f"    Fit did not converge for alpha={alpha}, n_traders={n_traders}")

            results[alpha][n_traders] = {
                "acf": auto_corr,
                "popt": popt,
                "pcov": pcov,
            }

            # Simulated ACF
            ax.loglog(
                lags,
                auto_corr[1:],
                color=color,
                label=f"$N={n_traders}$",
            )
            # Fitted power law (dashed)
            if not np.isnan(popt[0]):
                ax.loglog(
                    lags,
                    power_law(lags, n_traders**(alpha - 2) / alpha, -alpha + 1),
                    color=color,
                    linestyle="--",
                    linewidth=1,
                )

        # Theoretical decay anchored at lag-1 of the first simulation
        # first_acf = results[alpha][n_traders_list[0]]["acf"]
        # theoretical_decay = first_acf[1] * lags ** (-theoretical_gamma)
        # ax.loglog(
        #     lags,
        #     theoretical_decay,
        #     "k:",
        #     linewidth=2,
        #     label=r"Theory: $\tau^{-(\alpha-1)}$",
        # )

        ax.set_title(rf"$\alpha = {alpha}$  ($\gamma = \alpha-1 = {theoretical_gamma:.1f}$)")
        ax.set_xlabel(r"Lag $\tau$")
        ax.set_ylabel(r"ACF $C(\tau)$")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()
    return results


# --- Study parameters ---
alphas        = [1.2, 1.5, 1.8]
n_traders_list = [10, 20, 40]
total_steps   = 100_000_000
max_lag       = 1000

results = study_alpha_traders(
    alphas=alphas,
    n_traders_list=n_traders_list,
    total_steps=total_steps,
    max_lag=max_lag,
    fit_start_lag=50,
    save_path="images\\lmf_study.png",
)

alphas        = [1.5]
n_traders_list = [1, 5, 50]
total_steps   = 100_000_000
max_lag       = 1000

results = study_alpha_traders(
    alphas=alphas,
    n_traders_list=n_traders_list,
    total_steps=total_steps,
    max_lag=max_lag,
    fit_start_lag=50,
    save_path="images\\lmf_study_1.png",
)