import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit

def power_law(x, constant, alpha):
    return constant * x**alpha

def simulate_lmf(alpha, n_traders, total_steps):
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

def simulate_lmf_lambda(alpha, lam, total_steps):
    """
    λ-model simulation: N(t) fluctuates via a Poisson-like arrival process.
 
    This implements the λ model of Lillo, Mike & Farmer (2005), Section II.
    Unlike the fixed-N model, the number of active hidden orders N(t) is not
    kept constant. Instead, at every timestep a new hidden order arrives with
    probability λ when N(t) > 0, or with probability 1 when N(t) = 0
    (ensuring the pool never empties permanently).
 
    Dynamics per timestep
    ---------------------
    1. Arrival step  – add a new hidden order with probability λ (or 1 if
       N(t) == 0).  Each new order gets a random sign ±1 and a length L drawn
       from the discrete Pareto P(L) ∝ L^{-(α+1)}, i.e. np.random.pareto(α)+1.
    2. Execution step – pick one of the N(t) active hidden orders uniformly at
       random; remove one unit from it, record its sign as the revealed order.
       If the order is now exhausted, remove it from the pool.
 
    The mean number of active orders E[N(t)] diverges as λ → λ_c from below,
    where the critical value is
 
        λ_c = (α − 1) / α                                          [Eq. 18]
 
    For λ < λ_c the pool is stable and the ACF of revealed order signs decays
    asymptotically as τ^{-(α-1)}, the same exponent as the fixed-N model.
 
    Parameters
    ----------
    alpha : float
        Pareto tail exponent of hidden-order lengths.  Must satisfy alpha > 1.
    lam : float
        Arrival probability per timestep when N(t) > 0.
        Must be in (0, 1); should be < λ_c = (alpha-1)/alpha for stability.
    total_steps : int
        Number of revealed orders (timesteps) to generate.
 
    Returns
    -------
    order_flow : np.ndarray, shape (total_steps,)
        Time series of revealed order signs (+1 buy, -1 sell).
    n_active : np.ndarray, shape (total_steps,)
        Number of active hidden orders recorded *after* each timestep.
        Useful for studying liquidity fluctuations (Section IV of the paper).
 
    Notes
    -----
    Critical threshold (Eq. 18):
        λ_c = (α − 1) / α
 
    For α = 1.5  →  λ_c ≈ 0.333
    For α = 1.3  →  λ_c ≈ 0.231
    For α = 1.7  →  λ_c ≈ 0.412
 
    If λ ≥ λ_c the pool may grow without bound; a RuntimeWarning is raised
    when N(t) exceeds 10 000 to alert the caller.
    """
    if alpha <= 1:
        raise ValueError("alpha must be > 1 for the Pareto mean to be finite.")
    if not (0 < lam < 1):
        raise ValueError("lam must be in the open interval (0, 1).")
 
    lam_c = (alpha - 1) / alpha
    if lam >= lam_c:
        import warnings
        warnings.warn(
            f"lam={lam:.4f} >= lam_c={lam_c:.4f} (critical value for alpha={alpha}). "
            "N(t) may grow without bound.", RuntimeWarning, stacklevel=2
        )
 
    def new_hidden_order():
        length = int(np.random.pareto(alpha) + 1)
        side = np.random.choice([1, -1])
        return [side, length]
 
    # Pool of hidden orders: list of [sign, remaining_length]
    # Start with a single hidden order to avoid an empty pool at t=0
    pool = [new_hidden_order()]
 
    order_flow = np.zeros(total_steps, dtype=np.int8)
    n_active   = np.zeros(total_steps, dtype=np.int32)
 
    warned_large = False
 
    for t in range(total_steps):
        n = len(pool)
 
        # --- Arrival step ---
        # With prob 1 if pool empty, else with prob lam
        if n == 0 or np.random.random() < (1.0 if n == 0 else lam):
            pool.append(new_hidden_order())
 
        # --- Execution step ---
        # Pick a random hidden order and execute one unit from it
        n = len(pool)
        idx = np.random.randint(0, n)
        side, remaining = pool[idx]
        order_flow[t] = side
        remaining -= 1
        if remaining <= 0:
            # Order fully executed: swap with last for O(1) removal
            pool[idx] = pool[-1]
            pool.pop()
        else:
            pool[idx][1] = remaining
 
        n_active[t] = len(pool)
 
        if not warned_large and n_active[t] > 10_000:
            import warnings
            warnings.warn(
                f"N(t) exceeded 10 000 at step {t}. "
                "Consider using lam < lam_c to keep the pool stable.",
                RuntimeWarning, stacklevel=2
            )
            warned_large = True
 
    return order_flow, n_active

def plot(
    alphas,
    n_traders_list,
    total_steps=100_000_000,
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
            flow = simulate_lmf(alpha, n_traders, total_steps)
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

def plot_lambda(
    alpha,
    lambdas,
    total_steps=100_000_000,
    max_lag=1000,
    fit_start_lag=25,
    save_path=None,
):
    """
    Run a grid study over multiple lambda values for the λ model.
 
    Produces two subplots:
      (a) ACF of revealed order signs for each λ, with theoretical slope.
      (b) ACF of N(t) (liquidity fluctuations) for each λ (Section IV).
 
    Parameters
    ----------
    alpha : float
        Pareto tail exponent.
    lambdas : list of float
        Arrival probabilities to compare (all should be < λ_c for stability).
    total_steps : int
        Length of the simulated time series.
    max_lag : int
        Maximum lag for ACF.
    fit_start_lag : int
        First lag used for power-law fitting.
    save_path : str or None
        If given, figure is saved here.
 
    Returns
    -------
    results : dict
        results[lam] = {'acf_flow': ..., 'acf_N': ..., 'n_active': ...}
    """
    lam_c = (alpha - 1) / alpha
    lags  = np.arange(1, max_lag + 1)
    colors = ['blue', 'red', 'green', 'orange']
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    results = {}
 
    for color, lam in zip(colors, lambdas):
        print(f"  Simulating λ model: alpha={alpha}, lambda={lam} (lambda_c={lam_c:.3f}) …")
        flow, n_active = simulate_lmf_lambda(alpha, lam, total_steps)
 
        acf_flow = acf(flow,     nlags=max_lag, fft=True)
        acf_N    = acf(n_active, nlags=max_lag, fft=True)
 
        results[lam] = {"acf_flow": acf_flow, "acf_N": acf_N, "n_active": n_active}
 
        label = rf"$\lambda={lam}$"
        ax1.loglog(lags, np.abs(acf_flow[1:]), color=color, label=label)
        ax2.loglog(lags, np.abs(acf_N[1:]),    color=color, label=label)
 
    # Theoretical slope τ^{-(α-1)} anchored at lag-1 of first simulation
    first_acf = results[lambdas[0]]["acf_flow"]
    theory = first_acf[1] * lags ** (-(alpha - 1))
    ax1.loglog(lags, theory, "k--", linewidth=1.5, label=rf"Theory $\tau^{{-{alpha-1:.2f}}}$")
    ax2.loglog(lags, theory, "k--", linewidth=1.5, label=rf"Theory $\tau^{{-{alpha-1:.2f}}}$")
 
    for ax, title in zip(
        [ax1, ax2],
        [r"ACF of order signs $x_t$", r"ACF of active orders $N(t)$"],
    ):
        ax.set_title(rf"{title}  ($\alpha={alpha}$, $\lambda_c={lam_c:.3f}$)")
        ax.set_xlabel(r"Lag $\tau$")
        ax.set_ylabel(r"ACF")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
 
    plt.savefig('images\\lmf_lambda_model.png')
    plt.close()
    return results

if __name__ == '__main__':
    # --- Study parameters ---
    """
    alphas        = [1.2, 1.5, 1.8]
    n_traders_list = [10, 20, 40]
    total_steps   = 10_000_000
    max_lag       = 1000

    results = plot(
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

    results = plot(
        alphas=alphas,
        n_traders_list=n_traders_list,
        total_steps=total_steps,
        max_lag=max_lag,
        fit_start_lag=50,
        save_path="images\\lmf_study_test.png",
    )
    """

    plot_lambda(1.5, [0.2, 0.3], total_steps=100_000_000)

