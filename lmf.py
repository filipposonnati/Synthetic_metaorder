import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
import warnings

def power_law(x, constant, alpha):
    return constant * x**alpha

def simulate_lmf(alpha, n_traders, total_steps, p_plus = 0.5):
    """
    Advanced simulation: pool of traders with overlapping metaorders.
    """
    trader_state = np.zeros((n_traders, 2), dtype=int)
    
    def get_new_metaorder():
        length = int(np.random.pareto(alpha) + 1)
        side = np.sign(np.random.rand() - (1 - p_plus))
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

def simulate_lmf_lambda(alpha, lam, total_steps, p_plus = 0.5):
    """
    λ-model simulation di Lillo, Mike & Farmer (2005) con tracciamento dei metaordini.
    
    Parameters
    ----------
    alpha : float
        Esponente di coda della distribuzione di Pareto dei metaordini (> 1).
    lam : float
        Probabilità di arrivo di un nuovo metaordine per timestep (0 < lam < 1).
    total_steps : int
        Numero di step temporali della simulazione.
        
    Returns
    -------
    order_flow : np.ndarray
        Serie temporale dei segni degli ordini eseguiti (+1 o -1).
    n_active : np.ndarray
        Numero di ordini attivi nel pool per ogni istante t.
    storico_metaordini : list of dict
        Registro di tutti i metaordini completati durante la simulazione.
        Ogni dizionario contiene:
          - 'id': identificativo univoco dell'ordine.
          - 'lunghezza_iniziale': numero di esecuzioni necessarie alla nascita (L).
          - 'step_creazione': l'istante t in cui l'ordine è entrato nel pool.
          - 'step_completamento': l'istante t in cui l'ordine è stato esaurito.
          - 'lifetime_effettivo': durata totale dell'ordine in passi di clock globali.
    """
    if alpha <= 1:
        raise ValueError("alpha deve essere > 1 affinché la media di Pareto sia finita.")
    if not (0 < lam < 1):
        raise ValueError("lam deve essere compreso nell'intervallo aperto (0, 1).")
 
    lam_c = (alpha - 1) / alpha
    if lam >= lam_c:
        warnings.warn(
            f"lam={lam:.4f} >= lam_c={lam_c:.4f} (valore critico per alpha={alpha}). "
            "N(t) potrebbe crescere indefinitamente.", RuntimeWarning, stacklevel=2
        )
 
    # Generatore di ID univoci per i metaordini
    id_counter = 0

    def new_hidden_order():
        nonlocal id_counter
        length = int(np.random.pareto(alpha) + 1)
        side = np.sign(np.random.rand() - (1 - p_plus))
        # Struttura: [segno, tracking_lunghezza_residua, id_univoco, lunghezza_iniziale, step_nascita]
        ordine = [side, length, id_counter, length, t_attore]
        id_counter += 1
        return ordine
 
    # Inizializzazione delle strutture dati
    # Per permettere la registrazione corretta dello step di nascita al tempo t=0
    t_attore = 0 
    pool = [new_hidden_order()]
 
    order_flow = np.zeros(total_steps, dtype=np.int8)
    n_active   = np.zeros(total_steps, dtype=np.int32)
    
    # Registro in cui salveremo i dati dei metaordini conclusi
    storico_metaordini = []
    warned_large = False
 
    for t in range(total_steps):
        t_attore = t  # Aggiorna il tempo corrente per la funzione di creazione
        n = len(pool)
 
        # --- 1. Arrival step ---
        if n == 0 or np.random.random() < (1.0 if n == 0 else lam):
            pool.append(new_hidden_order())
 
        # --- 2. Execution step ---
        n = len(pool)
        idx = np.random.randint(0, n)
        
        # Estraiamo i dati dell'ordine selezionato
        side, remaining, o_id, L_init, t_birth = pool[idx]
        
        order_flow[t] = side
        remaining -= 1
        
        if remaining <= 0:
            # L'ordine è stato interamente eseguito. Registriamo le sue metriche.
            storico_metaordini.append({
                "id": o_id,
                "lunghezza_iniziale": L_init,
                "step_creazione": t_birth,
                "step_completamento": t,
                "lifetime_effettivo": t - t_birth + 1
            })
            # O(1) Rimozione dal pool (swap con l'ultimo elemento e pop)
            pool[idx] = pool[-1]
            pool.pop()
        else:
            # Aggiorna solo la lunghezza rimanente dell'ordine nel pool
            pool[idx][1] = remaining
 
        n_active[t] = len(pool)
 
        if not warned_large and n_active[t] > 10_000:
            warnings.warn(
                f"N(t) ha superato 10 000 allo step {t}. Il sistema potrebbe essere instabile.",
                RuntimeWarning, stacklevel=2
            )
            warned_large = True
 
    return order_flow, n_active, storico_metaordini

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

