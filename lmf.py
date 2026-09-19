import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
from scipy import stats
import warnings

def sample_metaorder_length(dist_type='pareto', alpha=1.5, **kwargs):
    """
    Genera la lunghezza L del metaordine in base alla distribuzione scelta.
    Garantisce sempre L >= 1.
    """
    dist_type = dist_type.lower()
    
    if dist_type == 'pareto':
        # Default: Pareto continua campionata e floorata a int
        return int(np.random.pareto(alpha) + 1)
    
    elif dist_type in ['zeta', 'zipf']:
        # Zeta / Zipf discreta pura: P(k) ~ k^-alpha
        return int(np.random.zipf(alpha))
    
    elif dist_type == 'yule':
        # Yule-Simon: asintoticamente k^-(alpha + 1), richiede parametro rho = alpha
        rho = kwargs.get('rho', alpha)
        return int(stats.yulesimon.rvs(rho))
    
    elif dist_type in ['logarithmic', 'logser']:
        # Logaritmica: p deve essere compreso tra 0 e 1 (default 0.8)
        p = kwargs.get('p', 0.8)
        return int(stats.logser.rvs(p))
    
    elif dist_type == 'lomax':
        # Lomax discreta (Pareto tipo II): P(k) ~ (1 + k/scale)^-alpha
        scale = kwargs.get('scale', 1.0)
        return int(np.floor(stats.lomax.rvs(alpha, scale=scale))) + 1
    
    elif dist_type in ['nbinom', 'negative_binomial']:
        # Binomiale Negativa traslata (+1) per supporto L >= 1
        n_param = kwargs.get('r', 1)
        p_param = kwargs.get('p', 0.1)
        return int(np.random.negative_binomial(n_param, p_param)) + 1
    
    else:
        raise ValueError(f"Distribuzione '{dist_type}' non supportata.")

def power_law(x, constant, alpha):
    return constant * x**alpha

def simulate_lmf(alpha, n_traders, total_steps, p_plus=0.5, dist_type='pareto', dist_kwargs=None):
    """
    Advanced simulation: pool of traders with overlapping metaorders.
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    trader_state = np.zeros((n_traders, 2), dtype=int)
    
    def get_new_metaorder():
        length = sample_metaorder_length(dist_type=dist_type, alpha=alpha, **dist_kwargs)
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

def simulate_lmf_lambda(alpha, lam, total_steps, p_plus=0.5, dist_type='pareto', dist_kwargs=None):
    """
    λ-model simulation di Lillo, Mike & Farmer (2005) con tracciamento dei metaordini.
    
    Parameters
    ----------
    alpha : float
        Esponente di coda della distribuzione dei metaordini (> 1).
    lam : float
        Probabilità di arrivo di un nuovo metaordine per timestep (0 < lam < 1).
    total_steps : int
        Numero di step temporali della simulazione.
    p_plus : float
        Probabilità che il metaordine sia di acquisto (+1).
    dist_type : str
        Distribuzione dei metaordini (default 'pareto'). Opzioni: 'pareto', 'zeta', 'yule', 'logarithmic', 'lomax', 'nbinom'.
    dist_kwargs : dict or None
        Parametri aggiuntivi per la distribuzione scelta.
        
    Returns
    -------
    order_flow : np.ndarray
        Serie temporale dei segni degli ordini eseguiti (+1 o -1).
    n_active : np.ndarray
        Numero di ordini attivi nel pool per ogni istante t.
    storico_metaordini : list of dict
        Registro di tutti i metaordini completati durante la simulazione.
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if alpha <= 1 and dist_type == 'pareto':
        raise ValueError("alpha deve essere > 1 affinché la media di Pareto sia finita.")
    if not (0 < lam < 1):
        raise ValueError("lam deve essere compreso nell'intervallo aperto (0, 1).")
 
    lam_c = (alpha - 1) / alpha if alpha > 1 else 0.5
    if lam >= lam_c:
        warnings.warn(
            f"lam={lam:.4f} >= lam_c={lam_c:.4f} (valore critico per alpha={alpha}). "
            "N(t) potrebbe crescere indefinitamente.", RuntimeWarning, stacklevel=2
        )
 
    # Generatore di ID univoci per i metaordini
    id_counter = 0

    def new_hidden_order():
        nonlocal id_counter
        length = sample_metaorder_length(dist_type=dist_type, alpha=alpha, **dist_kwargs)
        side = np.sign(np.random.rand() - (1 - p_plus))
        # Struttura: [segno, tracking_lunghezza_residua, id_univoco, lunghezza_iniziale, step_nascita]
        ordine = [side, length, id_counter, length, t_attore]
        id_counter += 1
        return ordine
 
    # Inizializzazione delle strutture dati
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
    dist_type='pareto',
    dist_kwargs=None
):
    """
    Run a grid study over multiple alpha values and numbers of traders.
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
            print(f"  Simulating alpha={alpha}, n_traders={n_traders} (dist={dist_type}) …")
            flow = simulate_lmf(alpha, n_traders, total_steps, dist_type=dist_type, dist_kwargs=dist_kwargs)
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
    dist_type='pareto',
    dist_kwargs=None
):
    """
    Run a grid study over multiple lambda values for the λ model.
    """
    lam_c = (alpha - 1) / alpha if alpha > 1 else 0.5
    lags  = np.arange(1, max_lag + 1)
    colors = ['blue', 'red', 'green', 'orange']
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    results = {}
 
    for color, lam in zip(colors, lambdas):
        print(f"  Simulating λ model: alpha={alpha}, lambda={lam} (dist={dist_type}) …")
        flow, n_active, _ = simulate_lmf_lambda(alpha, lam, total_steps, dist_type=dist_type, dist_kwargs=dist_kwargs)
 
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
    else:
        plt.savefig('images\\lmf_lambda_model.png')
 
    plt.close()
    return results

if __name__ == '__main__':
    # Esecuzione standard con distribuzione Pareto (comportamento originario):
    plot_lambda(1.5, [0.2, 0.3], total_steps=100_000_000)

    # Esempio di utilizzo con altre distribuzioni (opzionale):
    # plot_lambda(1.5, [0.2, 0.3], total_steps=10_000_000, dist_type='zeta')
    # plot_lambda(1.5, [0.2, 0.3], total_steps=10_000_000, dist_type='yule', dist_kwargs={'rho': 1.5})