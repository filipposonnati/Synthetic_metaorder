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
        return int(np.random.pareto(alpha) + 1)
    
    elif dist_type in ['zeta', 'zipf']:
        return int(np.random.zipf(alpha))
    
    elif dist_type == 'yule':
        rho = kwargs.get('rho', alpha)
        return int(stats.yulesimon.rvs(rho))
    
    elif dist_type in ['logarithmic', 'logser']:
        p = kwargs.get('p', 0.8)
        return int(stats.logser.rvs(p))
    
    elif dist_type == 'lomax':
        scale = kwargs.get('scale', 1.0)
        return int(np.floor(stats.lomax.rvs(alpha, scale=scale))) + 1
    
    elif dist_type in ['nbinom', 'negative_binomial']:
        n_param = kwargs.get('r', 1)
        p_param = kwargs.get('p', 0.1)
        return int(np.random.negative_binomial(n_param, p_param)) + 1
    
    else:
        raise ValueError(f"Distribuzione '{dist_type}' non supportata.")

def power_law(x, constant, alpha):
    return constant * x**alpha

def simulate_lmf(alpha, n_traders, total_steps, p_plus=0.5, dist_type='pareto', dist_kwargs=None, p_trade_random=0.0):
    """
    Advanced simulation: pool of traders with overlapping metaorders.
    
    p_trade_random : float, opzionale
        Probabilità (tra 0.0 e 1.0) che la transazione corrente sia rumore con segno casuale
        anziché provenire da un metaordine.
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
        # Con probabilità p_trade_random viene generata una transazione totalmente casuale
        if p_trade_random > 0.0 and np.random.random() < p_trade_random:
            order_flow[t] = np.sign(np.random.rand() - (1 - p_plus))
        else:
            idx = np.random.randint(0, n_traders)
            side, remaining = trader_state[idx]
            order_flow[t] = side
            remaining -= 1
            if remaining <= 0:
                trader_state[idx] = get_new_metaorder()
            else:
                trader_state[idx, 1] = remaining
            
    return order_flow

def simulate_lmf_lambda(alpha, lam, total_steps, p_plus=0.5, dist_type='pareto', dist_kwargs=None, p_trade_random=0.0):
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
    p_trade_random : float, optional (default=0.0)
        Probabilità che al tempo t venga eseguita una transazione dal segno casuale 
        anziché consumare una quota di un metaordine attivo.
        
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
        # Verifica se generare un trade dal segno casuale (noise trade)
        if p_trade_random > 0.0 and np.random.random() < p_trade_random:
            order_flow[t] = np.sign(np.random.rand() - (1 - p_plus))
        else:
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

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
import warnings

def plot(
    alphas,
    n_traders_list,
    total_steps=10_000_000,
    max_lag=1000,
    fit_start_lag=25,
    save_path=None,
    dist_type='pareto',
    dist_kwargs=None,
    p_trade_random=0.0
):
    """
    Run specific paired simulations for fixed trader pool model:
    Run i uses (alphas[i], n_traders_list[i], p_trade_random[i]).
    Plots a theoretical power-law curve for each unique alpha value.
    """
    # Normalize inputs to lists of equal length
    alpha_list = [alphas] if isinstance(alphas, (float, int)) else list(alphas)
    trader_list = [n_traders_list] if isinstance(n_traders_list, (float, int)) else list(n_traders_list)
    p_noise_list = [p_trade_random] if isinstance(p_trade_random, (float, int)) else list(p_trade_random)

    n_runs = max(len(alpha_list), len(trader_list), len(p_noise_list))

    # Pad lists if shorter than n_runs by repeating the last element
    alpha_list += [alpha_list[-1]] * (n_runs - len(alpha_list))
    trader_list += [trader_list[-1]] * (n_runs - len(trader_list))
    p_noise_list += [p_noise_list[-1]] * (n_runs - len(p_noise_list))

    lags = np.arange(1, max_lag + 1)
    fit_slice = slice(fit_start_lag - 1, None)
    
    #colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    fig, ax = plt.subplots(figsize=(8, 6))
    results = []

    for i in range(n_runs):
        alpha = alpha_list[i]
        n_traders = trader_list[i]
        p_noise = p_noise_list[i]
        theoretical_gamma = alpha - 1

        #color = colors[i % len(colors)]
        #ls = linestyles[i % len(linestyles)]

        print(f"  Run {i+1}/{n_runs}: alpha={alpha}, N={n_traders}, p_noise={p_noise} (dist={dist_type}) …")
        
        flow = simulate_lmf(
            alpha, 
            n_traders, 
            total_steps, 
            dist_type=dist_type, 
            dist_kwargs=dist_kwargs, 
            p_trade_random=p_noise
        )
        auto_corr = acf(flow, nlags=max_lag, fft=True)

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
            print(f"    Fit did not converge for Run {i+1}")

        results.append({
            "alpha": alpha,
            "n_traders": n_traders,
            "p_noise": p_noise,
            "acf": auto_corr,
            "popt": popt,
            "pcov": pcov,
        })

        label = rf"Run {i+1}: $\alpha={alpha}$, $N={n_traders}$, $p_{{noise}}={p_noise}$"
        ax.loglog(lags, auto_corr[1:], linestyle='-', label=label)

    # Plot theoretical power-law curve once for each unique alpha
    unique_alphas = set(alpha_list)
    for alpha_val in unique_alphas:
        # Find the first run matching this alpha to scale the theoretical curve height
        first_match_idx = alpha_list.index(alpha_val)
        ref_acf = results[first_match_idx]["acf"]
        theory = ref_acf[1] * lags ** (-(alpha_val - 1))
        ax.loglog(
            lags, 
            theory, 
            color='black', 
            linestyle='--', 
            linewidth=1.2, 
            label=rf"$\tau^{{-{alpha_val-1:.2f}}}$"
        )

    #ax.set_title("Order Sign ACF (Paired Simulation Runs)")
    ax.set_xlabel(r"Lag $\tau$")
    ax.set_ylabel(r"ACF $C(\tau)$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    #plt.show()
    plt.close()
    return results


def plot_lambda(
    alphas,
    lambdas,
    total_steps=10_000_000,
    max_lag=1000,
    fit_start_lag=25,
    save_path=None,
    dist_type='pareto',
    dist_kwargs=None,
    p_trade_random=0.0
):
    """
    Run specific paired simulations for the λ-model:
    Run i uses (alphas[i], lambdas[i], p_trade_random[i]).
    Plots a theoretical power-law curve for each unique alpha value.
    """
    # Normalize inputs to lists of equal length
    alpha_list = [alphas] if isinstance(alphas, (float, int)) else list(alphas)
    lambda_list = [lambdas] if isinstance(lambdas, (float, int)) else list(lambdas)
    p_noise_list = [p_trade_random] if isinstance(p_trade_random, (float, int)) else list(p_trade_random)

    n_runs = max(len(alpha_list), len(lambda_list), len(p_noise_list))

    # Pad lists if shorter than n_runs by repeating the last element
    alpha_list += [alpha_list[-1]] * (n_runs - len(alpha_list))
    lambda_list += [lambda_list[-1]] * (n_runs - len(lambda_list))
    p_noise_list += [p_noise_list[-1]] * (n_runs - len(p_noise_list))

    lags = np.arange(1, max_lag + 1)
    #colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    #linestyles = ['-', '--', ':', '-.']

    fig, ax = plt.subplots(figsize=(8, 6))
    results = []

    for i in range(n_runs):
        alpha = alpha_list[i]
        lam = lambda_list[i]
        p_noise = p_noise_list[i]

        #color = colors[i % len(colors)]
        #ls = linestyles[i % len(linestyles)]

        print(f"  Run {i+1}/{n_runs}: alpha={alpha}, lambda={lam}, p_noise={p_noise} (dist={dist_type}) …")
        
        flow, n_active, historico = simulate_lmf_lambda(
            alpha, 
            lam, 
            total_steps, 
            dist_type=dist_type, 
            dist_kwargs=dist_kwargs, 
            p_trade_random=p_noise
        )

        acf_flow = acf(flow, nlags=max_lag, fft=True)
        
        results.append({
            "alpha": alpha,
            "lambda": lam,
            "p_noise": p_noise,
            "acf_flow": acf_flow,
            "n_active": n_active,
            "historico": historico
        })

        label = rf"Run {i+1}: $\alpha={alpha}$, $\lambda={lam}$, $p_{{noise}}={p_noise}$"
        ax.loglog(lags, np.abs(acf_flow[1:]), linestyle='-', label=label)

    # Plot theoretical power-law curve once for each unique alpha
    unique_alphas = set(alpha_list)
    for alpha_val in unique_alphas:
        # Find the first run matching this alpha to scale the theoretical curve height
        first_match_idx = alpha_list.index(alpha_val)
        ref_acf = results[first_match_idx]["acf_flow"]
        theory = ref_acf[1] * lags ** (-(alpha_val - 1))
        ax.loglog(
            lags, 
            theory, 
            color='black', 
            linestyle='--', 
            linewidth=1.2, 
            label=rf"$\tau^{{-{alpha_val-1:.2f}}}$"
        )

    #ax.set_title(r"Order Sign ACF in $\lambda$-Model (Paired Runs)")
    ax.set_xlabel(r"Lag $\tau$")
    ax.set_ylabel(r"ACF $C(\tau)$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    #plt.show()
    plt.close()
    return results

if __name__ == '__main__':
    plot_lambda(
        alphas=[1.3, 1.5, 1.5, 1.5], 
        lambdas=[0.2, 0.3, 0.3, 0.2], 
        total_steps=10_000_000, 
        p_trade_random=[0.0, 0.0, 0.1, 0.0],
        save_path='images/lmf_lambda_comparison.png'
    )

    plot(
        alphas=[1.5, 1.5, 1.5, 1.5, 1.3], 
        n_traders_list=[1, 10, 50, 10, 10], 
        total_steps=10_000_000, 
        p_trade_random=[0.0, 0.0, 0.0, 0.1, 0.0],
        save_path='images/lmf_comparison.png'
    )