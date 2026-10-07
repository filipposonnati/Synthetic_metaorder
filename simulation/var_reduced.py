import inspect

import numpy as np
from statsmodels.tsa.api import VAR

import matplotlib.pyplot as plt

# Palette di riferimento (superficie chiara): slot categorici 1 e 2
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e6e5e1"
_BLUE = "#2a78d6"    # volumi
_ORANGE = "#eb6834"  # rendimenti


def plot_residuals(residuals, names=("Volumi", "Rendimenti")):
    fig, axes = plt.subplots(1, len(names), figsize=(11, 4))

    for k, name in enumerate(names):
        e = residuals[:, k]
        # Rimuove eventuali NaN per evitare errori di calcolo
        e = e[~np.isnan(e)]
        
        mean = np.mean(e)
        sd = np.std(e, ddof=1)
        ax = axes[k]

        # 1. BINS DINAMICI: Usa la regola di Freedman-Diaconis o un numero fisso pulito
        ax.hist(e, bins='auto', density=True, alpha=0.65, edgecolor='white', color='#1f77b4')

        # 2. CURVA NORMALE MOLTO PIÙ FLUIDA (con la media reale dei residui)
        # Genera 500 punti ben distribuiti tra min e max reali
        x_min = min(e.min(), mean - 4 * sd)
        x_max = max(e.max(), mean + 4 * sd)
        x = np.linspace(x_min, x_max, 500)
        
        pdf = np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        ax.plot(x, pdf, color='red', lw=1.8, label='Normale')

        # 3. XLIM RAGIONEVOLE (±3.5 deviazioni standard dalla media)
        ax.set_xlim(mean - 3.5 * sd, mean + 3.5 * sd)

        # 4. RISOLUZIONE FORMATO ASSE X (Notazione scientifica + rotazione se serve)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
        ax.tick_params(axis='x', rotation=15, labelsize=9)

        ax.set_title(f"Distribuzione {name}", fontsize=11, pad=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    plt.show()

def var_reduced_fit(returns, volumes, p=10, plot=False):
    """Stima un VAR(p) in forma ridotta su [volumi, rendimenti], senza intercetta
    (come nella versione con lstsq). Con plot=True mostra il grafico dei residui;
    con save_path lo salva anche su file."""
    data = np.column_stack([volumes, returns])  # colonna 0 = volumi, colonna 1 = rendimenti
    fitted = VAR(data).fit(maxlags=p, trend="n")

    if plot:
        plot_residuals(np.asarray(fitted.resid))

    # R² per equazione: 1 - SSR/SST, con SST rispetto alla media di y.
    # Il modello è senza intercetta, quindi l'R² centrato può risultare negativo
    # se i regressori spiegano meno della sola media.
    resid = np.asarray(fitted.resid)
    y = np.asarray(fitted.endog)[fitted.k_ar:]          # variabile dipendente (senza i primi p valori)
    ssr = (resid ** 2).sum(axis=0)
    sst = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ssr / sst
    n_obs, n_reg = resid.shape[0], fitted.neqs * fitted.k_ar
    r2_adj = 1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - n_reg)

    print(f"R² volumi: {r2[0]:.4f} (adj {r2_adj[0]:.4f}) | "
          f"R² rendimenti: {r2[1]:.4f} (adj {r2_adj[1]:.4f})")

    return {
        "p": p,
        "model": fitted,                 # oggetto VARResults di statsmodels
        "sigma_cov": fitted.sigma_u,     # covarianza dei residui
        "r2": {"volumi": r2[0], "rendimenti": r2[1]},
        "r2_adj": {"volumi": r2_adj[0], "rendimenti": r2_adj[1]},
    }
def simulate_var_reduced(results, n_steps, initial_v, initial_r,
                         initial_price=100.0, bootstrap=False,
                         center_residuals=True, seed=None):
    fitted = results["model"]
    p = results["p"]
    coefs = fitted.coefs                      # (p, k, k)
    k = coefs.shape[1]
    rng = np.random.default_rng(seed)

    # shock: gaussiani con covarianza sigma_u, oppure residui ricampionati
    if bootstrap:
        resid = np.asarray(fitted.resid, dtype=np.float64)
        resid = resid[~np.isnan(resid).any(axis=1)]
        if center_residuals:
            resid = resid - resid.mean(axis=0)
        shocks = resid[rng.integers(0, len(resid), size=n_steps)]
    else:
        shocks = rng.multivariate_normal(np.zeros(k), fitted.sigma_u, size=n_steps)

    # ricorsione VAR (senza intercetta)
    y = np.empty((p + n_steps, k))
    y[:p] = np.column_stack([initial_v, initial_r])[-p:]
    for t in range(p, p + n_steps):
        y[t] = np.einsum("ikj,ij->k", coefs, y[t - p:t][::-1]) + shocks[t - p]

    sim_v, sim_r = y[p:, 0], y[p:, 1]
    prices = initial_price * np.concatenate([[1.0], np.cumprod(1 + sim_r[:-1])])
    return prices, sim_v, sim_r