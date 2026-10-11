import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import json
from pathlib import Path
from os import listdir
import os

from utils import clear_data, open_data, save_simulated_data


def plot_residuals(residuals, names=("Volumes", "Returns")):
    """Istogramma dei residui con densità normale sovrapposta.

    residuals: matrice (n, k) con una colonna per ciascun modello,
    oppure lista di array 1D (uno per modello).
    """
    fig, axes = plt.subplots(1, len(names), figsize=(11, 4))

    for k, name in enumerate(names):
        e = np.asarray(residuals[k] if isinstance(residuals, (list, tuple))
                       else residuals[:, k], dtype=float)
        e = e[~np.isnan(e)]

        mean = np.mean(e)
        sd = np.std(e, ddof=1)
        ax = axes[k]

        ax.hist(e, bins='auto', density=True, alpha=0.65, edgecolor='white', color='#1f77b4')

        x_min = min(e.min(), mean - 4 * sd)
        x_max = max(e.max(), mean + 4 * sd)
        x = np.linspace(x_min, x_max, 500)
        pdf = np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        ax.plot(x, pdf, color='red', lw=1.8, label='Normal')

        ax.set_xlim(mean - 3.5 * sd, mean + 3.5 * sd)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
        ax.tick_params(axis='x', rotation=15, labelsize=9)

        ax.set_title(f"{name} Distribution", fontsize=11, pad=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


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


def impact_fit(returns, volumes, delta=1.0, p=10, plot=False):
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

    if plot:
        plot_residuals([np.asarray(res_vol.resid), np.asarray(res_ret.resid)])

    print('Volumes R2: ', res_vol.rsquared, res_vol.rsquared_adj)
    print('Returns R2: ', res_ret.rsquared, res_ret.rsquared_adj)

    return {
        "p": p,
        "delta": delta,
        "volume_model": {
            "params": res_vol.params,
            "sigma2": res_vol.mse_resid,
            "resid": np.asarray(res_vol.resid),
            "rsquared": res_vol.rsquared,
            "rsquared_adj": res_vol.rsquared_adj,
        },
        "return_model": {
            "params": res_ret.params,
            "sigma2": res_ret.mse_resid,
            "resid": np.asarray(res_ret.resid),
            "rsquared": res_ret.rsquared,
            "rsquared_adj": res_ret.rsquared_adj,
        },
    }


def impact_simulate(results, n_steps, initial_v, initial_price=100.0,
                    seed=None, bootstrap=False):
    """initial_v: array cronologico (dal più vecchio al più recente) di lunghezza p.

    bootstrap=False -> shock gaussiani con varianza residua di ciascun modello
    bootstrap=True  -> shock ricampionati con reinserimento dai residui stimati
                       (stesso istante per volume e rendimento)
    """
    rng = np.random.default_rng(seed)
    p = results["p"]
    delta = results["delta"]

    initial_v = np.asarray(initial_v, dtype=float)
    if initial_v.shape != (p,):
        raise ValueError(f"initial_v deve avere lunghezza p={p}")

    v_params = results["volume_model"]["params"]
    r_params = results["return_model"]["params"]

    # Shock pre-generati, uno per passo
    if bootstrap:
        v_resid = results["volume_model"]["resid"]
        r_resid = results["return_model"]["resid"]
        idx = rng.integers(0, len(v_resid), size=n_steps)  # stesso indice per i due modelli
        v_shocks, r_shocks = v_resid[idx], r_resid[idx]
    else:
        v_shocks = rng.normal(0, np.sqrt(results["volume_model"]["sigma2"]), n_steps)
        r_shocks = rng.normal(0, np.sqrt(results["return_model"]["sigma2"]), n_steps)

    sim_v = np.zeros(n_steps + p)
    sim_r = np.zeros(n_steps + p)
    prices = np.zeros(n_steps + p + 1)
    sim_v[:p] = initial_v
    prices[p] = initial_price

    for t in range(p, n_steps + p):
        v_lagged = sim_v[t - p:t][::-1]  # V_{t-1}, ..., V_{t-p}

        # 1. Volume
        sim_v[t] = np.dot(np.concatenate([[1.0], v_lagged]), v_params) + v_shocks[t - p]

        # 2. Rendimento
        exog_r = np.concatenate([[1.0, power_transform(sim_v[t], delta)],
                                 power_transform(v_lagged, delta)])
        sim_r[t] = np.dot(exog_r, r_params) + r_shocks[t - p]

        # 3. Prezzo
        prices[t + 1] = prices[t] * (1 + sim_r[t])

    return prices[p:p + n_steps], sim_v[p:], sim_r[p:]


def save_impact_model(results, path):
    """Salva i risultati di impact_fit in un file .npz (nessun pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vm, rm = results["volume_model"], results["return_model"]
    meta = {
        "p": int(results["p"]),
        "delta": float(results["delta"]),
        "volume_sigma2": float(vm["sigma2"]),
        "volume_rsquared": float(vm["rsquared"]),
        "volume_rsquared_adj": float(vm["rsquared_adj"]),
        "return_sigma2": float(rm["sigma2"]),
        "return_rsquared": float(rm["rsquared"]),
        "return_rsquared_adj": float(rm["rsquared_adj"]),
    }
    np.savez_compressed(
        path,
        meta=json.dumps(meta),
        volume_params=np.asarray(vm["params"]),
        volume_resid=np.asarray(vm["resid"]),
        return_params=np.asarray(rm["params"]),
        return_resid=np.asarray(rm["resid"]),
    )

def load_impact_model(path):
    """Ricarica un modello salvato con save_impact_model, nello stesso formato di impact_fit."""
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        return {
            "p": meta["p"],
            "delta": meta["delta"],
            "volume_model": {
                "params": f["volume_params"],
                "sigma2": meta["volume_sigma2"],
                "resid": f["volume_resid"],
                "rsquared": meta["volume_rsquared"],
                "rsquared_adj": meta["volume_rsquared_adj"],
            },
            "return_model": {
                "params": f["return_params"],
                "sigma2": meta["return_sigma2"],
                "resid": f["return_resid"],
                "rsquared": meta["return_rsquared"],
                "rsquared_adj": meta["return_rsquared_adj"],
            },
        }

def plot_impact_fit(results, residuals=True, show=True):
    """Visualizza i risultati di impact_fit (o di un modello caricato da file).

    Pannelli:
      1. coefficienti AR del modello dei volumi in funzione del lag
      2. kernel di impatto sui rendimenti (lag 0 = V_t contemporaneo)
      3. impatto cumulato, cioè la somma dei coefficienti fino al lag k
      4. funzione di impatto sign(v)|v|^delta
    Se residuals=True mostra anche gli istogrammi dei residui (plot_residuals).
    """
    p = results["p"]
    delta = results["delta"]
    vm, rm = results["volume_model"], results["return_model"]

    a = np.asarray(vm["params"])[1:]   # lag 1..p (esclude la costante)
    b = np.asarray(rm["params"])[1:]   # lag 0..p (esclude la costante)
    lags_v = np.arange(1, p + 1)
    lags_r = np.arange(0, p + 1)

    def draw(ax, x, y, title, xlabel="lag", ylabel=None):
        if len(x) <= 30:
            ax.stem(x, y, basefmt=" ")
        else:
            ax.plot(x, y, lw=1.2)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    draw(axes[0, 0], lags_v, a,
         f"Volume AR({p}) - R² = {vm['rsquared']:.3f} (adj {vm['rsquared_adj']:.3f})",
         ylabel="coefficiente")
    draw(axes[0, 1], lags_r, b,
         f"Kernel di impatto sui rendimenti - R² = {rm['rsquared']:.3f} (adj {rm['rsquared_adj']:.3f})",
         ylabel="coefficiente")
    draw(axes[1, 0], lags_r, np.cumsum(b),
         "Impatto cumulato sui rendimenti", ylabel="somma coefficienti")

    ax = axes[1, 1]
    x = np.linspace(-3, 3, 400)
    ax.plot(x, power_transform(x, delta), color='#d62728', lw=1.8, label=f"delta = {delta}")
    ax.plot(x, x, color='gray', lw=1, ls='--', label="lineare")
    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_title("Funzione di impatto sign(v)|v|^delta", fontsize=11)
    ax.set_xlabel("v")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Fit: p = {p}, delta = {delta}", fontsize=12)
    plt.tight_layout()
    if show:
        plt.show()

    if residuals:
        plot_residuals([np.asarray(vm["resid"]), np.asarray(rm["resid"])])

    return fig

if __name__ == '__main__':
    paths = np.array(listdir('..\\database\\data'))

    p = 100
    delta = 0.0
    bootstrap_residuals = True

    name = f'reg_delta_{delta}_{p}'

    dir = 'database\\data_' + name

    if not os.path.exists('..\\' + dir):
        os.makedirs('..\\' + dir)

    clear_data(name)

    model_dir = Path('..') / 'models' / name

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for path in paths:
        print(path)
        model_path = model_dir / (Path(path).stem + '.npz')

        prices, volumes, signs = open_data(path)

        r = (prices[1:] - prices[:-1]) / prices[:-1]
        v = volumes[:-1] * signs[:-1]

        # Prendi i primi 'p' valori come seme per la simulazione
        initial_v = v[:p]
        initial_r = r[:p]
        initial_price = prices[p]  # Prezzo reale al punto p

        if model_path.exists():
            print('Load Model')
            results = load_impact_model(model_path)       # salta il fit
        else:
            print('Compute Model')
            results = impact_fit(r, v, delta=delta, p=p)
            save_impact_model(results, model_path)

        save_impact_model(results, model_path)

        plot_impact_fit(results)

        exit()

        prices_sim, volumes_sim, r_sim = impact_simulate(
            results,
            n_steps=len(r),
            initial_v=initial_v,
            initial_price=initial_price,
            bootstrap=True
        )

        #save_simulated_data(f"..\\{dir}\\" + path, prices_sim, volumes_sim)