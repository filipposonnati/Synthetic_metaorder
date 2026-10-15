import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import json
from pathlib import Path
from os import listdir
import os
from numpy.lib.stride_tricks import sliding_window_view

from utils import clear_data, open_data, save_simulated_data

# ============================================================================
# Struttura del modulo
#
#   VOLUMI    : volume_fit_multi, volume_evaluate, volume_simulate,
#               save_volume_model, load_volume_model
#   RENDIMENTI: return_fit_multi, return_evaluate, return_simulate,
#               save_return_model, load_return_model
#   COMBINATI : impact_fit_multi, impact_evaluate, impact_simulate,
#               save_impact_model, load_impact_model  (wrapper dei precedenti)
#
# volume_model = {"p", "params", "sigma2", "resid", "rsquared", "rsquared_adj"}
# return_model = {"p", "delta", + gli stessi campi}
# ============================================================================


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_residuals(residuals, names=("Volumes", "Returns"), save_path=None, step=10):
    """Histogram of the residuals, using one residual every `step` (step=1 -> all)."""
    fig, axes = plt.subplots(1, len(names), figsize=(11, 4))
    axes = np.atleast_1d(axes)

    for k, name in enumerate(names):
        e = np.asarray(residuals[k] if isinstance(residuals, (list, tuple))
                       else residuals[:, k], dtype=float)
        e = e[~np.isnan(e)][::step]

        mean = np.mean(e)
        sd = np.std(e, ddof=1)
        lo, hi = mean - 3.5 * sd, mean + 3.5 * sd
        ax = axes[k]

        # fixed number of bins inside the visible range (bins='auto' explodes with heavy tails)
        ax.hist(e, bins=200, range=(lo, hi), density=True,
                alpha=0.8, edgecolor='white', color='#1f77b4')

        ax.set_xlim(lo, hi)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
        ax.tick_params(axis='x', rotation=15, labelsize=9)

        ax.set_title(f"{name} residuals distribution (1 every {step})", fontsize=11, pad=10)
        ax.set_xlabel("residual")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def _draw(ax, x, y, title, xlabel="lag", ylabel=None):
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


def plot_volume_fit(volume_model, out_dir=None, residuals=True, resid_step=10):
    """AR coefficients of the volume model and their cumulative sum."""
    p = volume_model["p"]
    a = np.asarray(volume_model["params"])[1:]   # lags 1..p (constant excluded)
    lags = np.arange(1, p + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _draw(axes[0], lags, a,
          f"Volume AR({p}) - in-sample R² = {volume_model['rsquared']:.3f} "
          f"(adj {volume_model['rsquared_adj']:.3f})", ylabel="coefficient")
    _draw(axes[1], lags, np.cumsum(a),
          "Cumulative AR coefficients on volumes", ylabel="cumulative sum of coefficients")
    plt.tight_layout()

    if out_dir is not None:
        fig.savefig(Path(out_dir) / "volume_fit.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()

    if residuals:
        plot_residuals([np.asarray(volume_model["resid"])], names=("Volumes",),
                       save_path=None if out_dir is None else Path(out_dir) / "volume_residuals.png",
                       step=resid_step)


def plot_return_fit(return_model, out_dir=None, residuals=True, resid_step=10):
    """Return impact kernel and its cumulative sum."""
    p, delta = return_model["p"], return_model["delta"]
    b = np.asarray(return_model["params"])[1:]   # lags 0..p (constant excluded)
    lags = np.arange(0, p + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _draw(axes[0], lags, b,
          f"Return impact kernel (delta={delta}) - in-sample R² = {return_model['rsquared']:.3f} "
          f"(adj {return_model['rsquared_adj']:.3f})", ylabel="coefficient")
    _draw(axes[1], lags, np.cumsum(b),
          "Cumulative impact on returns", ylabel="cumulative sum of coefficients")
    plt.tight_layout()

    if out_dir is not None:
        fig.savefig(Path(out_dir) / "return_fit.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()

    if residuals:
        plot_residuals([np.asarray(return_model["resid"])], names=("Returns",),
                       save_path=None if out_dir is None else Path(out_dir) / "return_residuals.png",
                       step=resid_step)


def plot_impact_fit(results, out_dir=None, residuals=True, resid_step=10):
    """Plot the fit results. If out_dir is given, figures are saved there
    (impact_fit.png, residuals.png) instead of being shown.

    Panels:
      1. AR coefficients of the volume model vs lag
      2. Return impact kernel (lag 0 = contemporaneous V_t)
      3. Cumulative AR coefficients of the volume model
      4. Cumulative impact on returns, i.e. sum of the kernel coefficients up to lag k
    """
    p = results["p"]
    delta = results["delta"]
    vm, rm = results["volume_model"], results["return_model"]

    a = np.asarray(vm["params"])[1:]   # lags 1..p (constant excluded)
    b = np.asarray(rm["params"])[1:]   # lags 0..p (constant excluded)
    lags_v = np.arange(1, p + 1)
    lags_r = np.arange(0, p + 1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    _draw(axes[0, 0], lags_v, a,
          f"Volume AR({p}) - in-sample R² = {vm['rsquared']:.3f} (adj {vm['rsquared_adj']:.3f})",
          ylabel="coefficient")
    _draw(axes[0, 1], lags_r, b,
          f"Return impact kernel - in-sample R² = {rm['rsquared']:.3f} (adj {rm['rsquared_adj']:.3f})",
          ylabel="coefficient")
    _draw(axes[1, 0], lags_v, np.cumsum(a),
          "Cumulative AR coefficients on volumes", ylabel="cumulative sum of coefficients")
    _draw(axes[1, 1], lags_r, np.cumsum(b),
          "Cumulative impact on returns", ylabel="cumulative sum of coefficients")

    fig.suptitle(f"Fit: p = {p}, delta = {delta}", fontsize=12)
    plt.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        fig.savefig(out_dir / "impact_fit.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()

    if residuals:
        plot_residuals([np.asarray(vm["resid"]), np.asarray(rm["resid"])],
                       save_path=None if out_dir is None else out_dir / "residuals.png",
                       step=resid_step)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
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


def _model_stats(resid, k, n, sy, sy2):
    ssr = resid @ resid
    sst = sy2 - sy ** 2 / n
    r2 = 1.0 - ssr / sst
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    return ssr / (n - k), r2, r2_adj


def _make_shocks(resid, sigma2, n_steps, seed, bootstrap, idx):
    """Shock per n_steps passi.
    idx       -> resid[idx] (permette di condividere l'estrazione tra volumi e rendimenti)
    bootstrap -> resampling con reinserimento dei residui
    altrimenti-> gaussiani con varianza sigma2"""
    if idx is not None:
        return np.asarray(resid)[np.asarray(idx)]
    rng = np.random.default_rng(seed)
    if bootstrap:
        return np.asarray(resid)[rng.integers(0, len(resid), size=n_steps)]
    return rng.normal(0, np.sqrt(sigma2), n_steps)


# ----------------------------------------------------------------------------
# Single-day fit with statsmodels (invariata)
# ----------------------------------------------------------------------------
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
            "p": p,
            "params": res_vol.params,
            "sigma2": res_vol.mse_resid,
            "resid": np.asarray(res_vol.resid),
            "rsquared": res_vol.rsquared,
            "rsquared_adj": res_vol.rsquared_adj,
        },
        "return_model": {
            "p": p,
            "delta": delta,
            "params": res_ret.params,
            "sigma2": res_ret.mse_resid,
            "resid": np.asarray(res_ret.resid),
            "rsquared": res_ret.rsquared,
            "rsquared_adj": res_ret.rsquared_adj,
        },
    }


# ----------------------------------------------------------------------------
# Blocchi (lag interni al giorno, nessuna matrice completa in memoria)
# ----------------------------------------------------------------------------
def _volume_blocks(volumes, p, chunk):
    """Genera blocchi (Xv, yv) di al più `chunk` righe per un giorno.
    Xv = [1, V_{t-1}, ..., V_{t-p}], yv = V_t."""
    volumes = np.asarray(volumes, dtype=float)
    n = len(volumes)
    if n <= p:
        return

    # vista (n-p, p) con colonne x_{t-1}, ..., x_{t-p}, senza copia
    Wv = sliding_window_view(volumes, p)[:n - p, ::-1]

    for s in range(0, n - p, chunk):
        e = min(s + chunk, n - p)
        Xv = np.empty((e - s, p + 1))
        Xv[:, 0] = 1.0
        Xv[:, 1:] = Wv[s:e]
        yield Xv, volumes[p + s:p + e]


def _return_blocks(returns, volumes, p, delta, chunk):
    """Genera blocchi (Xr, yr) di al più `chunk` righe per un giorno.
    Xr = [1, T(V_t), T(V_{t-1}), ..., T(V_{t-p})], yr = R_t, con T = power_transform."""
    returns = np.asarray(returns, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if len(returns) != len(volumes):
        raise ValueError("returns e volumes devono avere la stessa lunghezza")
    n = len(volumes)
    if n <= p:
        return

    tv = power_transform(volumes, delta)  # trasformata 1D, una volta sola
    Wt = sliding_window_view(tv, p)[:n - p, ::-1]

    for s in range(0, n - p, chunk):
        e = min(s + chunk, n - p)
        Xr = np.empty((e - s, p + 2))
        Xr[:, 0] = 1.0
        Xr[:, 1] = tv[p + s:p + e]
        Xr[:, 2:] = Wt[s:e]
        yield Xr, returns[p + s:p + e]


# ----------------------------------------------------------------------------
# FIT: volumi
# ----------------------------------------------------------------------------
def _fit_normal_equations(block_iter_factory, k):
    """OLS su più giorni via equazioni normali accumulate (memoria O(chunk * k)).
    block_iter_factory: funzione senza argomenti che restituisce un iteratore di
    blocchi (X, y) su tutti i giorni; viene chiamata due volte (accumulo e residui)."""
    XtX, Xty = np.zeros((k, k)), np.zeros(k)
    n, sy, sy2 = 0, 0.0, 0.0

    for X, y in block_iter_factory():
        XtX += X.T @ X
        Xty += X.T @ y
        n += len(y)
        sy += y.sum()
        sy2 += y @ y

    if n == 0:
        raise ValueError("nessun giorno con lunghezza > p")

    beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    resid = np.concatenate([y - X @ beta for X, y in block_iter_factory()])
    s2, r2, r2a = _model_stats(resid, k, n, sy, sy2)
    return beta, s2, r2, r2a, resid


def volume_fit_multi(volume_days, p=10, plot=False, chunk=10000):
    """Fit del solo modello dei volumi AR(p) su più giorni.

    volume_days: lista di array di volumi (con segno), uno per giorno."""
    beta, s2, r2, r2a, resid = _fit_normal_equations(
        lambda: (b for v in volume_days for b in _volume_blocks(v, p, chunk)), p + 1)

    if plot:
        plot_residuals([resid], names=("Volumes",))
    print('Volumes R2: ', r2, r2a)

    return {"p": p, "params": beta, "sigma2": s2, "resid": resid,
            "rsquared": r2, "rsquared_adj": r2a}


# ----------------------------------------------------------------------------
# FIT: rendimenti
# ----------------------------------------------------------------------------
def return_fit_multi(days, delta=1.0, p=10, plot=False, chunk=10000):
    """Fit del solo modello dei rendimenti (kernel di impatto) su più giorni.

    days: lista di tuple (returns, volumes), una per giorno.
    Nota: i residui sono nell'ordine dei giorni, come quelli di volume_fit_multi
    se chiamato sugli stessi giorni (necessario per il bootstrap congiunto)."""
    beta, s2, r2, r2a, resid = _fit_normal_equations(
        lambda: (b for r, v in days for b in _return_blocks(r, v, p, delta, chunk)), p + 2)

    if plot:
        plot_residuals([resid], names=("Returns",))
    print('Returns R2: ', r2, r2a)

    return {"p": p, "delta": delta, "params": beta, "sigma2": s2, "resid": resid,
            "rsquared": r2, "rsquared_adj": r2a}


def impact_fit_multi(days, delta=1.0, p=10, plot=False, chunk=10000):
    """Wrapper: fit di entrambi i modelli su più giorni.
    days: lista di tuple (returns, volumes), una per giorno."""
    vm = volume_fit_multi([v for _, v in days], p=p, chunk=chunk)
    rm = return_fit_multi(days, delta=delta, p=p, chunk=chunk)
    if plot:
        plot_residuals([vm["resid"], rm["resid"]])
    return {"p": p, "delta": delta, "volume_model": vm, "return_model": rm}


# ----------------------------------------------------------------------------
# SIMULAZIONE
# ----------------------------------------------------------------------------
def volume_simulate(volume_model, n_steps, initial_v, seed=None, bootstrap=False, idx=None):
    """Simula n_steps volumi dal modello AR(p).

    initial_v: array cronologico (dal più vecchio al più recente) di lunghezza p.
    idx      : (opzionale) indici dei residui da usare come shock, lunghezza n_steps.
    Ritorna sim_v di lunghezza n_steps (i p valori iniziali sono esclusi)."""
    p = volume_model["p"]
    initial_v = np.asarray(initial_v, dtype=float)
    if initial_v.shape != (p,):
        raise ValueError(f"initial_v deve avere lunghezza p={p}")

    params = np.asarray(volume_model["params"])
    c, a = params[0], params[1:]
    shocks = _make_shocks(volume_model["resid"], volume_model["sigma2"],
                          n_steps, seed, bootstrap, idx)

    sim_v = np.zeros(n_steps + p)
    sim_v[:p] = initial_v
    for t in range(p, n_steps + p):
        sim_v[t] = c + a @ sim_v[t - p:t][::-1] + shocks[t - p]   # V_{t-1}, ..., V_{t-p}
    return sim_v[p:]


def return_simulate(return_model, sim_v, initial_v, initial_price=100.0,
                    seed=None, bootstrap=False, idx=None, chunk=10000):
    """Simula i rendimenti (e i prezzi) dati i volumi.

    sim_v     : volumi (con segno) per i quali generare i rendimenti, lunghezza n_steps
                (possono venire da volume_simulate o essere dati reali).
    initial_v : p volumi precedenti a sim_v, dal più vecchio al più recente.
    idx       : (opzionale) indici dei residui da usare come shock.
    Ritorna (prices, sim_r), entrambi di lunghezza n_steps; prices[0] = initial_price,
    prices[k+1] = prices[k] * (1 + sim_r[k])."""
    p, delta = return_model["p"], return_model["delta"]
    sim_v = np.asarray(sim_v, dtype=float)
    initial_v = np.asarray(initial_v, dtype=float)
    if initial_v.shape != (p,):
        raise ValueError(f"initial_v deve avere lunghezza p={p}")
    n_steps = len(sim_v)

    params = np.asarray(return_model["params"])
    c, b0, b = params[0], params[1], params[2:]
    shocks = _make_shocks(return_model["resid"], return_model["sigma2"],
                          n_steps, seed, bootstrap, idx)

    # i volumi sono dati: nessuna ricorsione, calcolo vettoriale a blocchi
    tfull = power_transform(np.concatenate([initial_v, sim_v]), delta)
    W = sliding_window_view(tfull, p)[:n_steps, ::-1]      # T(V_{t-1}), ..., T(V_{t-p})
    sim_r = np.empty(n_steps)
    for s in range(0, n_steps, chunk):
        e = min(s + chunk, n_steps)
        sim_r[s:e] = c + b0 * tfull[p + s:p + e] + W[s:e] @ b
    sim_r += shocks

    prices = initial_price * np.concatenate([[1.0], np.cumprod(1 + sim_r[:-1])])
    return prices, sim_r


def impact_simulate(results, n_steps, initial_v, initial_price=100.0,
                    seed=None, bootstrap=False, joint_bootstrap=True):
    """Wrapper: volumi + rendimenti + prezzi.

    initial_v: array cronologico (dal più vecchio al più recente) di lunghezza p.

    bootstrap=False -> shock gaussiani con varianza residua di ciascun modello
    bootstrap=True  -> shock ricampionati con reinserimento dai residui stimati
    joint_bootstrap (solo se bootstrap=True):
        True  -> stesso istante per volume e rendimento (conserva la dipendenza
                 di ordine superiore: code, eteroschedasticità; richiede che i due
                 modelli siano stati stimati sugli stessi giorni, nello stesso ordine)
        False -> estrazioni indipendenti, coerente con l'assenza di correlazione
                 tra i residui nel modello
    """
    rng = np.random.default_rng(seed)
    vm, rm = results["volume_model"], results["return_model"]

    idx = (rng.integers(0, len(vm["resid"]), size=n_steps)
           if bootstrap and joint_bootstrap else None)
    seed_v, seed_r = np.random.SeedSequence(seed).spawn(2)

    sim_v = volume_simulate(vm, n_steps, initial_v, seed=seed_v, bootstrap=bootstrap, idx=idx)
    prices, sim_r = return_simulate(rm, sim_v, initial_v, initial_price=initial_price,
                                    seed=seed_r, bootstrap=bootstrap, idx=idx)
    return prices, sim_v, sim_r


# ----------------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------------
def save_volume_model(volume_model, path):
    """Salva il solo modello dei volumi in un .npz (nessun pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"p": int(volume_model["p"]),
            "sigma2": float(volume_model["sigma2"]),
            "rsquared": float(volume_model["rsquared"]),
            "rsquared_adj": float(volume_model["rsquared_adj"])}
    np.savez_compressed(path, meta=json.dumps(meta),
                        params=np.asarray(volume_model["params"]),
                        resid=np.asarray(volume_model["resid"]))


def load_volume_model(path):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        return {"p": meta["p"], "params": f["params"], "sigma2": meta["sigma2"],
                "resid": f["resid"], "rsquared": meta["rsquared"],
                "rsquared_adj": meta["rsquared_adj"]}


def save_return_model(return_model, path):
    """Salva il solo modello dei rendimenti in un .npz (nessun pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"p": int(return_model["p"]),
            "delta": float(return_model["delta"]),
            "sigma2": float(return_model["sigma2"]),
            "rsquared": float(return_model["rsquared"]),
            "rsquared_adj": float(return_model["rsquared_adj"])}
    np.savez_compressed(path, meta=json.dumps(meta),
                        params=np.asarray(return_model["params"]),
                        resid=np.asarray(return_model["resid"]))


def load_return_model(path):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        return {"p": meta["p"], "delta": meta["delta"], "params": f["params"],
                "sigma2": meta["sigma2"], "resid": f["resid"],
                "rsquared": meta["rsquared"], "rsquared_adj": meta["rsquared_adj"]}


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
    """Ricarica un modello salvato con save_impact_model, nello stesso formato di impact_fit_multi."""
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        return {
            "p": meta["p"],
            "delta": meta["delta"],
            "volume_model": {
                "p": meta["p"],
                "params": f["volume_params"],
                "sigma2": meta["volume_sigma2"],
                "resid": f["volume_resid"],
                "rsquared": meta["volume_rsquared"],
                "rsquared_adj": meta["volume_rsquared_adj"],
            },
            "return_model": {
                "p": meta["p"],
                "delta": meta["delta"],
                "params": f["return_params"],
                "sigma2": meta["return_sigma2"],
                "resid": f["return_resid"],
                "rsquared": meta["return_rsquared"],
                "rsquared_adj": meta["return_rsquared_adj"],
            },
        }


# ----------------------------------------------------------------------------
# Out-of-sample evaluation
# ----------------------------------------------------------------------------
def _oos_stats(acc, k):
    """acc = [n, sum(y), sum(y^2), SSR]. R2 is computed around the mean of the evaluated sample."""
    n, sy, sy2, ssr = acc
    if n == 0:
        return {"n": 0, "mse": np.nan, "rmse": np.nan, "rsquared": np.nan, "rsquared_adj": np.nan}
    sst = sy2 - sy ** 2 / n
    r2 = 1.0 - ssr / sst
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    return {"n": int(n), "mse": ssr / n, "rmse": np.sqrt(ssr / n),
            "rsquared": r2, "rsquared_adj": r2_adj}


def _evaluate(beta, blocks_per_day):
    """blocks_per_day: iterabile (un elemento per giorno) di iteratori di blocchi (X, y)."""
    beta = np.asarray(beta)
    tot = np.zeros(4)
    per_day = []
    for blocks in blocks_per_day:
        day = np.zeros(4)
        for X, y in blocks:
            e = y - X @ beta
            day += np.array([len(y), y.sum(), y @ y, e @ e])
        tot += day
        per_day.append(_oos_stats(day, len(beta))["rsquared"])
    out = _oos_stats(tot, len(beta))
    out["per_day_rsquared"] = per_day
    return out


def volume_evaluate(volume_model, volume_days, chunk=10000):
    """Valutazione out-of-sample (previsione one-step-ahead) del modello dei volumi.
    volume_days: lista di array di volumi, uno per giorno."""
    p = volume_model["p"]
    return _evaluate(volume_model["params"],
                     (_volume_blocks(v, p, chunk) for v in volume_days))


def return_evaluate(return_model, days, chunk=10000):
    """Valutazione out-of-sample del modello dei rendimenti (regressori = volumi reali).
    days: lista di tuple (returns, volumes), una per giorno."""
    p, delta = return_model["p"], return_model["delta"]
    return _evaluate(return_model["params"],
                     (_return_blocks(r, v, p, delta, chunk) for r, v in days))


def impact_evaluate(results, days, chunk=10000):
    """Wrapper: valutazione out-of-sample di entrambi i modelli.
    Ritorna statistiche aggregate su tutti i giorni di test e l'R2 di ciascun giorno."""
    return {"volume_model": volume_evaluate(results["volume_model"], [v for _, v in days], chunk),
            "return_model": return_evaluate(results["return_model"], days, chunk)}


def save_oos_stats(results, oos, out_dir, window_info=None):
    """Save in-sample vs out-of-sample statistics as JSON, print a comparison table, plot per-day R2."""
    out_dir = Path(out_dir)
    summary = {"window_info": window_info or {}}
    print(f"\n{'model':<14}{'sample':<14}{'n':>10}{'R2':>10}{'R2 adj':>10}{'RMSE':>14}")
    for m in ("volume_model", "return_model"):
        ins = results[m]
        ins_row = {"n": int(len(ins["resid"])),
                   "rmse": float(np.sqrt(np.mean(np.asarray(ins["resid"]) ** 2))),
                   "rsquared": float(ins["rsquared"]),
                   "rsquared_adj": float(ins["rsquared_adj"])}
        oos_row = {k: (float(v) if k != "per_day_rsquared" and k != "n" else v)
                   for k, v in oos[m].items()}
        oos_row["per_day_rsquared"] = [float(x) for x in oos[m]["per_day_rsquared"]]
        summary[m] = {"in_sample": ins_row, "out_of_sample": oos_row}

        for label, r in (("in-sample", ins_row), ("out-of-sample", oos_row)):
            print(f"{m:<14}{label:<14}{r['n']:>10}{r['rsquared']:>10.4f}"
                  f"{r['rsquared_adj']:>10.4f}{r['rmse']:>14.4e}")

    with open(out_dir / "oos_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Per-day out-of-sample R2 vs in-sample R2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m, title in zip(axes, ("volume_model", "return_model"), ("Volumes", "Returns")):
        r2_days = summary[m]["out_of_sample"]["per_day_rsquared"]
        ax.plot(np.arange(1, len(r2_days) + 1), r2_days, lw=1.2, label="out-of-sample (per day)")
        ax.axhline(summary[m]["in_sample"]["rsquared"], color='red', ls='--', label="in-sample")
        ax.axhline(summary[m]["out_of_sample"]["rsquared"], color='green', ls=':', label="out-of-sample (pooled)")
        ax.set_title(f"{title} model - R² on test days", fontsize=11)
        ax.set_xlabel("test day")
        ax.set_ylabel("R²")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "oos_r2_per_day.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Data filtering
# ----------------------------------------------------------------------------
def trim_top_volume(prices, volumes, signs, top_frac=0.05):
    """Drop the transactions whose |volume| is in the top `top_frac` of the day.
    Applied on the raw tick series, before computing returns, so that the
    remaining ticks form a consistent series (returns between consecutive kept ticks)."""
    prices, volumes, signs = map(np.asarray, (prices, volumes, signs))
    thr = np.quantile(np.abs(volumes), 1.0 - top_frac)
    keep = np.abs(volumes) <= thr
    return prices[keep], volumes[keep], signs[keep]


if __name__ == '__main__':
    paths = np.array(sorted(listdir('..\\database\\data')))   # sorted -> chronological order

    p = 1000
    delta = 0.0
    fitting_window = 100          # number of trading days used for the fit
    bootstrap_residuals = True
    joint_bootstrap = False

    trim_top = True        # <-- flag: False restores the original behaviour
    top_frac = 0.05        # fraction of largest |volume| trades dropped each day

    name = f'tim_{delta}_{p}'

    dir = 'database\\data_' + name
    if not os.path.exists('..\\' + dir):
        os.makedirs('..\\' + dir)

    model_dir = Path('..') / 'models' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'model.npz'

    def load_day(path):
        prices, volumes, signs = open_data(path)
        if trim_top:
            prices, volumes, signs = trim_top_volume(prices, volumes, signs, top_frac)
        r = (prices[1:] - prices[:-1]) / prices[:-1]
        v = volumes[:-1] * signs[:-1]
        return prices, r, v

    fit_paths = paths[:fitting_window]
    test_paths = paths[fitting_window:]

    # ---- Single fit on the first fitting_window days ----
    if model_path.exists():
        print('Load Model')
        results = load_impact_model(model_path)
    else:
        print('Compute Model')
        days = []
        for path in fit_paths:
            _, r, v = load_day(path)
            days.append((r, v))
        results = impact_fit_multi(days, delta=delta, p=p)
        save_impact_model(results, model_path)

    # ---- Plots saved next to the model ----
    plot_impact_fit(results, out_dir=None)#model_dir)

    # ---- Out-of-sample evaluation on the remaining days ----
    if len(test_paths) > 0:
        test_days = []
        for path in test_paths:
            _, r, v = load_day(path)
            test_days.append((r, v))
        oos = impact_evaluate(results, test_days)
        save_oos_stats(results, oos, model_dir,
                       window_info={"fitting_window": fitting_window,
                                    "n_fit_days": len(fit_paths),
                                    "n_test_days": len(test_paths)})
    else:
        print('No days left for out-of-sample evaluation')

    clear_data(name)

    # ---- Simulation for each day with the common model ----
    for path in paths:
        print(path)
        prices, r, v = load_day(path)

        initial_v = v[:p]
        initial_price = prices[p]

        prices_sim, volumes_sim, r_sim = impact_simulate(
            results,
            n_steps=len(r),
            initial_v=initial_v,
            initial_price=initial_price,
            bootstrap=bootstrap_residuals,
            joint_bootstrap=joint_bootstrap
        )

        save_simulated_data(f"..\\{dir}\\" + path, prices_sim, volumes_sim)