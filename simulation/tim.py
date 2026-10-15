import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import json
from pathlib import Path
from os import listdir
import os

from utils import clear_data, open_data, save_simulated_data

def plot_residuals(residuals, names=("Volumes", "Returns"), save_path=None, step=10):
    """Histogram of the residuals, using one residual every `step` (step=1 -> all)."""
    fig, axes = plt.subplots(1, len(names), figsize=(11, 4))

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

def plot_impact_fit(results, out_dir=None, residuals=True, resid_step = 10):
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
         f"Volume AR({p}) - in-sample R² = {vm['rsquared']:.3f} (adj {vm['rsquared_adj']:.3f})",
         ylabel="coefficient")
    draw(axes[0, 1], lags_r, b,
         f"Return impact kernel - in-sample R² = {rm['rsquared']:.3f} (adj {rm['rsquared_adj']:.3f})",
         ylabel="coefficient")
    draw(axes[1, 0], lags_v, np.cumsum(a),
         "Cumulative AR coefficients on volumes", ylabel="cumulative sum of coefficients")
    draw(axes[1, 1], lags_r, np.cumsum(b),
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

from numpy.lib.stride_tricks import sliding_window_view


def _blocks(returns, volumes, p, delta, chunk):
    """Genera blocchi (X_vol, y_vol, X_ret, y_ret) di al più `chunk` righe per un giorno.
    I lag restano interni al giorno; nessuna matrice completa in memoria."""
    returns = np.asarray(returns, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if len(returns) != len(volumes):
        raise ValueError("returns e volumes devono avere la stessa lunghezza")
    n = len(volumes)
    if n <= p:
        return

    tv = power_transform(volumes, delta)  # trasformata 1D, una volta sola
    # vista (n-p, p) con colonne x_{t-1}, ..., x_{t-p}, senza copia
    Wv = sliding_window_view(volumes, p)[:n - p, ::-1]
    Wt = sliding_window_view(tv, p)[:n - p, ::-1]

    for s in range(0, n - p, chunk):
        e = min(s + chunk, n - p)
        m = e - s

        Xv = np.empty((m, p + 1))
        Xv[:, 0] = 1.0
        Xv[:, 1:] = Wv[s:e]

        Xr = np.empty((m, p + 2))
        Xr[:, 0] = 1.0
        Xr[:, 1] = tv[p + s:p + e]
        Xr[:, 2:] = Wt[s:e]

        yield Xv, volumes[p + s:p + e], Xr, returns[p + s:p + e]


def impact_fit_multi(days, delta=1.0, p=10, plot=False, chunk=10000):
    """Fit di un unico modello su più giorni via equazioni normali accumulate.

    days: lista di tuple (returns, volumes), una per giorno. Matematicamente
    equivalente all'OLS sulle righe impilate, ma con memoria O(chunk * p).
    """
    kv, kr = p + 1, p + 2
    XtX_v, Xty_v = np.zeros((kv, kv)), np.zeros(kv)
    XtX_r, Xty_r = np.zeros((kr, kr)), np.zeros(kr)
    n = 0
    sy_v = sy2_v = sy_r = sy2_r = 0.0

    # Passo 1: accumulo X'X, X'y e somme per l'R2
    for returns, volumes in days:
        for Xv, yv, Xr, yr in _blocks(returns, volumes, p, delta, chunk):
            XtX_v += Xv.T @ Xv
            Xty_v += Xv.T @ yv
            XtX_r += Xr.T @ Xr
            Xty_r += Xr.T @ yr
            n += len(yv)
            sy_v += yv.sum();  sy2_v += yv @ yv
            sy_r += yr.sum();  sy2_r += yr @ yr

    if n == 0:
        raise ValueError("nessun giorno con lunghezza > p")

    beta_v = np.linalg.lstsq(XtX_v, Xty_v, rcond=None)[0]
    beta_r = np.linalg.lstsq(XtX_r, Xty_r, rcond=None)[0]

    # Passo 2: residui a blocchi (stesso ordine dei giorni)
    resid_v, resid_r = [], []
    for returns, volumes in days:
        for Xv, yv, Xr, yr in _blocks(returns, volumes, p, delta, chunk):
            resid_v.append(yv - Xv @ beta_v)
            resid_r.append(yr - Xr @ beta_r)
    resid_v = np.concatenate(resid_v)
    resid_r = np.concatenate(resid_r)

    def stats(resid, k, sy, sy2):
        ssr = resid @ resid
        sst = sy2 - sy ** 2 / n
        r2 = 1.0 - ssr / sst
        r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
        return ssr / (n - k), r2, r2_adj

    s2_v, r2_v, r2a_v = stats(resid_v, kv, sy_v, sy2_v)
    s2_r, r2_r, r2a_r = stats(resid_r, kr, sy_r, sy2_r)

    if plot:
        plot_residuals([resid_v, resid_r])

    print('Volumes R2: ', r2_v, r2a_v)
    print('Returns R2: ', r2_r, r2a_r)

    return {
        "p": p,
        "delta": delta,
        "volume_model": {"params": beta_v, "sigma2": s2_v, "resid": resid_v,
                         "rsquared": r2_v, "rsquared_adj": r2a_v},
        "return_model": {"params": beta_r, "sigma2": s2_r, "resid": resid_r,
                         "rsquared": r2_r, "rsquared_adj": r2a_r},
    }

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


def impact_evaluate(results, days, chunk=10000):
    """Out-of-sample evaluation on days not used for the fit.

    One-step-ahead predictions: the regressors are the REAL lagged volumes of each
    test day (lags never cross day boundaries), the coefficients are the fitted ones.
    Returns pooled stats over all test days and the R2 of each day.
    """
    p, delta = results["p"], results["delta"]
    betas = {"volume_model": np.asarray(results["volume_model"]["params"]),
             "return_model": np.asarray(results["return_model"]["params"])}

    tot = {m: np.zeros(4) for m in betas}
    per_day = {m: [] for m in betas}

    for returns, volumes in days:
        day = {m: np.zeros(4) for m in betas}
        for Xv, yv, Xr, yr in _blocks(returns, volumes, p, delta, chunk):
            for m, X, y in (("volume_model", Xv, yv), ("return_model", Xr, yr)):
                e = y - X @ betas[m]
                day[m] += np.array([len(y), y.sum(), y @ y, e @ e])
        for m in betas:
            tot[m] += day[m]
            per_day[m].append(_oos_stats(day[m], len(betas[m]))["rsquared"])

    out = {}
    for m in betas:
        out[m] = _oos_stats(tot[m], len(betas[m]))
        out[m]["per_day_rsquared"] = per_day[m]
    return out


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
    delta = 1.0
    fitting_window = 100          # number of trading days used for the fit
    bootstrap_residuals = True

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

    exit()

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
        )

        save_simulated_data(f"..\\{dir}\\" + path, prices_sim, volumes_sim)