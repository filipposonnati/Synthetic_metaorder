"""
VAR strutturale (identificazione ricorsiva con Cholesky) su [volumi, rendimenti].

1. Stima in forma ridotta (var_reduced_fit):   y_t = sum_k A_k y_{t-k} + u_t,  Cov(u) = Sigma
2. Cholesky con ordinamento (V, R):            Sigma = L D L',  L triangolare bassa a diagonale unitaria
3. Forma strutturale:                          A0 y_t = sum_k B_k y_{t-k} + eps_t
       A0 = L^{-1},  B_k = A0 A_k,  eps_t = A0 u_t,  Cov(eps) = D (diagonale)
   cioe'
       V_t = sum_k B_k[V,.] y_{t-k}                      + eps^V_t
       R_t = b0 * V_t + sum_k B_k[R,.] y_{t-k}           + eps^R_t,   b0 = -A0[1,0] = Sigma_VR / Sigma_VV
   b0 e' l'impatto contemporaneo v_t -> r_t.
4. Si salva il modello strutturale e si simula ricorsivamente con la forma strutturale
   (prima V_t, poi R_t dato V_t), con shock strutturali ortogonali.

coefs / coefs_struct hanno forma (p, eq, regressore), 0 = volumi, 1 = rendimenti.
"""
import json
import os
from os import listdir
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from utils import clear_data, open_data, save_simulated_data

_EQ = ("volumes", "returns")   # column 0 = volumes, column 1 = returns

from numpy.lib.stride_tricks import sliding_window_view

def _blocks(returns, volumes, p, chunk):
    """Yield blocks (X, y) of at most `chunk` rows for one day.

    y = [V_t, R_t]; X = [V_{t-1}, R_{t-1}, V_{t-2}, R_{t-2}, ..., V_{t-p}, R_{t-p}]
    (same column ordering as statsmodels' VAR). Lags stay inside the day and the
    full design matrix is never built."""
    returns = np.asarray(returns, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if len(returns) != len(volumes):
        raise ValueError("returns and volumes must have the same length")
    n = len(volumes)
    if n <= p:
        return

    data = np.column_stack([volumes, returns])                          # (n, 2)
    # view (n-p, 2, p): [row, variable, lag-1], without copying
    W = sliding_window_view(data, p, axis=0)[:n - p, :, ::-1]

    for s in range(0, n - p, chunk):
        e = min(s + chunk, n - p)
        X = np.ascontiguousarray(W[s:e].transpose(0, 2, 1)).reshape(e - s, 2 * p)
        yield X, data[p + s:p + e]

def _oos_stats(n, sy, sy2, ssr, n_reg):
    """R2 around the mean of the evaluated sample."""
    if n == 0:
        return {"n": 0, "mse": np.nan, "rmse": np.nan, "rsquared": np.nan, "rsquared_adj": np.nan}
    sst = sy2 - sy ** 2 / n
    r2 = 1.0 - ssr / sst
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - n_reg)
    return {"n": int(n), "mse": ssr / n, "rmse": float(np.sqrt(ssr / n)),
            "rsquared": float(r2), "rsquared_adj": float(r2_adj)}

def plot_residuals(residuals, names=("Volumes", "Returns"), save_path=None, step=10):
    """Histogram of the residuals, using one residual every `step` (step=1 -> all).
    residuals: array (n, 2), one column per equation."""
    fig, axes = plt.subplots(1, len(names), figsize=(11, 4))

    for k, name in enumerate(names):
        e = np.asarray(residuals[:, k], dtype=float)
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

def trim_top_volume(prices, volumes, signs, top_frac=0.05):
    """Drop the transactions whose |volume| is in the top `top_frac` of the day.
    Applied on the raw tick series, before computing returns, so that the
    remaining ticks form a consistent series (returns between consecutive kept ticks)."""
    prices, volumes, signs = map(np.asarray, (prices, volumes, signs))
    thr = np.quantile(np.abs(volumes), 1.0 - top_frac)
    keep = np.abs(volumes) <= thr
    return prices[keep], volumes[keep], signs[keep]

def plot_var_reduced_fit(results, out_dir=None, residuals=True, resid_step=10):
    """Plot the fit results (or of a model loaded from file).
    If out_dir is given, figures are saved there (var_fit.png, var_fit_cumulative.png, var_fit_std.png,
    var_fit_cumulative_std.png, residuals.png)
    instead of being shown.

    2x2 grid of the coefficients vs lag: rows are the equations (volumes, returns),
    columns are the lagged regressors (volumes, returns).
    """
    if out_dir is not None:
        out_dir = Path(out_dir)
    p = results["p"]
    fitted = results["model"]
    coefs = np.asarray(fitted.coefs)          # (p, eq, regressor), 0 = volumes, 1 = returns
    sigma = np.asarray(fitted.sigma_u)
    lags = np.arange(1, p + 1)

    names = ("Volumes", "Returns")
    colors = (_BLUE, _ORANGE)
    r2, r2_adj = results["r2"], results["r2_adj"]

    corr = sigma[0, 1] / np.sqrt(sigma[0, 0] * sigma[1, 1])

    def draw_grid(values, ylabel, title_fmt, suptitle, filename, annotate_r2=False):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.patch.set_facecolor(_SURFACE)
        for eq in range(2):
            for reg in range(2):
                ax = axes[eq, reg]
                ax.set_facecolor(_SURFACE)
                y = values[:, eq, reg]
                if p <= 30:
                    ax.stem(lags, y, basefmt=" ")
                else:
                    ax.plot(lags, y, lw=1.2, color=colors[reg])
                ax.axhline(0, color=_INK_2, lw=0.8)
                ax.set_title(title_fmt.format(reg=names[reg], eq=names[eq]),
                             fontsize=11, color=_INK)
                ax.set_xlabel("lag k")
                ax.set_ylabel(ylabel)
                ax.grid(True, color=_GRID, alpha=0.8)
                if annotate_r2 and reg == 0:
                    k = _EQ[eq]
                    ax.annotate(f"in-sample R² = {r2[k]:.3f} (adj {r2_adj[k]:.3f})",
                                xy=(0.02, 0.95), xycoords="axes fraction", va="top",
                                fontsize=9, color=_INK_2)
        fig.suptitle(suptitle, fontsize=12, color=_INK)
        plt.tight_layout()
        if out_dir is not None:
            fig.savefig(Path(out_dir) / filename, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    # Figure 1: coefficients per lag
    draw_grid(coefs, "coefficient",
              "{reg}(t-k) → {eq}(t)",
              f"Reduced-form VAR({p}) - residual correlation volumes/returns = {corr:.3f}",
              "var_fit.png", annotate_r2=True)

    # Figure 2: cumulative coefficients, sum_{j<=k} coefficient_j
    draw_grid(np.cumsum(coefs, axis=0), "cumulative sum of coefficients",
              "Cumulative {reg}(t-k) → {eq}(t)",
              f"Reduced-form VAR({p}) - cumulative coefficients",
              "var_fit_cumulative.png")

    # Figures 3-4: standardized coefficients (before de-normalization), unit-free
    coefs_std = getattr(fitted, "coefs_std", None)
    if coefs_std is None:
        print("coefs_std not available in this model (saved with an older version): "
              "refit to get the standardized figures")
    else:
        coefs_std = np.asarray(coefs_std)
        draw_grid(coefs_std, "standardized coefficient",
                  "{reg}(t-k) → {eq}(t)",
                  f"Reduced-form VAR({p}) - standardized coefficients (comparable)",
                  "var_fit_std.png")
        draw_grid(np.cumsum(coefs_std, axis=0), "cumulative standardized coefficient",
                  "Cumulative {reg}(t-k) → {eq}(t)",
                  f"Reduced-form VAR({p}) - cumulative standardized coefficients",
                  "var_fit_cumulative_std.png")

    if residuals:
        plot_residuals(np.asarray(fitted.resid),
                       save_path=None if out_dir is None else out_dir / "residuals.png",
                       step=resid_step)

# ----------------------------------------------------------------------------
# Reduced -> structural
# ----------------------------------------------------------------------------
def to_structural(results):
    """Converte i risultati di var_reduced_fit (forma ridotta) in forma strutturale."""
    fitted = results["model"]
    p = results["p"]
    coefs = np.asarray(fitted.coefs)                 # (p, eq, reg), A_k
    sigma_u = np.asarray(fitted.sigma_u)
    resid = np.asarray(fitted.resid)                 # (n, 2)

    P = np.linalg.cholesky(sigma_u)                  # Sigma = P P', P triangolare bassa
    diagP = np.diag(P)
    L = P / diagP[None, :]                           # L unit-lower-triangular
    A0 = np.linalg.inv(L)                            # [[1, 0], [a10, 1]]
    sigma_eps = np.diag(diagP ** 2)                  # Cov(eps) = D

    coefs_struct = np.einsum("ij,kjl->kil", A0, coefs)   # B_k = A0 A_k
    resid_struct = resid @ A0.T                          # eps_t = A0 u_t

    # R2 in-sample strutturale (rendimenti: include V_t come regressore contemporaneo)
    m = 2 * p
    n = len(resid)
    r2, r2_adj = {}, {}
    for i, eq in enumerate(_EQ):
        sst = (resid[:, i] ** 2).sum() / (1.0 - results["r2"][eq])
        ssr = (resid_struct[:, i] ** 2).sum()
        n_reg = m + (1 if i == 1 else 0)
        r2[eq] = float(1.0 - ssr / sst)
        r2_adj[eq] = float(1.0 - (1.0 - r2[eq]) * (n - 1) / (n - n_reg))

    b0 = -A0[1, 0]
    print(f"Cholesky (ordine V, R): impatto contemporaneo b0 (v_t -> r_t) = {b0:.4e}")
    print(f"Struttural R2 volumi: {r2['volumes']:.4f} | rendimenti: {r2['returns']:.4f}")

    return {
        "p": p,
        "n_obs": results["n_obs"],
        "b0": float(b0),
        "A0": A0,
        "model": SimpleNamespace(coefs=coefs_struct, resid=resid_struct,
                                 sigma_eps=sigma_eps, A0=A0, k_ar=p, neqs=2,
                                 coefs_reduced=coefs, sigma_u=sigma_u),
        "r2": r2,
        "r2_adj": r2_adj,
    }


def var_structural_fit(days, p=10, chunk=5000, plot=False):
    """Fit forma ridotta su piu' giorni + passaggio alla forma strutturale (Cholesky)."""
    red = var_reduced_fit(days, p=p, chunk=chunk, plot=plot)
    return to_structural(red)


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot_var_structural_fit(results, out_dir=None, residuals=True, resid_step=10):
    """Coefficienti strutturali per lag (e cumulati). Per l'equazione dei rendimenti il
    regressore V ha anche il lag 0 (impatto contemporaneo b0)."""
    out_dir = Path(out_dir) if out_dir is not None else None
    p = results["p"]
    fitted = results["model"]
    coefs = np.asarray(fitted.coefs)
    b0 = results["b0"]
    names = ("Volumes", "Returns")
    r2, r2_adj = results["r2"], results["r2_adj"]

    def draw(values, lag0, ylabel, title_fmt, suptitle, filename, annotate=False):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for eq in range(2):
            for reg in range(2):
                ax = axes[eq, reg]
                has0 = (eq == 1 and reg == 0)
                lags = np.arange(0 if has0 else 1, p + 1)
                y = values[:, eq, reg]
                if has0:
                    y = np.concatenate([[lag0], y])
                if len(lags) <= 30:
                    ax.stem(lags, y, basefmt=" ")
                else:
                    ax.plot(lags, y, lw=1.2)
                ax.axhline(0, color='k', lw=0.8)
                ax.set_title(title_fmt.format(reg=names[reg], eq=names[eq]), fontsize=11)
                ax.set_xlabel("lag k")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                if annotate and reg == 0:
                    k = _EQ[eq]
                    ax.annotate(f"in-sample R² = {r2[k]:.3f} (adj {r2_adj[k]:.3f})",
                                xy=(0.02, 0.95), xycoords="axes fraction", va="top", fontsize=9)
        fig.suptitle(suptitle, fontsize=12)
        plt.tight_layout()
        if out_dir is not None:
            fig.savefig(out_dir / filename, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    draw(coefs, b0, "coefficient", "{reg}(t-k) → {eq}(t)",
         f"Structural VAR({p}) (Cholesky V→R) - b0 = {b0:.3e}", "var_struct_fit.png", annotate=True)

    cum = np.cumsum(coefs, axis=0)
    # il lag 0 va sommato anche ai cumulati V→R
    cum_r = cum.copy()
    cum_r[:, 1, 0] += b0
    draw(cum_r, b0, "cumulative sum of coefficients", "Cumulative {reg}(t-k) → {eq}(t)",
         f"Structural VAR({p}) - cumulative coefficients (V→R includes lag 0)",
         "var_struct_fit_cumulative.png")

    if residuals:
        plot_residuals(np.asarray(fitted.resid),
                       save_path=None if out_dir is None else out_dir / "residuals_struct.png",
                       step=resid_step)


# ----------------------------------------------------------------------------
# Out-of-sample (forma strutturale, V_t reale come regressore contemporaneo per R_t)
# ----------------------------------------------------------------------------
def var_structural_evaluate(results, days, chunk=5000):
    p = results["p"]
    B = np.asarray(results["model"].coefs)               # (p, eq, reg)
    beta = B.transpose(0, 2, 1).reshape(2 * p, 2)        # (2p, eq)
    b0 = results["b0"]

    tot = {"n": 0, "sy": np.zeros(2), "sy2": np.zeros(2), "ssr": np.zeros(2)}
    per_day = {eq: [] for eq in _EQ}
    n_reg = (2 * p, 2 * p + 1)

    for returns, volumes in days:
        day = {"n": 0, "sy": np.zeros(2), "sy2": np.zeros(2), "ssr": np.zeros(2)}
        for X, y in _blocks(returns, volumes, p, chunk):
            pred = X @ beta
            pred[:, 1] += b0 * y[:, 0]                   # impatto contemporaneo v_t -> r_t
            e = y - pred
            day["n"] += len(y)
            day["sy"] += y.sum(axis=0)
            day["sy2"] += (y ** 2).sum(axis=0)
            day["ssr"] += (e ** 2).sum(axis=0)
        for key in tot:
            tot[key] = tot[key] + day[key]
        for i, eq in enumerate(_EQ):
            per_day[eq].append(_oos_stats(day["n"], day["sy"][i], day["sy2"][i],
                                          day["ssr"][i], n_reg[i])["rsquared"])
    out = {}
    for i, eq in enumerate(_EQ):
        out[eq] = _oos_stats(tot["n"], tot["sy"][i], tot["sy2"][i], tot["ssr"][i], n_reg[i])
        out[eq]["per_day_rsquared"] = per_day[eq]
    return out


def save_oos_stats(results, oos, out_dir, window_info=None):
    out_dir = Path(out_dir)
    resid = np.asarray(results["model"].resid)
    summary = {"window_info": window_info or {}}
    print(f"\n{'equation':<10}{'sample':<14}{'n':>10}{'R2':>10}{'R2 adj':>10}{'RMSE':>14}")
    for i, eq in enumerate(_EQ):
        ins_row = {"n": int(len(resid)),
                   "rmse": float(np.sqrt(np.mean(resid[:, i] ** 2))),
                   "rsquared": float(results["r2"][eq]),
                   "rsquared_adj": float(results["r2_adj"][eq])}
        oos_row = {k: v for k, v in oos[eq].items() if k != "mse"}
        oos_row["per_day_rsquared"] = [float(x) for x in oos[eq]["per_day_rsquared"]]
        summary[eq] = {"in_sample": ins_row, "out_of_sample": oos_row}
        for label, r in (("in-sample", ins_row), ("out-of-sample", oos_row)):
            print(f"{eq:<10}{label:<14}{r['n']:>10}{r['rsquared']:>10.4f}"
                  f"{r['rsquared_adj']:>10.4f}{r['rmse']:>14.4e}")

    with open(out_dir / "oos_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, eq, title in zip(axes, _EQ, ("Volumes", "Returns")):
        r2_days = summary[eq]["out_of_sample"]["per_day_rsquared"]
        ax.plot(np.arange(1, len(r2_days) + 1), r2_days, lw=1.2, label="out-of-sample (per day)")
        ax.axhline(summary[eq]["in_sample"]["rsquared"], color='red', ls='--', label="in-sample")
        ax.axhline(summary[eq]["out_of_sample"]["rsquared"], color='green', ls=':',
                   label="out-of-sample (pooled)")
        ax.set_title(f"{title} equation (structural) - R² on test days", fontsize=11)
        ax.set_xlabel("test day")
        ax.set_ylabel("R²")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "oos_r2_per_day.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Simulazione con la forma strutturale
# ----------------------------------------------------------------------------
def simulate_var_structural(results, n_steps, initial_v, initial_r,
                            initial_price=100.0, bootstrap=False, joint_bootstrap=True,
                            center_residuals=True, seed=None):
    """Simulazione ricorsiva della forma strutturale:
         V_t = sum_k B_k[V,.] y_{t-k} + eps^V_t
         R_t = b0 V_t + sum_k B_k[R,.] y_{t-k} + eps^R_t
    Shock strutturali ortogonali:
      bootstrap=False        -> gaussiani indipendenti con varianze D
      bootstrap=True         -> residui strutturali ricampionati;
         joint_bootstrap=True  -> stesso istante per V e R (conserva dipendenza di ordine superiore)
         joint_bootstrap=False -> estrazioni indipendenti per le due equazioni."""
    fitted = results["model"]
    p = results["p"]
    B = np.asarray(fitted.coefs)                         # (p, 2, 2)
    b0 = results["b0"]
    rng = np.random.default_rng(seed)

    if bootstrap:
        eps = np.asarray(fitted.resid, dtype=np.float64)
        eps = eps[~np.isnan(eps).any(axis=1)]
        if center_residuals:
            eps = eps - eps.mean(axis=0)
        if joint_bootstrap:
            shocks = eps[rng.integers(0, len(eps), size=n_steps)]
        else:
            shocks = np.column_stack([eps[rng.integers(0, len(eps), size=n_steps), k]
                                      for k in range(2)])
    else:
        sd = np.sqrt(np.diag(np.asarray(fitted.sigma_eps)))
        shocks = rng.normal(0.0, 1.0, size=(n_steps, 2)) * sd

    y = np.empty((p + n_steps, 2))
    y[:p] = np.column_stack([initial_v, initial_r])[-p:]
    for t in range(p, p + n_steps):
        lag = np.einsum("ikj,ij->k", B, y[t - p:t][::-1])
        e = shocks[t - p]
        v_t = lag[0] + e[0]
        y[t, 0] = v_t
        y[t, 1] = lag[1] + b0 * v_t + e[1]

    sim_v, sim_r = y[p:, 0], y[p:, 1]
    prices = initial_price * np.concatenate([[1.0], np.cumprod(1 + sim_r[:-1])])
    return prices, sim_v, sim_r


# ----------------------------------------------------------------------------
# Save / load (.npz, nessun pickle): si salva il modello STRUTTURALE
# ----------------------------------------------------------------------------
def save_var_structural_model(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = results["model"]
    meta = {"p": int(results["p"]), "n_obs": int(results["n_obs"]), "b0": float(results["b0"]),
            "r2_volumes": results["r2"]["volumes"], "r2_returns": results["r2"]["returns"],
            "r2_adj_volumes": results["r2_adj"]["volumes"],
            "r2_adj_returns": results["r2_adj"]["returns"]}
    np.savez_compressed(path, meta=json.dumps(meta),
                        coefs=np.asarray(f.coefs), A0=np.asarray(f.A0),
                        resid=np.asarray(f.resid), sigma_eps=np.asarray(f.sigma_eps),
                        coefs_reduced=np.asarray(f.coefs_reduced),
                        sigma_u=np.asarray(f.sigma_u))


def load_var_structural_model(path):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        d = {k: f[k] for k in f.files if k != "meta"}
    p = meta["p"]
    return {
        "p": p, "n_obs": meta["n_obs"], "b0": meta["b0"], "A0": d["A0"],
        "model": SimpleNamespace(coefs=d["coefs"], resid=d["resid"], sigma_eps=d["sigma_eps"],
                                 A0=d["A0"], k_ar=p, neqs=2,
                                 coefs_reduced=d["coefs_reduced"], sigma_u=d["sigma_u"]),
        "r2": {"volumes": meta["r2_volumes"], "returns": meta["r2_returns"]},
        "r2_adj": {"volumes": meta["r2_adj_volumes"], "returns": meta["r2_adj_returns"]},
    }


if __name__ == '__main__':
    paths = np.array(sorted(listdir('..\\database\\data')))   # sorted -> chronological order

    p = 1000
    fitting_window = 200
    bootstrap_residuals = True
    joint_bootstrap = True        # stesso istante per eps^V ed eps^R
    trim_top = True
    top_frac = 0.05

    name = f'var_{p}'

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
        v = volumes[:-1] * signs[:-1]          # il trade t impatta r_t (v_t -> r_t contemporaneo)
        return prices, r, v

    fit_paths = paths[:fitting_window]
    test_paths = paths[fitting_window:]

    # ---- Fit (forma ridotta -> Cholesky -> forma strutturale) ----
    if model_path.exists():
        print('Load Model')
        results = load_var_structural_model(model_path)
    else:
        print('Compute Model')
        days = []
        for path in fit_paths:
            _, r, v = load_day(path)
            days.append((r, v))
        results = var_structural_fit(days, p=p)
        save_var_structural_model(results, model_path)

    plot_var_structural_fit(results, out_dir=model_dir)

    # ---- Out-of-sample ----
    if len(test_paths) > 0:
        test_days = []
        for path in test_paths:
            _, r, v = load_day(path)
            test_days.append((r, v))
        oos = var_structural_evaluate(results, test_days)
        save_oos_stats(results, oos, model_dir,
                       window_info={"fitting_window": fitting_window,
                                    "n_fit_days": len(fit_paths),
                                    "n_test_days": len(test_paths),
                                    "trim_top": trim_top, "top_frac": top_frac,
                                    "b0": results["b0"]})
    else:
        print('No days left for out-of-sample evaluation')

    clear_data(name)

    # ---- Simulazione di ogni giorno con il modello strutturale ----
    for path in paths:
        print(path)
        prices, r, v = load_day(path)

        initial_v = v[:p]
        initial_r = r[:p]
        initial_price = prices[p]

        prices_sim, volumes_sim, r_sim = simulate_var_structural(
            results,
            n_steps=len(r),
            initial_v=initial_v,
            initial_r=initial_r,
            initial_price=initial_price,
            bootstrap=bootstrap_residuals,
            joint_bootstrap=joint_bootstrap
        )

        save_simulated_data(f"..\\{dir}\\" + path, prices_sim, volumes_sim)