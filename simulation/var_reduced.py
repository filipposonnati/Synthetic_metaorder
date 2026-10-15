import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

from os import listdir
from pathlib import Path
from types import SimpleNamespace
import os

from utils import clear_data, open_data, save_simulated_data

# Reference palette (light surface)
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e6e5e1"
_BLUE = "#2a78d6"    # volumes
_ORANGE = "#eb6834"  # returns

_EQ = ("volumes", "returns")   # column 0 = volumes, column 1 = returns


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
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


def plot_var_reduced_fit(results, out_dir=None, residuals=True, resid_step=10):
    """Plot the fit results (or of a model loaded from file).
    If out_dir is given, figures are saved there (var_fit.png, var_fit_cumulative.png, residuals.png)
    instead of being shown.

    2x2 grid of the coefficients vs lag: rows are the equations (volumes, returns),
    columns are the lagged regressors (volumes, returns).
    """
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

    if residuals:
        plot_residuals(np.asarray(fitted.resid),
                       save_path=None if out_dir is None else out_dir / "residuals.png",
                       step=resid_step)


# ----------------------------------------------------------------------------
# Fit on several days (no concatenation, no full design matrix in memory)
# ----------------------------------------------------------------------------
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


def var_reduced_fit(days, p=10, chunk=5000, plot=False):
    """Fit one reduced-form VAR(p) without intercept on [volumes, returns], using
    several days. Accumulates the normal equations X'X, X'y day by day, which is
    equivalent to OLS on the stacked rows, with memory O(chunk * p).

    days: list of tuples (returns, volumes), one per day.
    """
    k, m = 2, 2 * p
    XtX, XtY = np.zeros((m, m)), np.zeros((m, k))
    n = 0
    sy, sy2 = np.zeros(k), np.zeros(k)

    # Pass 1: accumulate
    for returns, volumes in days:
        for X, y in _blocks(returns, volumes, p, chunk):
            XtX += X.T @ X
            XtY += X.T @ y
            n += len(y)
            sy += y.sum(axis=0)
            sy2 += (y ** 2).sum(axis=0)

    if n == 0:
        raise ValueError("no day with length > p")

    # Solve on column-standardized normal equations: volumes (~1e3) and returns (~1e-4)
    # have very different scales, which would otherwise ruin the conditioning of X'X
    d = np.sqrt(np.diag(XtX))
    d[d == 0] = 1.0
    beta = np.linalg.lstsq(XtX / np.outer(d, d), XtY / d[:, None], rcond=None)[0] / d[:, None]   # (2p, 2)

    # Pass 2: residuals, block by block
    resid = []
    for returns, volumes in days:
        for X, y in _blocks(returns, volumes, p, chunk):
            resid.append(y - X @ beta)
    resid = np.concatenate(resid)                                       # (n, 2)

    # R² per equation: SST around the mean of y (the model has no intercept,
    # so the centered R² can be negative)
    ssr = (resid ** 2).sum(axis=0)
    sst = sy2 - sy ** 2 / n
    r2 = 1.0 - ssr / sst
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - m)
    sigma_u = resid.T @ resid / (n - m)

    # beta rows are (lag, regressor), columns are equations -> coefs (p, eq, regressor)
    coefs = beta.reshape(p, k, k).transpose(0, 2, 1)

    if plot:
        plot_residuals(resid)

    print(f"Volumes R2: {r2[0]:.4f} (adj {r2_adj[0]:.4f}) | "
          f"Returns R2: {r2[1]:.4f} (adj {r2_adj[1]:.4f})")

    return {
        "p": p,
        "n_obs": int(n),
        "model": SimpleNamespace(coefs=coefs, resid=resid, sigma_u=sigma_u, k_ar=p, neqs=k),
        "sigma_cov": sigma_u,
        "r2": {"volumes": float(r2[0]), "returns": float(r2[1])},
        "r2_adj": {"volumes": float(r2_adj[0]), "returns": float(r2_adj[1])},
    }


# ----------------------------------------------------------------------------
# Out-of-sample evaluation
# ----------------------------------------------------------------------------
def _oos_stats(n, sy, sy2, ssr, n_reg):
    """R2 around the mean of the evaluated sample."""
    if n == 0:
        return {"n": 0, "mse": np.nan, "rmse": np.nan, "rsquared": np.nan, "rsquared_adj": np.nan}
    sst = sy2 - sy ** 2 / n
    r2 = 1.0 - ssr / sst
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - n_reg)
    return {"n": int(n), "mse": ssr / n, "rmse": float(np.sqrt(ssr / n)),
            "rsquared": float(r2), "rsquared_adj": float(r2_adj)}


def var_evaluate(results, days, chunk=5000):
    """Out-of-sample evaluation on days not used for the fit.

    One-step-ahead predictions with the REAL lagged values of each test day
    (lags never cross day boundaries) and the fitted coefficients.
    Returns pooled stats over all test days and the R2 of each day, per equation.
    """
    p = results["p"]
    coefs = np.asarray(results["model"].coefs)                          # (p, eq, reg)
    beta = coefs.transpose(0, 2, 1).reshape(2 * p, 2)                   # (2p, eq)
    n_reg = 2 * p

    tot = {"n": 0, "sy": np.zeros(2), "sy2": np.zeros(2), "ssr": np.zeros(2)}
    per_day = {eq: [] for eq in _EQ}

    for returns, volumes in days:
        day = {"n": 0, "sy": np.zeros(2), "sy2": np.zeros(2), "ssr": np.zeros(2)}
        for X, y in _blocks(returns, volumes, p, chunk):
            e = y - X @ beta
            day["n"] += len(y)
            day["sy"] += y.sum(axis=0)
            day["sy2"] += (y ** 2).sum(axis=0)
            day["ssr"] += (e ** 2).sum(axis=0)
        for key in tot:
            tot[key] = tot[key] + day[key]
        for i, eq in enumerate(_EQ):
            per_day[eq].append(_oos_stats(day["n"], day["sy"][i], day["sy2"][i],
                                          day["ssr"][i], n_reg)["rsquared"])

    out = {}
    for i, eq in enumerate(_EQ):
        out[eq] = _oos_stats(tot["n"], tot["sy"][i], tot["sy2"][i], tot["ssr"][i], n_reg)
        out[eq]["per_day_rsquared"] = per_day[eq]
    return out


def save_oos_stats(results, oos, out_dir, window_info=None):
    """Save in-sample vs out-of-sample statistics as JSON, print a comparison table,
    plot the per-day R2."""
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
        ax.set_title(f"{title} equation - R² on test days", fontsize=11)
        ax.set_xlabel("test day")
        ax.set_ylabel("R²")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "oos_r2_per_day.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def simulate_var_reduced(results, n_steps, initial_v, initial_r,
                         initial_price=100.0, bootstrap=False,
                         center_residuals=True, seed=None):
    fitted = results["model"]
    p = results["p"]
    coefs = fitted.coefs                      # (p, k, k)
    k = coefs.shape[1]
    rng = np.random.default_rng(seed)

    # shocks: Gaussian with covariance sigma_u, or resampled residuals
    if bootstrap:
        resid = np.asarray(fitted.resid, dtype=np.float64)
        resid = resid[~np.isnan(resid).any(axis=1)]
        if center_residuals:
            resid = resid - resid.mean(axis=0)
        shocks = resid[rng.integers(0, len(resid), size=n_steps)]
    else:
        shocks = rng.multivariate_normal(np.zeros(k), fitted.sigma_u, size=n_steps)

    # VAR recursion (no intercept)
    y = np.empty((p + n_steps, k))
    y[:p] = np.column_stack([initial_v, initial_r])[-p:]
    for t in range(p, p + n_steps):
        y[t] = np.einsum("ikj,ij->k", coefs, y[t - p:t][::-1]) + shocks[t - p]

    sim_v, sim_r = y[p:, 0], y[p:, 1]
    prices = initial_price * np.concatenate([[1.0], np.cumprod(1 + sim_r[:-1])])
    return prices, sim_v, sim_r


# ----------------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------------
def save_var_reduced_model(results, path):
    """Save the results of var_reduced_fit in a .npz file (no pickle).

    Only the pieces used by simulate_var_reduced are saved
    (coefs, resid, sigma_u) plus p and the R2."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fitted = results["model"]
    meta = {
        "p": int(results["p"]),
        "n_obs": int(results["n_obs"]),
        "r2_volumes": float(results["r2"]["volumes"]),
        "r2_returns": float(results["r2"]["returns"]),
        "r2_adj_volumes": float(results["r2_adj"]["volumes"]),
        "r2_adj_returns": float(results["r2_adj"]["returns"]),
    }
    np.savez_compressed(
        path,
        meta=json.dumps(meta),
        coefs=np.asarray(fitted.coefs),
        resid=np.asarray(fitted.resid),
        sigma_u=np.asarray(fitted.sigma_u),
    )


def load_var_reduced_model(path):
    """Reload a model saved with save_var_reduced_model, same format as var_reduced_fit.

    results["model"] is a light object with attributes coefs, resid, sigma_u."""
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        coefs, resid, sigma_u = f["coefs"], f["resid"], f["sigma_u"]

    p = meta["p"]
    return {
        "p": p,
        "n_obs": meta["n_obs"],
        "model": SimpleNamespace(coefs=coefs, resid=resid, sigma_u=sigma_u,
                                 k_ar=p, neqs=coefs.shape[1]),
        "sigma_cov": sigma_u,
        "r2": {"volumes": meta["r2_volumes"], "returns": meta["r2_returns"]},
        "r2_adj": {"volumes": meta["r2_adj_volumes"], "returns": meta["r2_adj_returns"]},
    }


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
    fitting_window = 200          # number of trading days used for the fit
    bootstrap_residuals = True
    trim_top = True               # False restores the original behaviour
    top_frac = 0.01               # fraction of largest |volume| trades dropped each day

    name = f'var_reduced_{p}'

    dir = 'database\\data_' + name

    if not os.path.exists('..\\' + dir):
        os.makedirs('..\\' + dir)

    model_dir = Path('..') / 'models' / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'model.npz'      # a single model for all days

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
        results = load_var_reduced_model(model_path)
    else:
        print('Compute Model')
        days = []
        for path in fit_paths:
            _, r, v = load_day(path)
            days.append((r, v))
        results = var_reduced_fit(days, p=p)
        save_var_reduced_model(results, model_path)

    # ---- Plots saved next to the model ----
    plot_var_reduced_fit(results, out_dir=model_dir)

    # ---- Out-of-sample evaluation on the remaining days ----
    if len(test_paths) > 0:
        test_days = []
        for path in test_paths:
            _, r, v = load_day(path)
            test_days.append((r, v))
        oos = var_evaluate(results, test_days)
        save_oos_stats(results, oos, model_dir,
                       window_info={"fitting_window": fitting_window,
                                    "n_fit_days": len(fit_paths),
                                    "n_test_days": len(test_paths),
                                    "trim_top": trim_top,
                                    "top_frac": top_frac})
    else:
        print('No days left for out-of-sample evaluation')

    exit()

    clear_data(name)

    # ---- Simulation for each day with the common model ----
    for path in paths:
        print(path)
        prices, r, v = load_day(path)

        # Use the first 'p' values as the seed for the simulation
        initial_v = v[:p]
        initial_r = r[:p]
        initial_price = prices[p]

        prices_sim, volumes_sim, r_sim = simulate_var_reduced(
            results,
            n_steps=len(r),
            initial_v=initial_v,
            initial_r=initial_r,
            initial_price=initial_price,
            bootstrap=bootstrap_residuals
        )

        save_simulated_data(f"..\\{dir}\\" + path, prices_sim, volumes_sim)