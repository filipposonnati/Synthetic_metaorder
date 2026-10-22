"""
VAR strutturale (ricorsivo / Cholesky), caso lineare, modello unico su piu' giorni.

Ordinamento V -> R: nel tick t il flusso con segno V_t e' predeterminato rispetto al
rendimento R_t (con l'allineamento di load_day il trade t impatta r_t):

    V_t = sum_i a_i . [V_{t-i}, R_{t-i}]                       + u^V_t
    R_t = gamma * V_t + sum_i c_i . [V_{t-i}, R_{t-i}]         + e^R_t        u^V ⟂ e^R

Nessun intercetta (come var_reduced). Stima OLS equazione per equazione, accumulando le
equazioni normali giorno per giorno (nessuna concatenazione, memoria O(chunk * p)).

Relazione con la forma ridotta (stessi regressori ritardati X):
    coefs_ridotti[V] = a
    coefs_ridotti[R] = c + gamma * a           (OLS di R_t su X, identita' esatta)
    Sigma_ridotta    = [[s2V, gamma s2V], [gamma s2V, gamma^2 s2V + s2R]]

R2 dei rendimenti riportati:
    - "returns"        : condizionale a V_t (nowcast, confrontabile col TIM)
    - "returns_lagged" : solo passato, one-step-ahead; coincide con l'R2 di var_reduced
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from pathlib import Path
from types import SimpleNamespace
import os

from var_reduced import _blocks, _oos_stats, plot_residuals, trim_top_volume
from utils import clear_data, open_data, save_simulated_data

_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e6e5e1"
_BLUE = "#2a78d6"    # volumes
_ORANGE = "#eb6834"  # returns

_SERIES = ("volumes", "returns", "returns_lagged")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _solve(G, b):
    """Risolve G beta = b con colonne standardizzate (volumi ~1e3, rendimenti ~1e-4 hanno
    scale molto diverse). Ritorna (beta, d) con d = sqrt(diag(G))."""
    d = np.sqrt(np.diag(G))
    d[d == 0] = 1.0
    beta = np.linalg.lstsq(G / np.outer(d, d), b / d, rcond=None)[0] / d
    return beta, d


def reduced_coefs(coefs, gamma):
    """Coefficienti ritardati della forma ridotta (p, eq, reg) dai strutturali."""
    red = np.array(coefs, dtype=float)
    red[:, 1, :] = coefs[:, 1, :] + gamma * coefs[:, 0, :]
    return red


def reduced_sigma(gamma, s2_v, s2_r):
    return np.array([[s2_v, gamma * s2_v],
                     [gamma * s2_v, gamma ** 2 * s2_v + s2_r]])


def _r2(ssr, sst, n, k):
    r2 = 1.0 - ssr / sst
    return float(r2), float(1.0 - (1.0 - r2) * (n - 1) / (n - k))


# ----------------------------------------------------------------------------
# Fit su piu' giorni
# ----------------------------------------------------------------------------
def var_structural_fit(days, p=10, chunk=5000, plot=False):
    """Fit del VAR strutturale lineare V -> R su piu' giorni.

    days: lista di tuple (returns, volumes), una per giorno (volumes con segno)."""
    m = 2 * p
    XtX, XtY, YtY = np.zeros((m, m)), np.zeros((m, 2)), np.zeros((2, 2))
    n = 0
    sy, sy2 = np.zeros(2), np.zeros(2)

    # Pass 1: statistiche sufficienti (y = [V_t, R_t])
    for returns, volumes in days:
        for X, y in _blocks(returns, volumes, p, chunk):
            XtX += X.T @ X
            XtY += X.T @ y
            YtY += y.T @ y
            n += len(y)
            sy += y.sum(axis=0)
            sy2 += (y ** 2).sum(axis=0)
    if n == 0:
        raise ValueError("no day with length > p")

    # Equazione dei volumi: V_t su X
    a_vec, d_x = _solve(XtX, XtY[:, 0])

    # Equazione dei rendimenti: R_t su [X, V_t]
    G = np.zeros((m + 1, m + 1))
    G[:m, :m] = XtX
    G[:m, m] = G[m, :m] = XtY[:, 0]
    G[m, m] = YtY[0, 0]
    rhs = np.append(XtY[:, 1], YtY[0, 1])
    sol, d_r = _solve(G, rhs)
    c_vec, gamma = sol[:m], float(sol[m])
    b_red = c_vec + gamma * a_vec                      # R_t su X (forma ridotta)

    # Pass 2: residui
    resid, ssr_lag = [], 0.0
    for returns, volumes in days:
        for X, y in _blocks(returns, volumes, p, chunk):
            e_v = y[:, 0] - X @ a_vec
            e_r = y[:, 1] - gamma * y[:, 0] - X @ c_vec
            e_l = y[:, 1] - X @ b_red
            resid.append(np.column_stack([e_v, e_r]))
            ssr_lag += e_l @ e_l
    resid = np.concatenate(resid)                       # (n, 2): [u^V, e^R]

    ssr = (resid ** 2).sum(axis=0)
    sst = sy2 - sy ** 2 / n
    r2_v, r2a_v = _r2(ssr[0], sst[0], n, m)
    r2_r, r2a_r = _r2(ssr[1], sst[1], n, m + 1)
    r2_l, r2a_l = _r2(ssr_lag, sst[1], n, m)
    s2_v, s2_r = float(ssr[0] / (n - m)), float(ssr[1] / (n - m - 1))

    # beta righe (lag, regressore) -> (p, regressore); coefs (p, eq, reg)
    coefs = np.stack([a_vec.reshape(p, 2), c_vec.reshape(p, 2)], axis=1)

    # Coefficienti standardizzati (regressori e dipendente divisi per il loro RMS)
    d_y = np.sqrt(sy2)
    d_y[d_y == 0] = 1.0
    coefs_std = np.stack([(a_vec * d_x / d_y[0]).reshape(p, 2),
                          (c_vec * d_r[:m] / d_y[1]).reshape(p, 2)], axis=1)
    gamma_std = float(gamma * d_r[m] / d_y[1])

    if plot:
        plot_residuals(resid)

    print(f"Volumes R2: {r2_v:.4f} (adj {r2a_v:.4f}) | "
          f"Returns R2 (cond. a V_t): {r2_r:.4f} (adj {r2a_r:.4f}) | "
          f"Returns R2 (solo passato): {r2_l:.4f} (adj {r2a_l:.4f}) | gamma = {gamma:.4e}")

    return {
        "p": p,
        "n_obs": int(n),
        "model": SimpleNamespace(coefs=coefs, gamma=gamma, coefs_std=coefs_std,
                                 gamma_std=gamma_std, resid=resid,
                                 sigma_u=np.diag([s2_v, s2_r]), k_ar=p, neqs=2),
        "sigma_cov": reduced_sigma(gamma, s2_v, s2_r),
        "r2": {"volumes": r2_v, "returns": r2_r, "returns_lagged": r2_l},
        "r2_adj": {"volumes": r2a_v, "returns": r2a_r, "returns_lagged": r2a_l},
        "rmse_lagged": float(np.sqrt(ssr_lag / n)),
    }


# ----------------------------------------------------------------------------
# Out-of-sample
# ----------------------------------------------------------------------------
def var_evaluate(results, days, chunk=5000):
    """Valutazione out-of-sample one-step-ahead sui giorni di test (lag interni al giorno).

    returns         : R_t previsto con V_t reale del giorno di test e i lag reali
    returns_lagged  : R_t previsto solo dal passato (forma ridotta)"""
    p = results["p"]
    mdl = results["model"]
    coefs, gamma = np.asarray(mdl.coefs), float(mdl.gamma)
    a_vec = coefs[:, 0, :].reshape(2 * p)
    c_vec = coefs[:, 1, :].reshape(2 * p)
    b_red = c_vec + gamma * a_vec
    n_reg = {"volumes": 2 * p, "returns": 2 * p + 1, "returns_lagged": 2 * p}

    def zeros():
        return {"n": 0, "sy": np.zeros(3), "sy2": np.zeros(3), "ssr": np.zeros(3)}

    # colonne: 0 volumes, 1 returns, 2 returns_lagged
    tot, per_day = zeros(), {s: [] for s in _SERIES}
    for returns, volumes in days:
        day = zeros()
        for X, y in _blocks(returns, volumes, p, chunk):
            e = np.column_stack([y[:, 0] - X @ a_vec,
                                 y[:, 1] - gamma * y[:, 0] - X @ c_vec,
                                 y[:, 1] - X @ b_red])
            tgt = np.column_stack([y[:, 0], y[:, 1], y[:, 1]])
            day["n"] += len(y)
            day["sy"] += tgt.sum(axis=0)
            day["sy2"] += (tgt ** 2).sum(axis=0)
            day["ssr"] += (e ** 2).sum(axis=0)
        for key in tot:
            tot[key] = tot[key] + day[key]
        for i, s in enumerate(_SERIES):
            per_day[s].append(_oos_stats(day["n"], day["sy"][i], day["sy2"][i],
                                         day["ssr"][i], n_reg[s])["rsquared"])

    out = {}
    for i, s in enumerate(_SERIES):
        out[s] = _oos_stats(tot["n"], tot["sy"][i], tot["sy2"][i], tot["ssr"][i], n_reg[s])
        out[s]["per_day_rsquared"] = per_day[s]
    return out


def save_oos_stats(results, oos, out_dir, window_info=None):
    """JSON in-sample vs out-of-sample, tabella di confronto, R2 per giorno."""
    out_dir = Path(out_dir)
    resid = np.asarray(results["model"].resid)
    rmse_in = {"volumes": float(np.sqrt(np.mean(resid[:, 0] ** 2))),
               "returns": float(np.sqrt(np.mean(resid[:, 1] ** 2))),
               "returns_lagged": float(results["rmse_lagged"])}
    summary = {"window_info": window_info or {}}

    print(f"\n{'series':<16}{'sample':<14}{'n':>10}{'R2':>10}{'R2 adj':>10}{'RMSE':>14}")
    for s in _SERIES:
        ins_row = {"n": int(len(resid)), "rmse": rmse_in[s],
                   "rsquared": float(results["r2"][s]),
                   "rsquared_adj": float(results["r2_adj"][s])}
        oos_row = {k: v for k, v in oos[s].items() if k != "mse"}
        oos_row["per_day_rsquared"] = [float(x) for x in oos[s]["per_day_rsquared"]]
        summary[s] = {"in_sample": ins_row, "out_of_sample": oos_row}
        for label, r in (("in-sample", ins_row), ("out-of-sample", oos_row)):
            print(f"{s:<16}{label:<14}{r['n']:>10}{r['rsquared']:>10.4f}"
                  f"{r['rsquared_adj']:>10.4f}{r['rmse']:>14.4e}")

    with open(out_dir / "oos_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    titles = ("Volumes", "Returns (cond. on V_t)", "Returns (lagged only)")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, s, title in zip(axes, _SERIES, titles):
        r2_days = summary[s]["out_of_sample"]["per_day_rsquared"]
        ax.plot(np.arange(1, len(r2_days) + 1), r2_days, lw=1.2, label="out-of-sample (per day)")
        ax.axhline(summary[s]["in_sample"]["rsquared"], color='red', ls='--', label="in-sample")
        ax.axhline(summary[s]["out_of_sample"]["rsquared"], color='green', ls=':',
                   label="out-of-sample (pooled)")
        ax.set_title(f"{title} - R² on test days", fontsize=11)
        ax.set_xlabel("test day")
        ax.set_ylabel("R²")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "oos_r2_per_day.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_var_structural_fit(results, out_dir=None, residuals=True, resid_step=10):
    """Coefficienti vs lag. Righe = equazioni (volumes, returns), colonne = regressori.
    Il pannello V(t-k) -> R(t) parte dal lag 0 (gamma, effetto contemporaneo).
    Salva var_fit.png, var_fit_cumulative.png, var_fit_std.png, var_fit_cumulative_std.png,
    residuals.png in out_dir (altrimenti mostra)."""
    if out_dir is not None:
        out_dir = Path(out_dir)
    p = results["p"]
    mdl = results["model"]
    names = ("Volumes", "Returns")
    colors = (_BLUE, _ORANGE)
    r2, r2_adj = results["r2"], results["r2_adj"]
    sig = np.asarray(results["sigma_cov"])
    corr_red = sig[0, 1] / np.sqrt(sig[0, 0] * sig[1, 1])

    def full(coefs, gamma):
        """(p+1, eq, reg): lag 0 = gamma solo per (R <- V), altrove 0 (non esiste)."""
        out = np.full((p + 1, 2, 2), np.nan)
        out[1:] = coefs
        out[0, 1, 0] = gamma
        return out

    def draw_grid(values, ylabel, title_fmt, suptitle, filename, annotate=False):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.patch.set_facecolor(_SURFACE)
        for eq in range(2):
            for reg in range(2):
                ax = axes[eq, reg]
                ax.set_facecolor(_SURFACE)
                y = values[:, eq, reg]
                ok = ~np.isnan(y)
                lags = np.arange(p + 1)[ok]
                if p <= 30:
                    ax.stem(lags, y[ok], basefmt=" ")
                else:
                    ax.plot(lags, y[ok], lw=1.2, color=colors[reg])
                    if lags[0] == 0:
                        ax.plot(lags[:1], y[ok][:1], 'o', color=_INK, ms=4)
                ax.axhline(0, color=_INK_2, lw=0.8)
                ax.set_title(title_fmt.format(reg=names[reg], eq=names[eq]),
                             fontsize=11, color=_INK)
                ax.set_xlabel("lag k")
                ax.set_ylabel(ylabel)
                ax.grid(True, color=_GRID, alpha=0.8)
                if annotate and reg == 0:
                    if eq == 0:
                        txt = f"in-sample R² = {r2['volumes']:.3f} (adj {r2_adj['volumes']:.3f})"
                    else:
                        txt = (f"R² cond. V_t = {r2['returns']:.3f} (adj {r2_adj['returns']:.3f})\n"
                               f"R² solo passato = {r2['returns_lagged']:.3f}")
                    ax.annotate(txt, xy=(0.02, 0.95), xycoords="axes fraction", va="top",
                                fontsize=9, color=_INK_2)
        fig.suptitle(suptitle, fontsize=12, color=_INK)
        plt.tight_layout()
        if out_dir is not None:
            fig.savefig(out_dir / filename, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    base = f"Structural VAR({p}) V→R"
    vals = full(np.asarray(mdl.coefs), mdl.gamma)
    draw_grid(vals, "coefficient", "{reg}(t-k) → {eq}(t)",
              f"{base} - gamma = {mdl.gamma:.3e} - reduced-form residual corr = {corr_red:.3f}",
              "var_fit.png", annotate=True)
    draw_grid(np.nancumsum(vals, axis=0), "cumulative sum of coefficients",
              "Cumulative {reg}(t-k) → {eq}(t)", f"{base} - cumulative coefficients",
              "var_fit_cumulative.png")

    if getattr(mdl, "coefs_std", None) is None:
        print("coefs_std not available: refit to get the standardized figures")
    else:
        vals_s = full(np.asarray(mdl.coefs_std), mdl.gamma_std)
        draw_grid(vals_s, "standardized coefficient", "{reg}(t-k) → {eq}(t)",
                  f"{base} - standardized coefficients (comparable)", "var_fit_std.png")
        draw_grid(np.nancumsum(vals_s, axis=0), "cumulative standardized coefficient",
                  "Cumulative {reg}(t-k) → {eq}(t)",
                  f"{base} - cumulative standardized coefficients", "var_fit_cumulative_std.png")

    if residuals:
        plot_residuals(np.asarray(mdl.resid),
                       save_path=None if out_dir is None else out_dir / "residuals.png",
                       step=resid_step)


# ----------------------------------------------------------------------------
# Simulazione
# ----------------------------------------------------------------------------
def simulate_var_structural(results, n_steps, initial_v, initial_r,
                            initial_price=100.0, bootstrap=False,
                            center_residuals=True, seed=None):
    """Genera prima V_t, poi R_t = gamma V_t + (parte ritardata) + e^R_t.

    bootstrap=True  -> stessa riga dei residui (u^V, e^R) per entrambi: conserva la
                       dipendenza di ordine superiore (code, eteroschedasticita')
    bootstrap=False -> shock gaussiani indipendenti con varianze s2V, s2R
    Ritorna (prices, sim_v, sim_r); prices[k+1] = prices[k] * (1 + sim_r[k])."""
    mdl = results["model"]
    p = results["p"]
    coefs, gamma = np.asarray(mdl.coefs), float(mdl.gamma)       # (p, eq, reg)
    rng = np.random.default_rng(seed)

    if bootstrap:
        resid = np.asarray(mdl.resid, dtype=np.float64)
        resid = resid[~np.isnan(resid).any(axis=1)]
        if center_residuals:
            resid = resid - resid.mean(axis=0)
        shocks = resid[rng.integers(0, len(resid), size=n_steps)]
    else:
        sd = np.sqrt(np.diag(np.asarray(mdl.sigma_u)))
        shocks = rng.normal(size=(n_steps, 2)) * sd

    y = np.empty((p + n_steps, 2))
    y[:p] = np.column_stack([initial_v, initial_r])[-p:]
    for t in range(p, p + n_steps):
        lagged = np.einsum("ikj,ij->k", coefs, y[t - p:t][::-1])  # parte ritardata di (V, R)
        y[t, 0] = lagged[0] + shocks[t - p, 0]
        y[t, 1] = gamma * y[t, 0] + lagged[1] + shocks[t - p, 1]

    sim_v, sim_r = y[p:, 0], y[p:, 1]
    prices = initial_price * np.concatenate([[1.0], np.cumprod(1 + sim_r[:-1])])
    return prices, sim_v, sim_r


# ----------------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------------
def save_var_structural_model(results, path):
    """Salva in .npz (nessun pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mdl = results["model"]
    meta = {"p": int(results["p"]), "n_obs": int(results["n_obs"]),
            "gamma": float(mdl.gamma), "gamma_std": float(mdl.gamma_std),
            "rmse_lagged": float(results["rmse_lagged"])}
    for s in _SERIES:
        meta[f"r2_{s}"] = float(results["r2"][s])
        meta[f"r2_adj_{s}"] = float(results["r2_adj"][s])
    np.savez_compressed(path, meta=json.dumps(meta),
                        coefs=np.asarray(mdl.coefs), coefs_std=np.asarray(mdl.coefs_std),
                        resid=np.asarray(mdl.resid), sigma_u=np.asarray(mdl.sigma_u))


def load_var_structural_model(path):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        coefs, coefs_std = f["coefs"], f["coefs_std"]
        resid, sigma_u = f["resid"], f["sigma_u"]
    p, gamma = meta["p"], meta["gamma"]
    return {
        "p": p,
        "n_obs": meta["n_obs"],
        "model": SimpleNamespace(coefs=coefs, gamma=gamma, coefs_std=coefs_std,
                                 gamma_std=meta["gamma_std"], resid=resid,
                                 sigma_u=sigma_u, k_ar=p, neqs=2),
        "sigma_cov": reduced_sigma(gamma, sigma_u[0, 0], sigma_u[1, 1]),
        "r2": {s: meta[f"r2_{s}"] for s in _SERIES},
        "r2_adj": {s: meta[f"r2_adj_{s}"] for s in _SERIES},
        "rmse_lagged": meta["rmse_lagged"],
    }


# ----------------------------------------------------------------------------
# Test di equivalenza con la forma ridotta (stessi giorni)
# ----------------------------------------------------------------------------
def check_equivalence(days, p, results=None, tol=1e-6):
    """Confronta la forma ridotta implicita nel modello strutturale con var_reduced_fit
    sugli stessi giorni: coefficienti, Sigma_u e R2 (solo passato)."""
    from var_reduced import var_reduced_fit
    if results is None:
        results = var_structural_fit(days, p=p)
    red = var_reduced_fit(days, p=p)
    mdl = results["model"]
    c_diff = np.abs(reduced_coefs(np.asarray(mdl.coefs), mdl.gamma)
                    - np.asarray(red["model"].coefs)).max()
    c_scale = np.abs(np.asarray(red["model"].coefs)).max()
    r2_diff = abs(results["r2"]["returns_lagged"] - red["r2"]["returns"])
    r2v_diff = abs(results["r2"]["volumes"] - red["r2"]["volumes"])
    ok = (c_diff <= tol * max(c_scale, 1.0)) and r2_diff < tol and r2v_diff < tol
    print(f"[equivalenza] max|coefs_ridotti - var_reduced| = {c_diff:.3e} (scala {c_scale:.3e}) | "
          f"|dR2 returns_lagged| = {r2_diff:.2e} | |dR2 volumes| = {r2v_diff:.2e} -> "
          f"{'OK' if ok else 'ATTENZIONE'}")
    return ok


if __name__ == '__main__':
    paths = np.array(sorted(listdir('..\\database\\data')))   # sorted -> chronological order

    p = 1000
    fitting_window = 100          # number of trading days used for the fit
    bootstrap_residuals = True
    trim_top = True               # False restores the original behaviour
    top_frac = 0.05               # fraction of largest |volume| trades dropped each day
    check_reduced = False         # True: verifica l'equivalenza con var_reduced sugli stessi giorni

    name = f'var_structural_{p}'

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
    fit_days = None
    if model_path.exists():
        print('Load Model')
        results = load_var_structural_model(model_path)
    else:
        print('Compute Model')
        fit_days = []
        for path in fit_paths:
            _, r, v = load_day(path)
            fit_days.append((r, v))
        results = var_structural_fit(fit_days, p=p)
        save_var_structural_model(results, model_path)

    if check_reduced:
        if fit_days is None:
            fit_days = [load_day(path)[1:] for path in fit_paths]
        check_equivalence(fit_days, p, results)

    # ---- Plots saved next to the model ----
    plot_var_structural_fit(results, out_dir=model_dir)

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

    clear_data(name)

    # ---- Simulation for each day with the common model ----
    for path in paths:
        print(path)
        prices, r, v = load_day(path)

        # Use the first 'p' values as the seed for the simulation
        initial_v = v[:p]
        initial_r = r[:p]
        initial_price = prices[p]

        prices_sim, volumes_sim, r_sim = simulate_var_structural(
            results,
            n_steps=len(r),
            initial_v=initial_v,
            initial_r=initial_r,
            initial_price=initial_price,
            bootstrap=bootstrap_residuals
        )

        save_simulated_data(f"..\\{dir}\\" + path, prices_sim, volumes_sim)