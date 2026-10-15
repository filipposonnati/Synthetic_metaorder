"""
TIM parametrico con modello unico (un solo fit su più giorni).

  - Volumi    : AR(p) con costante sui volumi con segno, come in tim.py
                (simulati, con shock bootstrap o gaussiani).
  - Rendimenti: Transient Impact Model parametrico, come in real_data_price_gen.py.
                Propagatore G(l) = (l+1)^(-beta), l = 0..L-1, sul livello del log-prezzo;
                il regressore dei log-rendimenti e' il kernel differenziato
                K(l) = G(l) - G(l-1)  (G(-1) = 0, G(L) = 0, lunghezza L+1):

                    r_t = c + sigma_f * X_t + eta_t,
                    X_t = sum_{l=0..L} K(l) * sign(v_{t-l}) |v_{t-l}|^delta

                Parametri stimati con OLS: c e sigma_f. beta, delta, L sono iperparametri
                (beta puo' essere scelto con una griglia massimizzando l'R2 in-sample).
  - Prezzi    : P_{t+1} = P_t * exp(r_t)

Allineamento: come in tim.py v_t = volume_t * segno_t e r_t = log P_{t+1} - log P_t
(equivale a impact_lag = 1 in real_data_price_gen.py: il trade t impatta r_t).

Un solo modello per tutti i giorni: i lag/kernel non attraversano i confini di giornata
e le prime max(p, L) osservazioni di ogni giorno sono scartate dal fit (burn-in),
cosi' il regressore X_t e' sempre calcolato con il kernel completo.

Salva nella cartella del modello:
  tim_param_fit.png : parametri di best fit (AR volumi, kernel TIM, cumulate/propagatore)
  residuals.png     : istogrammi dei residui (volumi e rendimenti)
  oos_stats.json / oos_r2_per_day.png : R2 in-sample e out-of-sample
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from pathlib import Path
import os
from numpy.lib.stride_tricks import sliding_window_view

try:
    from scipy.signal import fftconvolve as _conv
except ImportError:                                   # fallback lento ma corretto
    def _conv(a, b):
        return np.convolve(a, b, mode="full")

from utils import clear_data, open_data, save_simulated_data


# ----------------------------------------------------------------------------
# Plot di base
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


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def power_transform(v, delta):
    """Funzione di impatto sign(v) * |v|^delta.
    delta = 1 -> lineare, delta = 0 -> solo il segno, 0<delta<1 -> power-law concavo."""
    return np.sign(v) * np.abs(v) ** delta


def _model_stats(resid, k, n, sy, sy2):
    """sigma2 residua, R2 (attorno alla media di y) e R2 adjusted."""
    ssr = resid @ resid
    sst = sy2 - sy ** 2 / n
    r2 = 1.0 - ssr / sst
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    return ssr / (n - k), r2, r2_adj


def _oos_stats(acc, k):
    """acc = [n, sum(y), sum(y^2), SSR]. R2 attorno alla media del campione valutato."""
    n, sy, sy2, ssr = acc
    if n == 0:
        return {"n": 0, "mse": np.nan, "rmse": np.nan, "rsquared": np.nan, "rsquared_adj": np.nan}
    sst = sy2 - sy ** 2 / n
    r2 = 1.0 - ssr / sst
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    return {"n": int(n), "mse": ssr / n, "rmse": np.sqrt(ssr / n),
            "rsquared": r2, "rsquared_adj": r2_adj}


# ----------------------------------------------------------------------------
# Volumi: blocchi, fit AR(p), valutazione, simulazione
# ----------------------------------------------------------------------------
def _volume_blocks(volumes, p, chunk):
    """Genera blocchi (Xv, yv) di al piu' `chunk` righe per un giorno.
    Xv = [1, V_{t-1}, ..., V_{t-p}], yv = V_t. I lag restano dentro il giorno."""
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


def _fit_scaled(block_iter_factory, k):
    """OLS su piu' giorni via equazioni normali accumulate (memoria O(chunk * k)),
    risolte con le colonne standardizzate: volumi (~1e3) e rendimenti (~1e-4) hanno scale
    molto diverse e peggiorerebbero il condizionamento di X'X.
    block_iter_factory: funzione senza argomenti che restituisce un iteratore di blocchi
    (X, y) su tutti i giorni; viene chiamata due volte (accumulo e residui)."""
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

    d = np.sqrt(np.diag(XtX))
    d[d == 0] = 1.0
    beta = np.linalg.lstsq(XtX / np.outer(d, d), Xty / d, rcond=None)[0] / d

    resid = np.concatenate([y - X @ beta for X, y in block_iter_factory()])
    s2, r2, r2a = _model_stats(resid, k, n, sy, sy2)
    return beta, s2, r2, r2a, resid


def _make_shocks(resid, sigma2, n_steps, seed, bootstrap):
    """Shock per n_steps passi: bootstrap dei residui oppure gaussiani con varianza sigma2."""
    rng = np.random.default_rng(seed)
    if bootstrap:
        return np.asarray(resid)[rng.integers(0, len(resid), size=n_steps)]
    return rng.normal(0, np.sqrt(sigma2), n_steps)


def volume_simulate(volume_model, n_steps, initial_v, seed=None, bootstrap=False):
    """Simula n_steps volumi dal modello AR(p).
    initial_v: array cronologico (dal piu' vecchio al piu' recente) di lunghezza p.
    Ritorna sim_v di lunghezza n_steps (i p valori iniziali sono esclusi)."""
    p = volume_model["p"]
    initial_v = np.asarray(initial_v, dtype=float)
    if initial_v.shape != (p,):
        raise ValueError(f"initial_v deve avere lunghezza p={p}")

    params = np.asarray(volume_model["params"])
    c, a = params[0], params[1:]
    shocks = _make_shocks(volume_model["resid"], volume_model["sigma2"],
                          n_steps, seed, bootstrap)

    sim_v = np.zeros(n_steps + p)
    sim_v[:p] = initial_v
    for t in range(p, n_steps + p):
        sim_v[t] = c + a @ sim_v[t - p:t][::-1] + shocks[t - p]   # V_{t-1}, ..., V_{t-p}
    return sim_v[p:]


def volume_evaluate(volume_model, volume_days, chunk=10000):
    """Valutazione out-of-sample (one-step-ahead) del modello dei volumi.
    volume_days: lista di array di volumi, uno per giorno."""
    p = volume_model["p"]
    beta = np.asarray(volume_model["params"])
    tot, per_day = np.zeros(4), []
    for v in volume_days:
        day = np.zeros(4)
        for X, y in _volume_blocks(v, p, chunk):
            e = y - X @ beta
            day += np.array([len(y), y.sum(), y @ y, e @ e])
        tot += day
        per_day.append(_oos_stats(day, len(beta))["rsquared"])
    out = _oos_stats(tot, len(beta))
    out["per_day_rsquared"] = per_day
    return out


# ----------------------------------------------------------------------------
# Salvataggio statistiche in-sample vs out-of-sample
# ----------------------------------------------------------------------------
def save_oos_stats(results, oos, out_dir, window_info=None):
    """Salva in JSON le statistiche in-sample vs out-of-sample, stampa la tabella
    di confronto e disegna l'R2 per giorno."""
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m, title in zip(axes, ("volume_model", "return_model"), ("Volumes", "Returns")):
        r2_days = summary[m]["out_of_sample"]["per_day_rsquared"]
        ax.plot(np.arange(1, len(r2_days) + 1), r2_days, lw=1.2, label="out-of-sample (per day)")
        ax.axhline(summary[m]["in_sample"]["rsquared"], color='red', ls='--', label="in-sample")
        ax.axhline(summary[m]["out_of_sample"]["rsquared"], color='green', ls=':',
                   label="out-of-sample (pooled)")
        ax.set_title(f"{title} model - R² on test days", fontsize=11)
        ax.set_xlabel("test day")
        ax.set_ylabel("R²")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "oos_r2_per_day.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Filtro dati
# ----------------------------------------------------------------------------
def trim_top_volume(prices, volumes, signs, top_frac=0.05):
    """Elimina le transazioni con |volume| nel top `top_frac` del giorno.
    Applicato sui tick grezzi, prima di calcolare i rendimenti, cosi' i tick rimasti
    formano una serie coerente (rendimenti tra tick consecutivi mantenuti)."""
    prices, volumes, signs = map(np.asarray, (prices, volumes, signs))
    thr = np.quantile(np.abs(volumes), 1.0 - top_frac)
    keep = np.abs(volumes) <= thr
    return prices[keep], volumes[keep], signs[keep]


# ----------------------------------------------------------------------------
# TIM: kernel e regressore
# ----------------------------------------------------------------------------
def return_kernel(beta, kernel_L):
    """K = diff del propagatore G(l) = (l+1)^(-beta), l = 0..L-1, troncato.
    Lunghezza L+1 (lag 0..L)."""
    G = (np.arange(kernel_L, dtype=float) + 1.0) ** (-beta)
    return np.diff(np.concatenate(([0.0], G, [0.0])))


def _signal_from_transformed(tv, K, burn):
    """X_t = sum_l K(l) tv[t-l] per t = burn..n-1 (somme parziali prima di burn)."""
    return _conv(tv, K)[:len(tv)][burn:]


def _burn(p, kernel_L):
    return max(p, kernel_L)


# ----------------------------------------------------------------------------
# FIT (un solo modello su tutti i giorni)
# ----------------------------------------------------------------------------
def _prepare(days, delta, burn):
    """Per ogni giorno: (tv = T(v), y = r[burn:]). Salta i giorni troppo corti."""
    out = []
    for r, v in days:
        r = np.asarray(r, dtype=float)
        v = np.asarray(v, dtype=float)
        if len(r) != len(v):
            raise ValueError("returns e volumes devono avere la stessa lunghezza")
        if len(v) > burn:
            out.append((power_transform(v, delta), r[burn:]))
    if not out:
        raise ValueError("nessun giorno con lunghezza > max(p, L)")
    return out


def _suff_stats(prepared, K, burn):
    """n, Sx, Sy, Sxx, Sxy, Syy accumulati giorno per giorno."""
    n, Sx, Sy, Sxx, Sxy, Syy = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    for tv, y in prepared:
        x = _signal_from_transformed(tv, K, burn)
        n += len(y)
        Sx += x.sum(); Sy += y.sum()
        Sxx += x @ x; Sxy += x @ y; Syy += y @ y
    return n, Sx, Sy, Sxx, Sxy, Syy


def _solve(stats):
    """OLS y = c + sigma_f * x dalle statistiche sufficienti. Ritorna (c, sigma_f, r2)."""
    n, Sx, Sy, Sxx, Sxy, Syy = stats
    xm, ym = Sx / n, Sy / n
    sxx, sxy, syy = Sxx - n * xm * xm, Sxy - n * xm * ym, Syy - n * ym * ym
    sigma_f = sxy / sxx if sxx > 0 else 0.0
    c = ym - sigma_f * xm
    r2 = 1.0 - (syy - sigma_f * sxy) / syy if syy > 0 else 0.0
    return c, sigma_f, r2


def tim_param_fit(days, p=10, beta=0.25, delta=1.0, kernel_L=500,
                  beta_grid=None, chunk=10000):
    """Fit unico del TIM parametrico + AR(p) dei volumi.

    days     : lista di tuple (returns, volumes), una per giorno (returns = log-rendimenti).
    beta_grid: se dato, beta e' scelto sulla griglia massimizzando l'R2 in-sample dei
               rendimenti (altrimenti beta e' fissato).
    """
    if kernel_L > p:
        print(f"ATTENZIONE: kernel_L={kernel_L} > p={p}: nella simulazione il kernel parte "
              f"con storia incompleta dei volumi (initial_v ha solo p valori)")
    burn = _burn(p, kernel_L)

    # ---- volumi: AR(p) con costante (stesso modello di tim.py) ----
    b_v, s2_v, r2_v, r2a_v, res_v = _fit_scaled(
        lambda: (b for _, v in days for b in _volume_blocks(v, p, chunk)), p + 1)

    # ---- rendimenti: TIM parametrico ----
    prepared = _prepare(days, delta, burn)
    n_par = 2                                   # c, sigma_f
    if beta_grid is not None:
        scores = []
        for b in beta_grid:
            _, _, r2_b = _solve(_suff_stats(prepared, return_kernel(b, kernel_L), burn))
            scores.append(r2_b)
            print(f"  beta={b:.3f}  R2={r2_b:.6f}")
        beta = float(beta_grid[int(np.argmax(scores))])
        n_par = 3                               # anche beta
        print(f"beta scelto: {beta:.3f}")

    K = return_kernel(beta, kernel_L)
    stats = _suff_stats(prepared, K, burn)
    c, sigma_f, _ = _solve(stats)

    # secondo passaggio: residui diretti (evita cancellazioni numeriche nell'SSR)
    resid_r = np.concatenate([y - c - sigma_f * _signal_from_transformed(tv, K, burn)
                              for tv, y in prepared])
    n, _, Sy, _, _, Syy = stats
    s2_r, r2_r, r2a_r = _model_stats(resid_r, n_par, n, Sy, Syy)

    print(f"Volumes R2: {r2_v:.4f} (adj {r2a_v:.4f}) | "
          f"Returns R2: {r2_r:.4f} (adj {r2a_r:.4f})")
    print(f"TIM: beta={beta:.4f}  delta={delta}  L={kernel_L}  "
          f"c={c:.4e}  sigma_f={sigma_f:.4e}  sigma_eta={np.sqrt(s2_r):.4e}")

    return {
        "p": p, "delta": delta, "beta": beta, "kernel_L": kernel_L,
        "volume_model": {"p": p, "params": b_v, "sigma2": s2_v, "resid": res_v,
                         "rsquared": r2_v, "rsquared_adj": r2a_v},
        "return_model": {"p": p, "delta": delta, "beta": beta, "kernel_L": kernel_L,
                         "n_par": n_par, "const": c, "sigma_f": sigma_f,
                         "params": np.array([c, sigma_f]),
                         "sigma2": s2_r, "resid": resid_r,
                         "rsquared": r2_r, "rsquared_adj": r2a_r},
    }


# ----------------------------------------------------------------------------
# Valutazione out-of-sample
# ----------------------------------------------------------------------------
def return_param_evaluate(rm, days):
    """Previsione one-step-ahead dei rendimenti con regressori = volumi reali del giorno
    di test e (c, sigma_f) stimati. R2 attorno alla media del campione valutato."""
    K = return_kernel(rm["beta"], rm["kernel_L"])
    burn = _burn(rm["p"], rm["kernel_L"])
    k = rm["n_par"]
    tot, per_day = np.zeros(4), []
    for r, v in days:
        r = np.asarray(r, dtype=float)
        v = np.asarray(v, dtype=float)
        day = np.zeros(4)
        if len(v) > burn:
            x = _signal_from_transformed(power_transform(v, rm["delta"]), K, burn)
            y = r[burn:]
            e = y - rm["const"] - rm["sigma_f"] * x
            day = np.array([len(y), y.sum(), y @ y, e @ e])
        tot += day
        per_day.append(_oos_stats(day, k)["rsquared"])
    out = _oos_stats(tot, k)
    out["per_day_rsquared"] = per_day
    return out


def tim_param_evaluate(results, days, chunk=10000):
    """Out-of-sample di entrambi i modelli. days: lista di tuple (returns, volumes)."""
    return {"volume_model": volume_evaluate(results["volume_model"],
                                            [v for _, v in days], chunk),
            "return_model": return_param_evaluate(results["return_model"], days)}


# ----------------------------------------------------------------------------
# Simulazione: volumi AR (come tim.py) + prezzi TIM parametrico
# ----------------------------------------------------------------------------
def tim_param_simulate(results, n_steps, initial_v, initial_price=100.0,
                       seed=None, bootstrap=False):
    """initial_v: p volumi precedenti (dal piu' vecchio al piu' recente).
    Ritorna (prices, sim_v, sim_r), tutti di lunghezza n_steps;
    prices[0] = initial_price, prices[k+1] = prices[k] * exp(sim_r[k])."""
    vm, rm = results["volume_model"], results["return_model"]
    p = results["p"]
    initial_v = np.asarray(initial_v, dtype=float)
    seed_v, seed_r = np.random.SeedSequence(seed).spawn(2)

    sim_v = volume_simulate(vm, n_steps, initial_v, seed=seed_v, bootstrap=bootstrap)

    K = return_kernel(rm["beta"], rm["kernel_L"])
    tv = power_transform(np.concatenate([initial_v, sim_v]), rm["delta"])
    X = _conv(tv, K)[p:p + n_steps]               # X_t per i passi simulati

    rng = np.random.default_rng(seed_r)
    if bootstrap:
        shocks = rm["resid"][rng.integers(0, len(rm["resid"]), size=n_steps)]
    else:
        shocks = rng.normal(0.0, np.sqrt(rm["sigma2"]), n_steps)
    sim_r = rm["const"] + rm["sigma_f"] * X + shocks

    prices = initial_price * np.exp(np.concatenate([[0.0], np.cumsum(sim_r[:-1])]))
    return prices, sim_v, sim_r


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot_tim_param_fit(results, out_dir=None, residuals=True, resid_step=10):
    """Parametri di best fit e residui.
    Pannelli: coefficienti AR dei volumi | kernel dei rendimenti sigma_f*K(l)
              AR cumulati                | impatto cumulato sigma_f*G(l) (propagatore)."""
    p, delta, beta, L = results["p"], results["delta"], results["beta"], results["kernel_L"]
    vm, rm = results["volume_model"], results["return_model"]

    a = np.asarray(vm["params"])[1:]
    kern = rm["sigma_f"] * return_kernel(beta, L)
    lags_v, lags_r = np.arange(1, p + 1), np.arange(0, L + 1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _draw(axes[0, 0], lags_v, a,
          f"Volume AR({p}) - in-sample R² = {vm['rsquared']:.3f} (adj {vm['rsquared_adj']:.3f})",
          ylabel="coefficient")
    _draw(axes[0, 1], lags_r, kern,
          f"TIM kernel sigma_f·K(l) - in-sample R² = {rm['rsquared']:.3f} "
          f"(adj {rm['rsquared_adj']:.3f})", ylabel="coefficient")
    _draw(axes[1, 0], lags_v, np.cumsum(a),
          "Cumulative AR coefficients on volumes", ylabel="cumulative sum of coefficients")
    _draw(axes[1, 1], lags_r, np.cumsum(kern),
          "Cumulative impact sigma_f·G(l)", ylabel="cumulative sum of coefficients")
    fig.suptitle(f"Fit: p = {p}, beta = {beta:.3f}, delta = {delta}, L = {L}, "
                 f"c = {rm['const']:.2e}, sigma_f = {rm['sigma_f']:.2e}", fontsize=12)
    plt.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        fig.savefig(out_dir / "tim_param_fit.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()

    if residuals:
        plot_residuals([np.asarray(vm["resid"]), np.asarray(rm["resid"])],
                       save_path=None if out_dir is None else out_dir / "residuals.png",
                       step=resid_step)


# ----------------------------------------------------------------------------
# Save / load (.npz, nessun pickle)
# ----------------------------------------------------------------------------
def save_tim_param_model(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vm, rm = results["volume_model"], results["return_model"]
    meta = {"p": int(results["p"]), "delta": float(results["delta"]),
            "beta": float(results["beta"]), "kernel_L": int(results["kernel_L"]),
            "n_par": int(rm["n_par"]), "const": float(rm["const"]),
            "sigma_f": float(rm["sigma_f"]),
            "volume_sigma2": float(vm["sigma2"]),
            "volume_rsquared": float(vm["rsquared"]),
            "volume_rsquared_adj": float(vm["rsquared_adj"]),
            "return_sigma2": float(rm["sigma2"]),
            "return_rsquared": float(rm["rsquared"]),
            "return_rsquared_adj": float(rm["rsquared_adj"])}
    np.savez_compressed(path, meta=json.dumps(meta),
                        volume_params=np.asarray(vm["params"]),
                        volume_resid=np.asarray(vm["resid"]),
                        return_resid=np.asarray(rm["resid"]))


def load_tim_param_model(path):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        vp, vr, rr = f["volume_params"], f["volume_resid"], f["return_resid"]
    return {
        "p": meta["p"], "delta": meta["delta"], "beta": meta["beta"],
        "kernel_L": meta["kernel_L"],
        "volume_model": {"p": meta["p"], "params": vp, "sigma2": meta["volume_sigma2"],
                         "resid": vr, "rsquared": meta["volume_rsquared"],
                         "rsquared_adj": meta["volume_rsquared_adj"]},
        "return_model": {"p": meta["p"], "delta": meta["delta"], "beta": meta["beta"],
                         "kernel_L": meta["kernel_L"], "n_par": meta["n_par"],
                         "const": meta["const"], "sigma_f": meta["sigma_f"],
                         "params": np.array([meta["const"], meta["sigma_f"]]),
                         "sigma2": meta["return_sigma2"], "resid": rr,
                         "rsquared": meta["return_rsquared"],
                         "rsquared_adj": meta["return_rsquared_adj"]},
    }


if __name__ == '__main__':
    paths = np.array(sorted(listdir('..\\database\\data')))   # sorted -> ordine cronologico

    p = 1000
    beta = 0.25                   # esponente del propagatore G(l) = (l+1)^(-beta)
    beta_grid = None              # es. np.linspace(0.05, 0.6, 12) per stimare beta
    delta = 1.0
    kernel_L = 500                # lunghezza del propagatore (idealmente <= p)
    fitting_window = 100          # giorni usati per il fit
    bootstrap_residuals = True
    trim_top = True
    top_frac = 0.05

    name = f'tim_param_{delta}_{p}'

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
        r = np.diff(np.log(prices))                    # log-rendimenti, come nel TIM
        v = volumes[:-1] * signs[:-1]                  # il trade t impatta r_t
        return prices, r, v

    fit_paths = paths[:fitting_window]
    test_paths = paths[fitting_window:]

    # ---- Fit unico sui primi fitting_window giorni ----
    if model_path.exists():
        print('Load Model')
        results = load_tim_param_model(model_path)
    else:
        print('Compute Model')
        days = []
        for path in fit_paths:
            _, r, v = load_day(path)
            days.append((r, v))
        results = tim_param_fit(days, p=p, beta=beta, delta=delta, kernel_L=kernel_L,
                                beta_grid=beta_grid)
        save_tim_param_model(results, model_path)

    # ---- Parametri di best fit e residui ----
    plot_tim_param_fit(results, out_dir=model_dir)

    print(f"In-sample R2 volumi:     {results['volume_model']['rsquared']:.4f} "
          f"(adj {results['volume_model']['rsquared_adj']:.4f})")
    print(f"In-sample R2 rendimenti: {results['return_model']['rsquared']:.4f} "
          f"(adj {results['return_model']['rsquared_adj']:.4f})")

    # ---- Out-of-sample sui giorni restanti ----
    if len(test_paths) > 0:
        test_days = []
        for path in test_paths:
            _, r, v = load_day(path)
            test_days.append((r, v))
        oos = tim_param_evaluate(results, test_days)
        save_oos_stats(results, oos, model_dir,
                       window_info={"fitting_window": fitting_window,
                                    "n_fit_days": len(fit_paths),
                                    "n_test_days": len(test_paths),
                                    "p": p, "beta": results["beta"], "delta": delta,
                                    "kernel_L": kernel_L,
                                    "trim_top": trim_top, "top_frac": top_frac})
    else:
        print('No days left for out-of-sample evaluation')

    clear_data(name)

    # ---- Simulazione di ogni giorno con il modello comune ----
    for path in paths:
        print(path)
        prices, r, v = load_day(path)

        initial_v = v[:p]
        initial_price = prices[p]

        prices_sim, volumes_sim, r_sim = tim_param_simulate(
            results,
            n_steps=len(r),
            initial_v=initial_v,
            initial_price=initial_price,
            bootstrap=bootstrap_residuals
        )

        save_simulated_data(f"..\\{dir}\\" + path, prices_sim, volumes_sim)