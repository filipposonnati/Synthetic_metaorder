import argparse
from pathlib import Path
from os import listdir

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

import lmf_tim as lt

# Distribuzioni candidate (loc=0 fissato)
CANDIDATES = {
    "gamma":      stats.gamma,
    "weibull":    stats.weibull_min,
    "lognorm":    stats.lognorm,
    "invgauss":   stats.invgauss,
    "burr12":     stats.burr12,
}

def fit_candidate(name, z):
    """Fitta la distribuzione e calcola la log-verosimiglianza e l'AIC."""
    dist = CANDIDATES[name]
    try:
        params = dist.fit(z, floc=0)
        loglik = float(np.sum(dist.logpdf(z, *params)))
        k_params = len(params) - 1
        aic = 2 * k_params - 2 * loglik
        return {"name": name, "ok": True, "params": params, "aic": aic}
    except Exception as e:
        return {"name": name, "ok": False}

def pool_residuals(data_dir, p, q, max_days=50, seed=0):
    """Estrae e concatena i residui standardizzati z_hat = v_t / mu_t."""
    all_files = sorted(listdir(data_dir))
    if max_days and max_days < len(all_files):
        rng_sel = np.random.default_rng(seed)
        selected = sorted(rng_sel.choice(all_files, size=max_days, replace=False))
    else:
        selected = all_files

    all_z = []
    for fname in selected:
        _, volumes, _ = lt.open_real_data(fname, data_dir)
        v = np.maximum(np.asarray(volumes, dtype=float), 1e-12)
        m = max(p, q)
        if len(v) <= m + 5:
            continue
        params = lt.fit_mem_acd(v, p=p, q=q)
        mu = lt._mem_build_mu(v, params["omega"], params["alpha"], params["beta"], m)
        all_z.append(v[m:] / mu[m:])

    return np.concatenate(all_z)

def make_density_overlay(z, results, out_path, n_bins_lin=60, n_bins_log=40):
    """Grafico comparativo corpo (lineare) vs coda pesante (log-log)."""
    ok_results = [r for r in results if r["ok"]]
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Pannello sinistro: Corpo della distribuzione
    x_body = np.quantile(z, 0.99)
    ax_lin.hist(z[z <= x_body], bins=n_bins_lin, density=True, alpha=0.35, color="grey", label="Empirico")
    xs_body = np.linspace(1e-6, x_body, 2000)
    for r in ok_results:
        ax_lin.plot(xs_body, CANDIDATES[r["name"]].pdf(xs_body, *r["params"]), linewidth=1.8, label=f"{r['name']} (AIC:{r['aic']:.0f})")
    ax_lin.set_xlim(0, x_body)
    ax_lin.set_title("Corpo della Distribuzione (Scala Lineare)")
    ax_lin.legend(fontsize=8)

    # Pannello destro: Coda (Log-Log)
    x_max = np.quantile(z, 0.999)
    z_min_pos = max(z.min(), x_max * 1e-4)
    log_bins = np.logspace(np.log10(z_min_pos), np.log10(x_max), n_bins_log)
    ax_log.hist(z[(z >= z_min_pos) & (z <= x_max)], bins=log_bins, density=True, alpha=0.35, color="grey", label="Empirico")
    xs_tail = np.logspace(np.log10(z_min_pos), np.log10(x_max), 2000)
    for r in ok_results:
        ax_log.plot(xs_tail, CANDIDATES[r["name"]].pdf(xs_tail, *r["params"]), linewidth=1.8, label=f"{r['name']}")
    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.set_title("Intera Distribuzione e Coda (Log-Log)")
    ax_log.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Grafico densità salvato in: {out_path}")

def make_acf_diagnostic(z, out_path, nlags=40):
    """Nuovo grafico proposto: Analisi di indipendenza dei residui (I.I.D. check)."""
    acf_z = acf(z, nlags=nlags, fft=True)
    acf_z2 = acf((z - z.mean())**2, nlags=nlags, fft=True)
    lags = np.arange(nlags + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    conf = 1.96 / np.sqrt(len(z)) # Intervallo di confidenza al 95%

    # ACF residui standardizzati
    ax1.vlines(lags[1:], 0, acf_z[1:], colors="royalblue", linewidth=2)
    ax1.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax1.axhline(conf, color="red", linestyle="--", alpha=0.6, label="95% Conf.")
    ax1.axhline(-conf, color="red", linestyle="--", alpha=0.6)
    ax1.set_title("ACF dei Residui Standardizzati $z_t$")
    ax1.set_xlabel("Lag")
    ax1.legend()

    # ACF residui al quadrato
    ax2.vlines(lags[1:], 0, acf_z2[1:], colors="crimson", linewidth=2)
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax2.axhline(conf, color="red", linestyle="--", alpha=0.6)
    ax2.set_title("ACF dei Residui al Quadrato $(z_t - \\bar{z})^2$")
    ax2.set_xlabel("Lag")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Grafico ACF diagnostico salvato in: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../database/data")
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--q", type=int, default=1)
    ap.add_argument("--max-days", type=int, default=100)
    args = ap.parse_args()

    # Spostata la cartella di salvataggio dentro ./images
    out_dir = Path("../images/volume_distrib")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Estrazione residui...")
    z = pool_residuals(args.data_dir, args.p, args.q, max_days=args.max_days)
    
    print("Fit distribuzioni...")
    results = [fit_candidate(name, z) for name in CANDIDATES]

    # Generazione dei grafici utili
    make_density_overlay(z, results, out_dir / "densita_confronto.png")
    make_acf_diagnostic(z, out_dir / "residui_acf_diagnostica.png")

if __name__ == "__main__":
    main()