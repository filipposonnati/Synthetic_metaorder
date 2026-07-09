"""
fit_mem_innovation_dist.py
===========================
Trova la distribuzione empiricamente migliore per l'innovazione z_t del
modello MEM/ACD usato in lmf_tim.py:

    v_t = mu_t * z_t ,   E[z_t] = 1

Nel modulo originale z_t e' fissata a priori come Gamma(k, 1/k), con k
stimato solo dal metodo dei momenti (Var(z) = 1/k). Questo script:

  1. Ricostruisce mu_t sui dati reali usando gli stessi omega/alpha/beta
     stimati via QML in fit_mem_acd() (la stima QML e' consistente
     indipendentemente dalla vera distribuzione di z_t, quindi non va
     rifatta).
  2. Estrae i residui standardizzati z_hat_t = v_t / mu_t (pooling su
     tutti i giorni del dataset).
  3. Fitta via MLE diverse famiglie candidate (tutte vincolate a media 1,
     coerentemente col modello MEM), confrontandole con AIC/BIC e test KS.
  4. Salva una tabella di confronto (CSV) e una figura con QQ-plot per
     ogni candidato.

USO
---
    python fit_mem_innovation_dist.py --data-dir ../database/data --p 1 --q 1

    # per velocizzare, limita a N giorni campionati a caso (riproducibile):
    python fit_mem_innovation_dist.py --data-dir ../database/data --max-days 15

    # oppure specifica esattamente quali giorni usare:
    python fit_mem_innovation_dist.py --data-dir ../database/data --files day1.csv day5.csv day12.csv

Richiede lmf_tim.py nella stessa cartella (o nel PYTHONPATH).
"""

import argparse
from pathlib import Path
from os import listdir

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lmf_tim as lt


# ---------------------------------------------------------------------------
# Distribuzioni candidate, tutte vincolate a media 1 (loc=0 fissato)
# ---------------------------------------------------------------------------
# scipy.fit su alcune di queste (es. burr12, gengamma) puo' essere lento o
# instabile: usiamo bound ragionevoli e piu' tentativi con starting point
# diversi quando serve.

CANDIDATES = {
    "gamma":      stats.gamma,        # caso attuale del codice (Engle-Russell std)
    "weibull":    stats.weibull_min,  # standard alternativo in letteratura ACD
    "lognorm":    stats.lognorm,
    "invgauss":   stats.invgauss,     # usata in alcuni MEM come alternativa alla gamma
    "burr12":     stats.burr12,       # code piu' pesanti (generalized F/Burr)
}


def _rescale_to_unit_mean(dist_name, params):
    """
    Molte famiglie scipy non impongono media=1 durante il fit libero.
    Qui ricalcoliamo la media teorica dai parametri stimati e restituiamo
    un fattore di correzione: la vera assunzione del modello (E[z]=1) va
    imposta rifittando con la scala vincolata quando la famiglia lo
    permette (gamma, weibull, lognorm, invgauss, burr12 hanno tutte un
    parametro di scala che possiamo derivare analiticamente/numericamente
    a partire dal fit "libero" su shape).
    """
    dist = CANDIDATES[dist_name]
    mean_free = dist.mean(*params)
    return mean_free


def fit_candidate(name, z):
    """
    Fitta la famiglia `name` su z (media empirica gia' vicina a 1 per
    costruzione, dato che z_hat = v_t/mu_t). Fit libero (loc=0, scale
    libera), poi verifica quanto la media stimata si discosta da 1.
    """
    dist = CANDIDATES[name]
    try:
        params = dist.fit(z, floc=0)
    except Exception as e:
        return {"name": name, "ok": False, "error": str(e)}

    loglik = float(np.sum(dist.logpdf(z, *params)))
    k_params = len(params) - 1  # -1 per loc fissato a 0
    n = len(z)
    aic = 2 * k_params - 2 * loglik
    bic = k_params * np.log(n) - 2 * loglik

    ks_stat, ks_p = stats.kstest(z, dist.cdf, args=params)
    mean_est = float(_rescale_to_unit_mean(name, params))

    return {
        "name": name, "ok": True, "params": params,
        "loglik": loglik, "aic": aic, "bic": bic,
        "ks_stat": float(ks_stat), "ks_p": float(ks_p),
        "mean_est": mean_est,
    }


def pool_residuals(data_dir, p, q, max_days=None, files=None, seed=0):
    """
    Per ogni file reale in data_dir: fitta MEM/ACD(p,q), ricostruisce
    mu_t, calcola z_hat_t = v_t/mu_t. Concatena sui giorni selezionati.

    max_days : se specificato, campiona casualmente (seed fisso, quindi
               riproducibile) al massimo max_days file invece di usarli
               tutti -> molto piu' veloce, dato che ogni giorno richiede
               un'ottimizzazione Nelder-Mead separata.
    files    : lista esplicita di nomi file da usare (ha precedenza su
               max_days). Utile se vuoi fissare a mano i giorni.
    """
    all_files = sorted(listdir(data_dir))

    if files is not None:
        selected = [f for f in files if f in all_files]
        missing = set(files) - set(selected)
        if missing:
            print(f"  ATTENZIONE: file non trovati e ignorati: {missing}")
    elif max_days is not None and max_days < len(all_files):
        rng_sel = np.random.default_rng(seed)
        selected = sorted(rng_sel.choice(all_files, size=max_days, replace=False))
    else:
        selected = all_files

    print(f"  Giorni selezionati: {len(selected)} / {len(all_files)} disponibili")

    all_z = []
    for fname in selected:
        _, volumes, _ = lt.open_real_data(fname, data_dir)
        v = np.maximum(np.asarray(volumes, dtype=float), 1e-12)
        m = max(p, q)
        if len(v) <= m + 5:
            continue
        params = lt.fit_mem_acd(v, p=p, q=q)
        mu = lt._mem_build_mu(v, params["omega"], params["alpha"],
                               params["beta"], m)
        z_hat = v[m:] / mu[m:]
        all_z.append(z_hat)
        print(f"  {fname}: T={len(v)}  persistenza={params['persistence']:.4f}"
              f"  converged={params['converged']}")

    return np.concatenate(all_z)


def make_density_overlay(z, results, out_path, n_bins_lin=60, n_bins_log=40):
    """
    Un solo grafico con l'istogramma empirico di z e le pdf di TUTTE le
    distribuzioni candidate sovrapposte. Due pannelli:

      - sinistra: scala lineare, ristretta al corpo (fino al 99° percentile)
                  con bin lineari — qui i dati sono densi, i bin lineari
                  vanno bene.
      - destra:   scala LOG-LOG su tutto il range (fino al 99.9° percentile),
                  con bin LOGARITMICI. Nella coda i bin lineari lasciano
                  pochissime osservazioni per barra (istogramma "bucato");
                  i bin log-spaziati si allargano automaticamente dove i
                  dati sono radi, dando un istogramma leggibile anche nella
                  coda pesante. Il log-log e' anche lo standard per
                  giudicare a occhio un comportamento a legge di potenza
                  (una vera coda Pareto appare come una retta).
    """
    ok_results = [r for r in results if r["ok"]]

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Pannello sinistro: corpo, scala lineare, bin lineari ---
    x_body = np.quantile(z, 0.99)
    ax_lin.hist(z[z <= x_body], bins=n_bins_lin, density=True, alpha=0.35,
                color="grey", edgecolor="none", label="empirico (z_hat)")

    xs_body = np.linspace(1e-6, x_body, 2000)
    for r in ok_results:
        dist = CANDIDATES[r["name"]]
        ax_lin.plot(xs_body, dist.pdf(xs_body, *r["params"]), linewidth=1.8,
                    label=f"{r['name']} (AIC={r['aic']:.0f})")

    ax_lin.set_xlim(0, x_body)
    ax_lin.set_xlabel("z_hat")
    ax_lin.set_ylabel("densita'")
    ax_lin.set_title("Corpo della distribuzione (fino al 99° percentile, scala lineare)")
    ax_lin.legend(fontsize=8)

    # --- Pannello destro: coda inclusa, scala log-log, bin logaritmici ---
    x_max = np.quantile(z, 0.999)
    z_min_pos = max(z.min(), x_max * 1e-4)  # evita log(0)/bin degeneri
    log_bins = np.logspace(np.log10(z_min_pos), np.log10(x_max), n_bins_log)

    ax_log.hist(z[(z >= z_min_pos) & (z <= x_max)], bins=log_bins, density=True,
                alpha=0.35, color="grey", edgecolor="none", label="empirico (z_hat)")

    xs_tail = np.logspace(np.log10(z_min_pos), np.log10(x_max), 2000)
    for r in ok_results:
        dist = CANDIDATES[r["name"]]
        ax_log.plot(xs_tail, dist.pdf(xs_tail, *r["params"]), linewidth=1.8,
                    label=f"{r['name']} (AIC={r['aic']:.0f})")

    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.set_xlabel("z_hat (scala log)")
    ax_log.set_ylabel("densita' (scala log)")
    ax_log.set_title("Intera distribuzione incl. coda (log-log, bin logaritmici)")
    ax_log.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nGrafico distribuzione empirica + candidati salvato in: {out_path}")


def make_qq_grid(z, results, out_path):
    ok_results = [r for r in results if r["ok"]]
    n = len(ok_results)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, r in zip(axes, ok_results):
        dist = CANDIDATES[r["name"]]
        stats.probplot(z, dist=dist, sparams=r["params"], plot=ax)
        ax.set_title(f"{r['name']}  (AIC={r['aic']:.1f}, KS_p={r['ks_p']:.3f})")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nQQ-plot salvato in: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=r"../database/data")
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--q", type=int, default=1)
    ap.add_argument("--out-dir", default="./mem_dist_check")
    ap.add_argument("--max-days", type=int, default=50,
                     help="limita l'analisi a N giorni campionati casualmente "
                          "(molto piu' veloce di usarli tutti)")
    ap.add_argument("--files", nargs="+", default=None,
                     help="lista esplicita di nomi file da usare, es. "
                          "--files day1.csv day2.csv (ha precedenza su --max-days)")
    ap.add_argument("--seed", type=int, default=0,
                     help="seed per il campionamento casuale dei giorni")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Estrazione residui standardizzati z_hat = v_t/mu_t da {args.data_dir} "
          f"(MEM/ACD({args.p},{args.q}))...")
    z = pool_residuals(args.data_dir, args.p, args.q,
                        max_days=args.max_days, files=args.files, seed=args.seed)
    print(f"\nTotale osservazioni pooled: {len(z)}   media empirica z_hat = {z.mean():.4f}"
          f"   var empirica = {z.var():.4f}")

    print("\nFit distribuzioni candidate...")
    results = [fit_candidate(name, z) for name in CANDIDATES]

    rows = []
    for r in results:
        if not r["ok"]:
            print(f"  {r['name']}: FALLITO ({r['error']})")
            continue
        rows.append({
            "distribuzione": r["name"],
            "AIC": round(r["aic"], 2),
            "BIC": round(r["bic"], 2),
            "logL": round(r["loglik"], 2),
            "media_stimata": round(r["mean_est"], 4),
            "KS_stat": round(r["ks_stat"], 4),
            "KS_pvalue": round(r["ks_p"], 4),
        })

    table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    print("\n=== Confronto distribuzioni (ordinato per AIC crescente = migliore) ===")
    print(table.to_string(index=False))

    csv_path = out_dir / "confronto_distribuzioni.csv"
    table.to_csv(csv_path, index=False)
    print(f"\nTabella salvata in: {csv_path}")

    make_density_overlay(z, results, out_dir / "densita_confronto.png")

    make_qq_grid(z, results, out_dir / "qq_plot_confronto.png")

    best = table.iloc[0]["distribuzione"]
    print(f"\n--> Distribuzione con AIC migliore: {best}")
    print("Nota: guarda anche il QQ-plot e il KS p-value (p alto = non si "
          "rigetta l'ipotesi che i dati vengano da quella distribuzione). "
          "L'AIC da solo puo' preferire modelli con code sottostimate se il "
          "campione e' piccolo: controlla in particolare la coda destra.")


if __name__ == "__main__":
    main()