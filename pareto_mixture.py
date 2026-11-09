from pathlib import Path
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls


def pareto_mixture(alpha: float, n_types: int, kmax: int = 2000):
    """Replica la costruzione della miscela in methods_s.py."""
    if alpha <= 0:
        raise ValueError("alpha deve essere positivo.")
    if n_types < 1:
        raise ValueError("n_types deve essere almeno 1.")
    if kmax < 2:
        raise ValueError("kmax deve essere almeno 2.")

    if int(n_types) == 1:
        # Come nel codice originale: il caso a una componente richiede alpha > 1.
        if alpha <= 1.0:
            raise ValueError("n_types=1 richiede alpha > 1.")
        return np.array([1.0]), np.array([1.0 / float(np.sum(
            1.0 / np.arange(1, 1000000, dtype=float) ** alpha
        ))])

    J = int(n_types)
    k_all = np.arange(1, kmax + 1, dtype=float)

    # Gli stessi 150 punti log-spaziati e arrotondati del codice originale.
    ks = np.unique(np.round(np.geomspace(1, kmax, 150)).astype(int)).astype(float)
    target = ks ** (-alpha)

    # Griglia geometrica dei tassi.
    rates = np.geomspace(1.0 - 2.0 ** (-alpha), 1.0 / kmax, J)

    # Matrice del fit relativo: ogni riga è divisa per il target Pareto.
    A = (1.0 - rates)[None, :] ** (ks[:, None] - 1.0) / target[:, None]
    b = np.ones(len(ks))

    # Media troncata della Pareto: sum_{k=1}^{kmax} P(L >= k).
    pareto_mean_truncated = np.sum(k_all ** (-alpha))

    # Media troncata di ciascuna geometrica, normalizzata per la media Pareto.
    geometric_means = np.array([
        np.sum((1.0 - r) ** (k_all - 1.0)) for r in rates
    ])
    Mn = geometric_means / pareto_mean_truncated

    # Vincoli quasi duri, con peso 1e3, esattamente come nel codice.
    A_aug = np.vstack([
        A,
        1e3 * Mn[None, :],
        1e3 * np.ones((1, J))
    ])
    b_aug = np.concatenate([b, [1e3], [1e3]])

    weights, residual_norm = nnls(A_aug, b_aug)
    keep = weights > 1e-8
    weights = weights[keep] / weights[keep].sum()
    rates = rates[keep]

    return weights, rates


def survival_curves(alpha, weights, rates, kmax):
    k = np.arange(1, kmax + 1, dtype=float)
    target = k ** (-alpha)
    mixture = (
        weights[None, :] *
        (1.0 - rates)[None, :] ** (k[:, None] - 1.0)
    ).sum(axis=1)
    return k, target, mixture


def save_weights_csv(outdir, weights, rates):
    path = outdir / "mixture_components.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "weight", "termination_rate", "geometric_mean"])
        for j, (w, r) in enumerate(zip(weights, rates), start=1):
            writer.writerow([j, f"{w:.12g}", f"{r:.12g}", f"{1/r:.12g}"])
    return path


def make_plots(alpha, weights, rates, k, target, mixture, outdir):
    # 1. Survival target e miscela in scala log-log.
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.loglog(k, target, label=rf"Pareto target: $k^{{-{alpha:g}}}$", linewidth=2.2)
    ax.loglog(k, mixture, "--", label="Mixture of geometric distributions", linewidth=2.2)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$P(L \geq k)$")
    ax.set_title("Discrete Pareto survival approximation")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("images\\pareto_mixture\\survival_loglog.png", dpi=180)
    plt.close(fig)

    # 2. Errore logaritmico: positivo = sovrastima, negativo = sottostima.
    log_error = np.log(mixture / target)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.semilogx(k, log_error, linewidth=1.8)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\log(S_{\rm mix}(k)/S_{\rm Pareto}(k))$")
    ax.set_title("Logarithmic survival approximation error")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig("images\\pareto_mixture\\log_survival_error.png", dpi=180)
    plt.close(fig)

    # 3. Errore relativo assoluto.
    rel_error = np.abs(mixture / target - 1.0)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.loglog(k, np.maximum(rel_error, 1e-16), linewidth=1.8)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Absolute relative error")
    ax.set_title("Absolute relative error")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig("images\\pareto_mixture\\relative_error.png", dpi=180)
    plt.close(fig)

    # 4. Pesi e tassi delle componenti effettivamente mantenute.
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(1, len(weights) + 1)
    bars = ax.bar(x, weights)
    ax.set_xlabel("Geometric component")
    ax.set_ylabel("Mixture weight")
    ax.set_title("Estimated mixture weights")
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"r={rate:.3g}", ha="center", va="bottom", fontsize=8,
                rotation=0)
    fig.tight_layout()
    fig.savefig("images\\pareto_mixture\\mixture_weights.png", dpi=180)
    plt.close(fig)


def main():
    alpha = 1.5
    n_types = 6
    kmax = 1000

    outdir = 'images'

    weights, rates = pareto_mixture(alpha, n_types, kmax)
    k, target, mixture = survival_curves(
        alpha, weights, rates, kmax
    )

    # Metriche analoghe a quelle stampate da _pareto_mixture.
    pareto_mean = target.sum()
    mixture_mean = mixture.sum()
    log_error = np.abs(np.log(mixture / target))

    ks_fit = np.unique(np.round(
        np.geomspace(1, kmax, 150)
    ).astype(int))
    fit_mask = np.isin(k.astype(int), ks_fit)
    fit_log_error = np.abs(np.log(mixture[fit_mask] / target[fit_mask]))

    print("\n=== Pareto / geometric-mixture approximation ===")
    print(f"alpha                         : {alpha:g}")
    print(f"requested components          : {n_types}")
    print(f"retained components           : {len(weights)}")
    print(f"kmax                          : {kmax}")
    print(f"sum(weights)                   : {weights.sum():.10f}")
    print(f"truncated Pareto mean          : {pareto_mean:.6f}")
    print(f"truncated mixture mean         : {mixture_mean:.6f}")
    print(f"max |log error|, k <= 100      : {log_error[k <= min(100, kmax)].max():.6f}")
    print(f"max |log error|, full range    : {log_error.max():.6f}")
    print(f"max |log error|, fit grid      : {fit_log_error.max():.6f}")

    print("\nComponents:")
    print(" component       weight       rate r       geometric mean 1/r")
    for j, (w, r) in enumerate(zip(weights, rates), start=1):
        print(f" {j:9d}   {w:12.7f}   {r:12.7g}   {1/r:16.6g}")

    make_plots(alpha, weights, rates, k, target, mixture, outdir)


if __name__ == "__main__":
    main()
