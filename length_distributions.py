"""
length_distributions.py
=======================
Confronta le distribuzioni delle lunghezze dei metaordini per diverse coppie (model, conf).

  model : suffisso della cartella dei meta (database\\meta_{model}); '' per i dati reali (database\\meta)
  conf  : nome del CSV senza estensione (es. 'meta_s_mcmc_K300_1.5_L5')

Per ogni coppia disegna, in scala log-log:
  1) P(L >= k)  (distribuzione cumulata complementare, per i dati discreti e' la forma piu' pulita)
  2) pmf in bin logaritmici
e una retta di riferimento Pareto k^-alpha. Si puo' scegliere la colonna: NbChild (n. di figli)
oppure MetaDurationTrades (durata in trade).

Uso: modificare PAIRS e lanciare  python length_distributions.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIGURAZIONE: una distribuzione per ogni coppia (model, conf)
# ----------------------------------------------------------------------------
PAIRS = [
    # (model,               conf)
    ('lmf_emp_tim_param_0.3', 'meta_lmf'),
    #('lmf_emp_tim_0.3', 'meta_s_3_hmmP6_1.5'),
    ('lmf_emp_tim_param_0.3', 'meta_s_mcmc_K300_1.5'),
    ('',                'meta_s_mcmc_K300_1.5_L50'),
    #('',                'meta_s_3_hmmP6_1.5'),
]
LABELS = None                  # None = "model | conf"; oppure lista di etichette della stessa lunghezza di PAIRS
COLUMN = 'NbChild'             # 'NbChild' | 'MetaDurationTrades'
MIN_VALUE = 2                  # considera solo L >= MIN_VALUE
PARETO_REF = [1.5]             # esponenti delle rette di riferimento P(L>=k) ~ k^-alpha (lista vuota = nessuna)
N_BINS = 25                    # bin logaritmici per la pmf
BASE_DIR = 'database'
OUT_DIR = os.path.join('images', 'length_dist')


def meta_path(model, conf):
    d = os.path.join(BASE_DIR, 'meta' + (f'_{model}' if model else ''))
    return os.path.join(d, conf if conf.endswith('.csv') else conf + '.csv')


def load_lengths(path):
    x = pd.read_csv(path, usecols=[COLUMN])[COLUMN].to_numpy()
    x = x[np.isfinite(x)]
    return x[x >= MIN_VALUE].astype(float)


def ccdf(x):
    """P(L >= k) per i valori distinti k."""
    k, c = np.unique(x, return_counts=True)
    return k, c[::-1].cumsum()[::-1] / len(x)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = LABELS or [f"{m or 'real'} | {c}" for m, c in PAIRS]
    data = []
    for (model, conf), lab in zip(PAIRS, labels):
        p = meta_path(model, conf)
        if not os.path.exists(p):
            print(f"[{lab}] file non trovato: {p} -> saltato")
            continue
        x = load_lengths(p)
        data.append((lab, x))
        q = np.quantile(x, [0.5, 0.9, 0.99])
        print(f"[{lab}] n = {len(x)}  media = {x.mean():.2f}  quantili 50/90/99 = {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}  max = {x.max():.0f}")
    if not data:
        raise SystemExit("nessun file trovato")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for lab, x in data:
        k, c = ccdf(x)
        a1.plot(k, c, marker='.', ms=3, lw=1, label=lab)
        edges = np.unique(np.logspace(np.log10(MIN_VALUE), np.log10(x.max() + 1), N_BINS + 1).astype(int))
        h, e = np.histogram(x, bins=edges)
        w = np.diff(e)
        ok = h > 0
        a2.plot(np.sqrt(e[:-1] * e[1:])[ok], (h / w / len(x))[ok], marker='o', ms=3, lw=1, label=lab)
    kk = np.logspace(np.log10(MIN_VALUE), np.log10(max(x.max() for _, x in data)), 50)
    for a in PARETO_REF:
        a1.plot(kk, (kk / MIN_VALUE) ** -a, 'k--', lw=1, label=f'Pareto {a}')
        a2.plot(kk, (kk / MIN_VALUE) ** -(a + 1) * (a / MIN_VALUE), 'k--', lw=1)
    for ax, yl in ((a1, r'$P(L \geq k)$'), (a2, 'pmf (bin logaritmici)')):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(COLUMN); ax.set_ylabel(yl)
        ax.grid(True, which='both', alpha=0.3)
    a1.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'length_dist_{COLUMN}.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("salvato:", out)


if __name__ == '__main__':
    main()