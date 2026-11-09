"""
check_preimbalance.py
=====================
Confronta le partizioni in metaordini (vera, lmf, series, controlli) sugli stessi dati.

Per ogni CSV prodotto da meta_creation_s.py calcola, in bin LOGARITMICI di
pi = MetaVolume/TradedVolume (stessi bin per tutti i file, sul range complessivo dei dati):
  - media di PreImbalance (s * sbilanciamento del flusso di mercato prima del primo figlio)
  - media di MetaImpact
  - frazione di metaordini con MetaImpact < 0
  - numero di metaordini
e confronta le distribuzioni di pi, NbChild e MetaDurationTrades con quelle del riferimento (KS).

Uso: modificare FILES e lanciare  python check_preimbalance.py
Output: stampa a video + images/check_preimbalance/{table.csv, check.png}
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# ----------------------------------------------------------------------------
# CONFIGURAZIONE: etichetta -> percorso del CSV (il primo e' il riferimento = partizione vera)
# ----------------------------------------------------------------------------
META_DIR = os.path.join('database', 'meta_lmf_emp_tim_param_0.3')      # cartella dei CSV
FILES = {
    'truth':  os.path.join(META_DIR, 'meta_s_truth.csv'),
    'hmm2':    os.path.join(META_DIR, 'meta_s_2_hmm_1.5.csv'),
    'hmm3':    os.path.join(META_DIR, 'meta_s_3_hmm_1.5.csv'),
    'lmf2':    os.path.join(META_DIR, 'meta_s_2_lmf_W4.0_K3.0_g0.5_1.5.csv'),
    'lmf3':    os.path.join(META_DIR, 'meta_s_3_lmf_W4.0_K3.0_g0.5_1.5.csv'),
    #'lmf4':    os.path.join(META_DIR, 'meta_s_4_lmf_W4.0_K3.0_g0.5_1.5.csv'),
    'series': os.path.join(META_DIR, 'meta_s_3_uniform_1.5.csv'),
    # 'iid_lmf': os.path.join('database', 'meta_iid', 'meta_s_20_lmf_W4.0_K3.0_g0.5_1.5.csv'),   # controllo segni i.i.d.
    # 'real_lmf': os.path.join('database', 'meta', 'meta_s_20_lmf_W4.0_K3.0_g0.5_1.5.csv'),      # dati reali
}
N_BINS = 20                  # bin logaritmici di pi
MIN_NBCHILD = 2
LOW_PI_QUANTILE = 0.3        # "pi piccolo" = sotto questo quantile di pi del riferimento
MIN_BIN_COUNT = 30           # i bin con meno metaordini non vengono mostrati nei grafici
OUT_DIR = os.path.join('images', 'check_preimbalance')
COLS = ['MetaVolume', 'TradedVolume', 'MetaImpact', 'NbChild', 'MetaDurationTrades', 'PreImbalance']


def load(path):
    df = pd.read_csv(path, usecols=COLS)
    df = df[df['NbChild'] >= MIN_NBCHILD].copy()
    df['pi'] = df['MetaVolume'] / df['TradedVolume']
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['pi', 'MetaImpact', 'PreImbalance'])
    return df[df['pi'] > 0]


def sem(x):
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {}
    for name, path in FILES.items():
        if not os.path.exists(path):
            print(f"[{name}] file non trovato: {path} -> saltato")
            continue
        data[name] = load(path)
        print(f"[{name}] {len(data[name])} metaordini (NbChild >= {MIN_NBCHILD}, PreImbalance valido)")
    if len(data) < 2:
        raise SystemExit("servono almeno due file (riferimento + uno da confrontare)")

    ref_name = next(iter(data))
    ref = data[ref_name]

    # bin logaritmici comuni sul range complessivo (esclusi gli estremi 0.01% per outlier)
    pooled = np.concatenate([d['pi'].to_numpy() for d in data.values()])
    lo, hi = np.quantile(pooled, [1e-4, 1 - 1e-4])
    edges = np.logspace(np.log10(lo), np.log10(hi), N_BINS + 1)
    print(f"\nbin logaritmici di pi: {N_BINS} bin tra {lo:.3e} e {hi:.3e}")

    rows = []
    for name, df in data.items():
        df['bin'] = pd.cut(df['pi'], bins=edges, include_lowest=True, labels=False)
        for b, g in df.groupby('bin'):
            rows.append(dict(file=name, bin=int(b), pi_med=g['pi'].median(), n=len(g),
                             pre=g['PreImbalance'].mean(), pre_sem=sem(g['PreImbalance']),
                             impact=g['MetaImpact'].mean(), impact_sem=sem(g['MetaImpact']),
                             frac_neg=(g['MetaImpact'] < 0).mean()))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT_DIR, 'table.csv'), index=False)

    pd.set_option('display.width', 220)
    for metric, title in (('pre', 'PreImbalance medio'), ('impact', 'MetaImpact medio'),
                          ('frac_neg', 'frazione con MetaImpact < 0'), ('n', 'numero di metaordini')):
        print(f"\n=== {title} per bin logaritmico di pi (bin 0 = pi piu' piccolo) ===")
        piv = tab.pivot(index='bin', columns='file', values=metric)
        piv.insert(0, 'pi_med', tab[tab['file'] == ref_name].set_index('bin')['pi_med'])
        print(piv.round(5).to_string())

    # confronto delle distribuzioni con il riferimento
    print(f"\n=== KS rispetto a '{ref_name}' (statistica; 0 = stessa distribuzione) ===")
    rng = np.random.default_rng(0)
    for name, df in data.items():
        if name == ref_name:
            continue
        out = []
        for col in ('pi', 'NbChild', 'MetaDurationTrades'):
            a = np.log10(ref[col].to_numpy()[rng.permutation(len(ref))[:200000]])
            b = np.log10(df[col].to_numpy()[rng.permutation(len(df))[:200000]])
            out.append(f"{col}: {ks_2samp(a, b).statistic:.3f}")
        print(f"  {name:<10} " + " | ".join(out))

    # sintesi per pi piccolo: soglia = quantile del riferimento, uguale per tutti i file
    thr = float(np.quantile(ref['pi'], LOW_PI_QUANTILE))
    print(f"\n=== Sintesi: metaordini con pi <= {thr:.3e} (quantile {LOW_PI_QUANTILE} di '{ref_name}') ===")
    summ = {}
    for name, df in data.items():
        g = df[df['pi'] <= thr]
        summ[name] = {'n': len(g), 'PreImbalance': g['PreImbalance'].mean(),
                      'MetaImpact': g['MetaImpact'].mean(), 'frac_neg': (g['MetaImpact'] < 0).mean()}
    print(pd.DataFrame(summ).T.round(5).to_string())

    # grafici (bin con pochi metaordini esclusi)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name in data:
        t = tab[(tab['file'] == name) & (tab['n'] >= MIN_BIN_COUNT)].sort_values('bin')
        x = t['pi_med'].to_numpy()
        axes[0].errorbar(x, t['pre'], t['pre_sem'], marker='o', ms=4, label=name)
        axes[1].errorbar(x, t['impact'], t['impact_sem'], marker='o', ms=4, label=name)
        axes[2].plot(x, t['frac_neg'], marker='o', ms=4, label=name)
    for ax, yl in zip(axes, ('PreImbalance medio', 'MetaImpact medio', 'frazione I < 0')):
        ax.set_xscale('log')
        ax.set_xlabel(r'$\pi$ (mediana del bin)')
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)
    axes[0].axhline(0, color='k', lw=0.8)
    axes[1].axhline(0, color='k', lw=0.8)
    axes[2].axhline(0.5, color='k', lw=0.8, ls=':')
    axes[0].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'check.png'), dpi=150)
    plt.close(fig)
    print(f"\nSalvati: {OUT_DIR}\\table.csv e {OUT_DIR}\\check.png")


if __name__ == '__main__':
    main()