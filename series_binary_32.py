"""Verifica numerica dell'eq. (32): distribuzione pooled dei metaordini, N=2, p1 != p2.

Catena di verifica (dal piu' elementare al risultato finale):
  1. quote di mercato empiriche p_k
  2. rho_eff,k empirica (sotto-sequenza del trader k) vs formula p rho1 / (1-(1-p) rho1)
  3. pesi w_k = quota di run del trader k, empirici vs teorici
  4. PMF pooled empirica vs P_tot(n)  (+ chi-quadro)
  5. pendenza della coda vs lambda_tail = -ln(p_eff del trader piu' grande)

Usa i moduli methods.py e series_binary.py (stessa pipeline delle verifiche precedenti).
Serve rho1 > 0 e quote diverse (kind='power' con N=2: p2/p1 = 2^-alpha).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare

import methods
from series_binary import generate_correlated_binary_sequence

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14
})

# ---------------- parametri ----------------
T = 10_000_000          # numero di trade
rho1 = 0.5             # autocorrelazione lag-1 (deve essere > 0)
kind, expo = 'power', 2.0   # N=2 Zipf alpha=1 -> p1=2/3, p2=1/3
N = 2
NMAX = 30              # range di n per il confronto
MIN_EXP = 5            # conteggio atteso minimo per il chi-quadro


# ---------------- teoria ----------------
def rho_eff(p, r1):
    return p * r1 / (1.0 - (1.0 - p) * r1)


def p_eff(p, r1):
    return (1.0 + rho_eff(p, r1)) / 2.0


def weights(p, r1):
    pe = p_eff(p, r1)
    w = p * (1 - pe)
    return w / w.sum()


def p_tot(n, p, r1):
    """Eq. (32) generalizzata a N componenti."""
    p = np.asarray(p, float)
    pe = p_eff(p, r1)
    n = np.asarray(n)[:, None]
    num = (p * (1 - pe) ** 2 * pe ** (n - 1)).sum(axis=1)
    den = (p * (1 - pe)).sum()
    return num / den


# ---------------- simulazione ----------------
signs = generate_correlated_binary_sequence(T, rho=rho1)
traders = np.array(methods.mapping_function(pd.DataFrame(index=range(T)), N, kind, expo))
labels = sorted(set(traders))
p_emp = np.array([(traders == l).mean() for l in labels])
print("1. quote empiriche p_k:", p_emp)

# run: stessa logica di process_one_file, ma tenendo l'etichetta del trader
db = pd.DataFrame({'sign': signs, 'trader': traders})
db = db.sort_values('trader', kind='stable').reset_index(drop=True)
db['metaid'] = ((db['trader'] != db['trader'].shift()) |
                (db['sign'] != db['sign'].shift())).cumsum()
runs = db.groupby('metaid').agg(n=('sign', 'size'), trader=('trader', 'first'))
n_runs = len(runs)

# ---------------- 2. rho_eff ----------------
print("\n2. rho_eff,k   (empirica | teoria)")
for l, pk in zip(labels, p_emp):
    s = signs[traders == l].astype(float)
    emp = np.mean(s[:-1] * s[1:])
    print(f"   {l}: {emp:.4f} | {rho_eff(pk, rho1):.4f}")

# ---------------- 3. pesi dei run ----------------
w_emp = np.array([(runs['trader'] == l).mean() for l in labels])
w_th = weights(p_emp, rho1)
print("\n3. w_k  (empirico | teorico)")
for l, a, b in zip(labels, w_emp, w_th):
    print(f"   {l}: {a:.4f} | {b:.4f}")
print("   (peso 'naive' = p_k:", p_emp, ")")

# ---------------- 4. PMF pooled ----------------
n_vals = np.arange(1, NMAX + 1)
counts = np.array([(runs['n'] == n).sum() for n in n_vals], float)
pmf_emp = counts / n_runs
pmf_th = p_tot(n_vals, p_emp, rho1)

# chi-quadro: n=1..K con atteso >= MIN_EXP, piu' un bin di coda
exp_counts = pmf_th * n_runs
K = int(np.max(np.where(exp_counts >= MIN_EXP)[0])) + 1
obs = np.append(counts[:K], n_runs - counts[:K].sum())
exp = np.append(exp_counts[:K], n_runs - exp_counts[:K].sum())
chi2, pval = chisquare(obs, f_exp=exp)
print(f"\n4. chi-quadro su n=1..{K}+coda: chi2={chi2:.1f}, dof={len(obs)-1}, p-value={pval:.3f}")
print(f"   max errore relativo su n<={K}: "
      f"{np.max(np.abs(pmf_emp[:K] / pmf_th[:K] - 1)):.3%}")

# confronto con la singola geometrica (che deve FALLIRE se p1 != p2)
pe_big = p_eff(p_emp.max(), rho1)
single = (1 - pe_big) * pe_big ** (n_vals - 1)
print("   (confronto: max err. rel. di una singola geometrica col p_eff del trader maggiore: "
      f"{np.max(np.abs(pmf_emp[:K] / single[:K] - 1)):.1%})")

# ---------------- 5. coda ----------------
lam_tail = -np.log(p_eff(p_emp.max(), rho1))
n0 = max(5, K // 2)
sl = np.polyfit(n_vals[n0 - 1:K], np.log(pmf_emp[n0 - 1:K]), 1)[0]
print(f"\n5. pendenza coda (n={n0}..{K}): {-sl:.4f} | lambda_tail teorico = {lam_tail:.4f}")

# ---------------- grafico ----------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogy(n_vals, pmf_emp, 'o', label='Empirical')
ax.semilogy(n_vals, pmf_th, '-', lw=2, label='Theory')
pe = p_eff(p_emp, rho1)
for k in range(N):
    comp = w_th[k] * (1 - pe[k]) * pe[k] ** (n_vals - 1)
    ax.semilogy(n_vals, comp, '--', lw=1, label=f'{labels[k]}')
ax.semilogy(n_vals, single, ':', color='k', label='Singolar geometrical (bigger trader)')
ax.set_xlabel('n')
ax.set_ylabel('P(n)')
ax.grid(True, which='both', alpha=0.3)
ax.legend()
plt.tight_layout()
#plt.show()
plt.savefig('images\\meta_child_dist\\series_markov_1_different.png', dpi=300)