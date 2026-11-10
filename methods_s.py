import re
from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import nnls
from scipy.special import zeta, gammaln

"""
METHODS_S
=========
Ricostruzione dei metaordini a partire dalla sequenza reale dei segni, campionando la
partizione a posteriori da un modello LMF (forward filtering + backward sampling esatti, con
stato nascosto sparso). Poi ogni trade e' assegnato a un metaordine/trader scelto
uniformemente tra quelli compatibili con il segno osservato.

MODELLI
  mode='hmm'         LMF a N trader fissi, lunghezza dei metaordini geometrica con tasso di
                     conclusione r = 1/zeta(pareto_exp) (stessa media della Pareto).
  mode='hmm_pareto'  come 'hmm' ma lunghezza Pareto(pareto_exp) approssimata da una miscela di
                     n_types geometriche (tipo nascosto per metaordine, hazard decrescente).
  mode='hmm_lambda'  numero di metaordini attivi VARIABILE come nel modello lambda di
                     Lillo-Mike-Farmer: a ogni passo un nuovo metaordine arriva con prob. lam
                     (segno +-1 a caso), un metaordine attivo a caso viene eseguito e termina
                     con prob. r = 1/zeta(pareto_exp). Stato nascosto: (n_plus, n_minus) con
                     n_plus + n_minus <= max_pool. Lunghezze geometriche.
  mode='mcmc'        ricostruzione per "linking": ogni trade sceglie un predecessore dello stesso
                     segno (o nessuno) e le catene sono i metaordini. Modello con normalizzazione:
                     al passo t viene eseguito un metaordine aperto (peso lam(eta'), eta' = trade
                     dall'ultimo figlio) o uno nuovo (peso nu[n], n = n. di metaordini aperti);
                     lunghezza Pareto(pareto_exp) FISSATA. lam(eta') e nu[n] sono nonparametrici,
                     appresi dai dati (EM stocastico + MCMC Metropolis-Hastings con mosse di
                     ri-collegamento e scambio di predecessori; accelerato con numba). Non c'e' un
                     N fisso: la densita' di metaordini attivi e' implicita in lam e nu.
                     Con la lunghezza libera scala delle lunghezze e densita' non sono identificabili
                     dai soli segni, quindi pareto_exp va fissato.
  mode='truth'       id veri dei metaordini (5a colonna dei file di lmf_synthetic_market con
                     save_meta_ids=True), come riferimento.

VARIANTI (valide per i tre modelli hmm*)
  eps        frazione di trade di RUMORE (segno casuale, nessun metaordine); questi trade
             diventano metaordini singoli con trader 'Noise' (esclusi da n_filter >= 2).
  griglie    nb_traders (modelli a N fisso), lam (hmm_lambda) ed eps possono essere liste: i
             parametri sono scelti per massima verosimiglianza dei segni osservati.
  block_size se dato, i parametri sono scelti separatamente per ogni blocco di block_size trade
             dentro il giorno (attivita' non stazionaria); la partizione riparte a ogni blocco
             (i metaordini non attraversano i confini dei blocchi). None = un blocco per giorno.

drop_prob in [0, 1]: probabilita' di scartare un metaordine con MetaImpact < 0 (0 = nessuno
scarto). E' una selezione sul risultato: usare con cautela.

Le proprieta' dei metaordini sono calcolate come in methods.generate; in piu':
  MetaDurationTrades = n. di trade di mercato (di qualunque trader) nella finestra;
  PreImbalance       = s * sbilanciamento del flusso di mercato (pesato per volume) negli
                       `pre_window` trade precedenti al primo figlio.
"""


def _as_list(x):
    return [float(v) if isinstance(v, (float, np.floating)) else v for v in x] \
        if isinstance(x, (list, tuple, np.ndarray)) else [x]


# ----------------------------------------------------------------------------
# Miscela di geometriche per la Pareto
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _pareto_mixture(alpha, n_types, kmax=2000):
    """Approssima P(L >= k) = k^(-alpha) con una miscela di geometriche:
         P(L >= k) ~ sum_j w_j (1 - r_j)^(k-1),   k = 1..kmax
    n_types = 1: una sola geometrica con r = 1/zeta(alpha) (stessa media; richiede alpha > 1).
    Altrimenti: tassi su griglia geometrica tra r_max = 1 - 2^-alpha (hazard al primo figlio) e
    1/kmax; pesi non negativi (NNLS) che minimizzano l'errore relativo su P(L >= k) per k
    log-spaziati (peso uniforme in log k, cosi' nessuna coda estrema domina il fit), con due
    vincoli quasi duri: sum(w) = 1 e media troncata E[min(L, kmax)] uguale a quella della Pareto.
    Il vincolo sulla media e' essenziale: senza, la media risulta sottostimata e i metaordini
    ricostruiti troppo corti."""
    if int(n_types) == 1:
        if alpha <= 1.0:
            raise ValueError("n_types=1 / 'hmm' / 'hmm_lambda' richiedono pareto_exp > 1")
        return np.array([1.0]), np.array([1.0 / float(zeta(alpha))])
    J = int(n_types)
    k_all = np.arange(1, kmax + 1, dtype=float)
    ks = np.unique(np.round(np.geomspace(1, kmax, 150)).astype(int)).astype(float)
    target = ks ** (-alpha)
    rates = np.geomspace(1.0 - 2.0 ** (-alpha), 1.0 / kmax, J)
    A = (1.0 - rates)[None, :] ** (ks[:, None] - 1.0) / target[:, None]
    b = np.ones(len(ks))
    mean_t = (k_all ** (-alpha)).sum()                           # E[min(L, kmax)] della Pareto
    Mn = np.array([((1.0 - r) ** (k_all - 1.0)).sum() for r in rates]) / mean_t
    A = np.vstack([A, 1e3 * Mn[None, :], 1e3 * np.ones((1, J))])
    b = np.concatenate([b, [1e3], [1e3]])
    w, _ = nnls(A, b)
    keep = w > 1e-8
    w, rates = w[keep] / w[keep].sum(), rates[keep]
    S = (w[None, :] * (1.0 - rates)[None, :] ** (k_all[:, None] - 1.0)).sum(axis=1)
    err = np.abs(np.log(S / k_all ** (-alpha)))
    print(f"[hmm] miscela per Pareto({alpha}): {len(w)} componenti, "
          f"r = {np.array2string(rates, precision=4)}, w = {np.array2string(w, precision=4)}; "
          f"media {S.sum():.3f} (Pareto {mean_t:.3f}), errore max su log P(L>=k): "
          f"{err[:100].max():.2f} (k<=100), {err.max():.2f} (k<={kmax})")
    return w, rates


# ----------------------------------------------------------------------------
# Modello a N trader fissi (stato = conteggi dei trader per (segno, tipo))
# Tipo tau = j per segno +, tau = J + j per segno -.
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _build_fixed_model(N, alpha, n_types, eps):
    w, r = _pareto_mixture(alpha, n_types)
    J = len(w)
    TT = 2 * J
    combos = list(combinations_with_replacement(range(TT), N))
    S = len(combos)
    counts = np.zeros((S, TT), dtype=np.int64)
    for idx, c in enumerate(combos):
        for tau in c:
            counts[idx, tau] += 1
    index = {tuple(counts[i]): i for i in range(S)}

    MT, Mc, ev, mask = {}, {}, {}, {}
    for s in (1, -1):
        rows, cols, vals = [], [], []
        events = {}
        for x in range(S):
            c = counts[x]
            if eps > 0:                                  # trade di rumore: stato invariato
                rows.append(x); cols.append(x); vals.append(eps / 2.0)
                events.setdefault((x, x), []).append((-1, False, -1, eps / 2.0))
            for j in range(J):
                tau = j if s > 0 else J + j
                if c[tau] == 0:
                    continue
                base = (1.0 - eps) * c[tau] / N          # trader di tipo tau scelto e segno s
                wt = base * (1.0 - r[j])                 # nessun rinnovo
                rows.append(x); cols.append(x); vals.append(wt)
                events.setdefault((x, x), []).append((tau, False, tau, wt))
                for tn in range(TT):                     # rinnovo: segno 1/2, componente w
                    wt = base * r[j] * 0.5 * w[tn % J]
                    y_c = c.copy()
                    y_c[tau] -= 1
                    y_c[tn] += 1
                    y = index[tuple(y_c)]
                    rows.append(x); cols.append(y); vals.append(wt)
                    events.setdefault((x, y), []).append((tau, True, tn, wt))
        Ms = sparse.coo_matrix((vals, (rows, cols)), shape=(S, S)).tocsr()
        sl = slice(0, J) if s > 0 else slice(J, TT)
        MT[s], Mc[s] = Ms.T.tocsr(), Ms.tocsc()
        ev[s] = events
        mask[s] = ((counts[:, sl].sum(axis=1) > 0) | (eps > 0)).astype(float)

    # prior sul primo trade: segno +-1 con prob. 1/2, componente ~ w_j / r_j (un trader passa
    # piu' tempo nei metaordini lunghi)
    nu = (w / r) / (w / r).sum()
    p_tau = np.concatenate([0.5 * nu, 0.5 * nu])
    logp = gammaln(N + 1) - gammaln(counts + 1).sum(axis=1) + (counts * np.log(p_tau)[None, :]).sum(axis=1)
    prior = np.exp(logp)
    prior /= prior.sum()

    return dict(kind='fixed', N=N, J=J, TT=TT, S=S, counts=counts, MT=MT, Mc=Mc, ev=ev,
                mask=mask, prior=prior)


# ----------------------------------------------------------------------------
# Modello lambda (pool variabile): stato (n_plus, n_minus)
# Evento = (arrivo in {0,+1,-1}, esito in {0: rumore, 1: eseguito e continua, 2: eseguito e termina})
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _build_lambda_model(lam, alpha, eps, max_pool):
    _, r_arr = _pareto_mixture(alpha, 1)
    r = float(r_arr[0])
    M = int(max_pool)
    states, index = [], {}
    for tot in range(M + 1):
        for a in range(tot + 1):
            index[(a, tot - a)] = len(states)
            states.append((a, tot - a))
    S = len(states)

    MT, Mc, ev, mask = {}, {}, {}, {}
    P_tot = sparse.csr_matrix((S, S))
    for s in (1, -1):
        rows, cols, vals = [], [], []
        events = {}

        def add(x, y, wt, e):
            if wt > 0:
                rows.append(x); cols.append(y); vals.append(wt)
                events.setdefault((x, y), []).append(e + (wt,))

        for x, (a, b) in enumerate(states):
            n = a + b
            q = 1.0 if n == 0 else (lam if n < M else 0.0)       # prob. di arrivo
            opts = []
            if q < 1.0:
                opts.append((0, 1.0 - q))
            if q > 0.0:
                opts.append((1, q / 2.0))
                opts.append((-1, q / 2.0))
            for arr, pa in opts:
                a2, b2 = a + (arr == 1), b + (arr == -1)
                n2 = a2 + b2
                if eps > 0:                                      # trade di rumore (segno s con prob. 1/2)
                    add(x, index[(a2, b2)], pa * eps / 2.0, (arr, 0))
                ns = a2 if s > 0 else b2
                if ns > 0:
                    pe = pa * (1.0 - eps) * ns / n2              # esecuzione di un metaordine di segno s
                    add(x, index[(a2, b2)], pe * (1.0 - r), (arr, 1))
                    a3, b3 = (a2 - 1, b2) if s > 0 else (a2, b2 - 1)
                    add(x, index[(a3, b3)], pe * r, (arr, 2))
        Ms = sparse.coo_matrix((vals, (rows, cols)), shape=(S, S)).tocsr()
        MT[s], Mc[s] = Ms.T.tocsr(), Ms.tocsc()
        ev[s] = events
        mask[s] = np.ones(S)
        P_tot = P_tot + Ms

    # prior: distribuzione stazionaria del numero di metaordini attivi (iterazione di potenza)
    v = np.zeros(S)
    v[0] = 1.0
    PT = P_tot.T.tocsr()
    for _ in range(3000):
        v = PT @ v
        v /= v.sum()
    prior = v

    return dict(kind='lambda', states=states, S=S, MT=MT, Mc=Mc, ev=ev, mask=mask, prior=prior)


# ----------------------------------------------------------------------------
# Filtro e campionamento (comuni ai modelli)
# ----------------------------------------------------------------------------
def _forward(mdl, obs_idx, keep=True):
    """Forward filtering. F[t] = P(x_t | obs[0..t-1]) (normalizzato); ll = log P(obs)."""
    S, T = mdl['S'], len(obs_idx)
    F = np.empty((T + 1, S), dtype=np.float32) if keep else None
    cur = mdl['prior'].copy()
    ll = 0.0
    if keep:
        F[0] = cur
    for t in range(T):
        key = 1 if obs_idx[t] == 0 else -1
        v = mdl['MT'][key] @ cur
        tot = v.sum()
        if tot <= 0.0:                                   # osservazione impossibile: penalita' e ripiego
            v = mdl['mask'][key] * (cur + 1e-300)
            tot = v.sum() if v.sum() > 0 else 1.0
            ll += -700.0
        else:
            ll += np.log(tot)
        cur = v / tot
        if keep:
            F[t + 1] = cur
    return F, ll


def _backward(mdl, F, obs_idx, rng):
    """Backward sampling della traiettoria dello stato nascosto x_0..x_T."""
    T = len(obs_idx)
    S = mdl['S']
    path = np.empty(T + 1, dtype=np.int64)
    pT = F[T].astype(np.float64)
    path[T] = rng.choice(S, p=pT / pT.sum())
    u = rng.random(T)
    for t in range(T - 1, -1, -1):
        m = mdl['Mc'][1 if obs_idx[t] == 0 else -1]
        y = path[t + 1]
        lo, hi = m.indptr[y], m.indptr[y + 1]
        idx = m.indices[lo:hi]
        wgt = F[t][idx].astype(np.float64) * m.data[lo:hi]
        tot = wgt.sum()
        if tot <= 0.0:
            path[t] = idx[int(np.argmax(m.data[lo:hi]))]
        else:
            k = int(np.searchsorted(np.cumsum(wgt), u[t] * tot))
            path[t] = idx[min(k, len(idx) - 1)]
    return path


def _pick_event(lst, u):
    if len(lst) == 1:
        return lst[0]
    w = np.fromiter((e[-1] for e in lst), dtype=float, count=len(lst))
    j = min(int(np.searchsorted(np.cumsum(w), u * w.sum())), len(lst) - 1)
    return lst[j]


# ----------------------------------------------------------------------------
# Assegnazione dei trade ai metaordini lungo la traiettoria campionata
# ----------------------------------------------------------------------------
def _assign_fixed_block(mdl, obs_idx, path, rng, next_id):
    N, TT, counts, ev = mdl['N'], mdl['TT'], mdl['counts'], mdl['ev']
    T = len(obs_idx)
    tr = np.full(T, -1, dtype=np.int64)
    mid_out = np.empty(T, dtype=np.int64)
    noise = np.zeros(T, dtype=bool)

    types = np.repeat(np.arange(TT), counts[path[0]])
    mid = np.arange(next_id, next_id + N)
    next_id += N
    pools = [[] for _ in range(TT)]
    for i, tau in enumerate(types):
        pools[int(tau)].append(i)

    u_pick = rng.random(T)
    u_ev = rng.random(T)
    for k in range(T):
        key = 1 if obs_idx[k] == 0 else -1
        tau_p, renew, tau_n, _ = _pick_event(ev[key][(int(path[k]), int(path[k + 1]))], u_ev[k])
        if tau_p < 0:                                    # trade di rumore: metaordine singolo
            noise[k] = True
            mid_out[k] = next_id
            next_id += 1
            continue
        pool = pools[tau_p]
        i = pool[int(u_pick[k] * len(pool))]
        tr[k] = i
        mid_out[k] = mid[i]
        if renew:
            mid[i] = next_id
            next_id += 1
            if tau_n != tau_p:
                pool.remove(i)
                pools[tau_n].append(i)
    return tr, mid_out, noise, next_id


def _assign_lambda_block(mdl, obs_idx, path, rng, next_id):
    states, ev = mdl['states'], mdl['ev']
    T = len(obs_idx)
    tr = np.full(T, -1, dtype=np.int64)
    mid_out = np.empty(T, dtype=np.int64)
    noise = np.zeros(T, dtype=bool)

    a, b = states[path[0]]
    pools = {1: [], -1: []}
    for _ in range(a):
        pools[1].append(next_id)
        next_id += 1
    for _ in range(b):
        pools[-1].append(next_id)
        next_id += 1

    u_pick = rng.random(T)
    u_ev = rng.random(T)
    for k in range(T):
        s = 1 if obs_idx[k] == 0 else -1
        arr, kind, _ = _pick_event(ev[s][(int(path[k]), int(path[k + 1]))], u_ev[k])
        if arr != 0:                                     # arrivo di un nuovo metaordine
            pools[arr].append(next_id)
            next_id += 1
        if kind == 0:                                    # trade di rumore
            noise[k] = True
            mid_out[k] = next_id
            next_id += 1
            continue
        pool = pools[s]
        j = int(u_pick[k] * len(pool))
        mid_out[k] = pool[j]
        if kind == 2:                                    # il metaordine termina
            pool[j] = pool[-1]
            pool.pop()
    return tr, mid_out, noise, next_id


def _assign_hmm(signs, day_codes, mode, alpha, n_types, N_list, eps_list, lam_list,
                block_size, max_pool):
    """Partizione campionata a posteriori, con scelta per massima verosimiglianza dei parametri
    (griglia) per ogni blocco. Ritorna (trader, metaid, noise) per ogni trade."""
    lam_kind = (mode == 'hmm_lambda')
    nt = 1 if mode in ('hmm', 'hmm_lambda') else int(n_types)
    if lam_kind:
        grid = [(float(l), float(e)) for l in lam_list for e in eps_list]
        build = lambda p: _build_lambda_model(p[0], float(alpha), p[1], int(max_pool))
    else:
        grid = [(int(N), float(e)) for N in N_list for e in eps_list]
        build = lambda p: _build_fixed_model(p[0], float(alpha), nt, p[1])
    rng = np.random.default_rng()

    n_tr = len(signs)
    trader_out = np.full(n_tr, -1, dtype=np.int64)
    meta_out = np.empty(n_tr, dtype=np.int64)
    noise_out = np.zeros(n_tr, dtype=bool)
    if n_tr == 0:
        return trader_out, meta_out, noise_out

    d_starts = np.flatnonzero(np.r_[True, day_codes[1:] != day_codes[:-1]])
    d_ends = np.r_[d_starts[1:], n_tr]
    chosen = Counter()
    next_id = 0

    for a, b in zip(d_starts, d_ends):
        if block_size:
            edges = list(range(a, b, int(block_size)))
            if len(edges) > 1 and b - edges[-1] < int(block_size) // 4:
                edges.pop()                              # accorpa l'ultimo blocco se troppo corto
        else:
            edges = [a]
        edges = edges + [b]

        for lo, hi in zip(edges[:-1], edges[1:]):
            obs_idx = np.where(signs[lo:hi] > 0, 0, 1)
            if len(grid) > 1:
                lls = [_forward(build(p), obs_idx, keep=False)[1] for p in grid]
                best = int(np.argmax(lls))
            else:
                best = 0
            chosen[grid[best]] += hi - lo
            mdl = build(grid[best])
            F, _ = _forward(mdl, obs_idx, keep=True)
            path = _backward(mdl, F, obs_idx, rng)
            if lam_kind:
                tr, mi, nz, next_id = _assign_lambda_block(mdl, obs_idx, path, rng, next_id)
            else:
                tr, mi, nz, next_id = _assign_fixed_block(mdl, obs_idx, path, rng, next_id)
            trader_out[lo:hi], meta_out[lo:hi], noise_out[lo:hi] = tr, mi, nz

    if len(grid) > 1 or block_size:
        names = ('lam', 'eps') if lam_kind else ('N', 'eps')
        tot = sum(chosen.values())
        summ = ', '.join(f"{names[0]}={p[0]:g} eps={p[1]:g}: {100 * c / tot:.0f}%"
                         for p, c in sorted(chosen.items()))
        print(f"[{mode}] parametri scelti (quota di trade): {summ}")
    return trader_out, meta_out, noise_out


try:
    from numba import njit
except ImportError:                                   # fallback: pura Python (molto lento)
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

LOG2 = 0.6931471805599453


# ============================================================================
# MCMC sulle catene di trade ("linking")
# ----------------------------------------------------------------------------
# Stato: pred[i] = predecessore del trade i (stesso segno, a distanza <= K) oppure -1
# (il trade apre un nuovo metaordine); ogni trade ha al piu' un successore.
# Le catene sono i metaordini.
#
# Modello (proprio, con normalizzazione): al passo t viene eseguito o un metaordine
# aperto (peso lam(eta'), eta' = t - ultimo figlio) o un metaordine nuovo (peso nu[n],
# n = n. di metaordini aperti); il segno osservato e' quello del metaordine eseguito
# (un nuovo metaordine ha segno +-1 con prob. 1/2). Un metaordine e' "aperto" tra un
# figlio e il successivo. Lunghezza L con pmf f(L). Joint di segni e partizione:
#
#   P = prod_t [ w_t / Z_t ] * prod_catene f(L),   Z_t = nu[n_t] + sum_{j aperti} lam(t-j)
#   w_t = lam(t - pred[t])   se pred[t] >= 0,   nu[n_t] / 2   altrimenti
# ============================================================================


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


@njit(cache=True)
def _link(a, b, sgn, cnt, S, lam):
    """Aggiunge (sgn=+1) o toglie (sgn=-1) il metaordine aperto tra a e b: t in (a, b]."""
    for t in range(a + 1, b + 1):
        cnt[t] += sgn
        S[t] += sgn * lam[t - a]


@njit(cache=True)
def _term(t, pred, cnt, S, loglam, nu, lognu, NMAX):
    c = cnt[t]
    if c > NMAX:
        c = NMAX
    z = nu[c] + S[t]
    if z < 1e-300:
        z = 1e-300
    if pred[t] >= 0:
        num = loglam[t - pred[t]]
    else:
        num = lognu[c] - LOG2
    return num - np.log(z)


@njit(cache=True)
def _chains_logf(x0, x1, x2, x3, pred, succ, logf, LMAX):
    hs = np.empty(4, np.int64)
    nh = 0
    total = 0.0
    for q in range(4):
        if q == 0:
            x = x0
        elif q == 1:
            x = x1
        elif q == 2:
            x = x2
        else:
            x = x3
        if x < 0:
            continue
        h = x
        while pred[h] >= 0:
            h = pred[h]
        dup = False
        for k in range(nh):
            if hs[k] == h:
                dup = True
        if dup:
            continue
        hs[nh] = h
        nh += 1
        n = 1
        y = h
        while succ[y] >= 0:
            y = succ[y]
            n += 1
        if n > LMAX:
            n = LMAX
        total += logf[n]
    return total


@njit(cache=True)
def _init_state(pred, K, lam):
    T = len(pred)
    succ = np.full(T, -1, np.int64)
    cnt = np.zeros(T, np.int64)
    S = np.zeros(T)
    for i in range(T):
        p = pred[i]
        if p >= 0:
            succ[p] = i
            _link(p, i, 1, cnt, S, lam)
    return succ, cnt, S


@njit(cache=True)
def _total_loglik(pred, succ, cnt, S, loglam, nu, lognu, NMAX, logf, LMAX):
    T = len(pred)
    tot = 0.0
    for t in range(T):
        tot += _term(t, pred, cnt, S, loglam, nu, lognu, NMAX)
        if pred[t] < 0:
            n = 1
            y = t
            while succ[y] >= 0:
                y = succ[y]
                n += 1
            if n > LMAX:
                n = LMAX
            tot += logf[n]
    return tot


@njit(cache=True)
def _sign_index(sign, K):
    """Indici per proporre candidati dello stesso segno: sp = posizioni dei trade + seguite da
    quelle dei trade -; gidx[i] = posizione di i in sp; mwin[i] = n. di trade dello stesso segno
    in [i-K, i-1]."""
    T = len(sign)
    npl = 0
    for t in range(T):
        if sign[t] > 0:
            npl += 1
    sp = np.empty(T, np.int64)
    gidx = np.empty(T, np.int64)
    mwin = np.zeros(T, np.int64)
    a = 0
    b = npl
    for t in range(T):
        if sign[t] > 0:
            sp[a] = t
            gidx[t] = a
            a += 1
        else:
            sp[b] = t
            gidx[t] = b
            b += 1
    for t in range(T):
        g = gidx[t]
        lo = g
        base = 0 if sign[t] > 0 else npl
        while lo - 1 >= base and sp[lo - 1] >= t - K:
            lo -= 1
        mwin[t] = g - lo
    return sp, gidx, mwin


@njit(cache=True)
def _logq_rank(r, m, rho, p_new):
    """log prob. di proporre il r-esimo trade dello stesso segno precedente (r = 1..m),
    geometrica troncata con parametro rho; con prob. p_new si propone invece -1."""
    return np.log(1.0 - p_new) + np.log(rho) + (r - 1) * np.log(1.0 - rho) - np.log(1.0 - (1.0 - rho) ** m)


@njit(cache=True)
def _sweep(sign, pred, succ, cnt, S, K, lam, loglam, nu, lognu, NMAX, logf, LMAX, p_new, n_moves,
           sp, gidx, mwin, rho, beta):
    """n_moves proposte Metropolis-Hastings di ri-collegamento (temperatura inversa beta).
    Ritorna (accettate, delta log P)."""
    T = len(sign)
    acc = 0
    dsum = 0.0
    if T < 2:
        return acc, dsum
    lpn = np.log(p_new)
    for _m in range(n_moves):
        i = np.random.randint(1, T)
        m = mwin[i]
        if m == 0:
            continue
        if np.random.random() < p_new:
            jn = -1
            logq_new = lpn
            r_new = 0
        else:
            # r ~ geometrica troncata su 1..m (inversione)
            u = np.random.random()
            r_new = 1 + int(np.floor(np.log(1.0 - u * (1.0 - (1.0 - rho) ** m)) / np.log(1.0 - rho)))
            if r_new > m:
                r_new = m
            if r_new < 1:
                r_new = 1
            jn = sp[gidx[i] - r_new]
            logq_new = _logq_rank(r_new, m, rho, p_new)
        p = pred[i]
        if jn == p:
            continue
        if p < 0:
            logq_old = lpn
        else:
            r_old = gidx[i] - gidx[p]
            logq_old = _logq_rank(r_old, m, rho, p_new)
        i2 = -1
        if jn >= 0 and succ[jn] >= 0:                 # j' occupato: scambio di predecessori
            if p < 0:
                continue
            i2 = succ[jn]
            if p >= i2 or i2 - p > K:
                continue
        # intervallo di t toccato
        lo = i + 1
        if p >= 0 and p + 1 < lo:
            lo = p + 1
        if jn >= 0 and jn + 1 < lo:
            lo = jn + 1
        hi = i
        if i2 > hi:
            hi = i2
        old_sum = 0.0
        for t in range(lo, hi + 1):
            old_sum += _term(t, pred, cnt, S, loglam, nu, lognu, NMAX)
        old_chain = _chains_logf(i, p, jn, i2, pred, succ, logf, LMAX)

        # applica
        if i2 < 0:
            if p >= 0:
                _link(p, i, -1, cnt, S, lam)
                succ[p] = -1
            if jn >= 0:
                _link(jn, i, 1, cnt, S, lam)
                succ[jn] = i
            pred[i] = jn
        else:
            _link(p, i, -1, cnt, S, lam)
            _link(jn, i2, -1, cnt, S, lam)
            _link(jn, i, 1, cnt, S, lam)
            _link(p, i2, 1, cnt, S, lam)
            pred[i] = jn
            succ[jn] = i
            pred[i2] = p
            succ[p] = i2

        new_sum = 0.0
        for t in range(lo, hi + 1):
            new_sum += _term(t, pred, cnt, S, loglam, nu, lognu, NMAX)
        new_chain = _chains_logf(i, p, jn, i2, pred, succ, logf, LMAX)
        dlp = (new_sum - old_sum) + (new_chain - old_chain)
        logr = beta * dlp + logq_old - logq_new
        if np.log(np.random.random()) < logr:
            acc += 1
            dsum += dlp
        else:                                         # ripristina
            if i2 < 0:
                if jn >= 0:
                    _link(jn, i, -1, cnt, S, lam)
                    succ[jn] = -1
                if p >= 0:
                    _link(p, i, 1, cnt, S, lam)
                    succ[p] = i
                pred[i] = p
            else:
                _link(jn, i, -1, cnt, S, lam)
                _link(p, i2, -1, cnt, S, lam)
                _link(p, i, 1, cnt, S, lam)
                _link(jn, i2, 1, cnt, S, lam)
                pred[i] = p
                succ[p] = i
                pred[i2] = jn
                succ[jn] = i2
    return acc, dsum


@njit(cache=True)
def _stats(pred, succ, cnt, S, K, age_bin, nu, NMAX, LMAX, len_bin,
           lam_num, lam_den, nu_num, nu_den, len_cnt):
    """Statistiche sufficienti per l'aggiornamento MM dei parametri."""
    T = len(pred)
    for t in range(T):
        c = cnt[t]
        if c > NMAX:
            c = NMAX
        z = nu[c] + S[t]
        if z < 1e-300:
            z = 1e-300
        nu_den[c] += 1.0 / z
        if pred[t] < 0:
            nu_num[c] += 1.0
        else:
            lam_num[age_bin[t - pred[t]]] += 1.0
    for j in range(T):
        i = succ[j]
        if i < 0:
            continue
        for t in range(j + 1, i + 1):
            c = cnt[t]
            if c > NMAX:
                c = NMAX
            z = nu[c] + S[t]
            if z < 1e-300:
                z = 1e-300
            lam_den[age_bin[t - j]] += 1.0 / z
    for h in range(T):
        if pred[h] < 0:
            n = 1
            y = h
            while succ[y] >= 0:
                y = succ[y]
                n += 1
            if n > LMAX:
                n = LMAX
            len_cnt[len_bin[n]] += 1.0


@njit(cache=True)
def _greedy_init(sign, K, q):
    """Partizione iniziale: ogni trade si collega, con prob. q, al piu' vicino trade precedente
    dello stesso segno ancora senza successore (entro K)."""
    T = len(sign)
    pred = np.full(T, -1, np.int64)
    has_succ = np.zeros(T, np.bool_)
    for t in range(1, T):
        if np.random.random() < q:
            for j in range(t - 1, max(t - K, 0) - 1, -1):
                if sign[j] == sign[t] and not has_succ[j]:
                    pred[t] = j
                    has_succ[j] = True
                    break
    return pred


@njit(cache=True)
def _forward_init(sign, K, lam, nu, NMAX, haz, LMAX):
    """Partizione iniziale campionata in avanti dal modello: il trade t sceglie un predecessore
    tra le code vive dello stesso segno con peso lam(eta'), oppure apre un metaordine nuovo
    (peso nu[n]/2, n = n. di code vive entro K); dopo ogni figlio il metaordine si conclude
    con probabilita' haz[lunghezza]."""
    T = len(sign)
    pred = np.full(T, -1, np.int64)
    succ = np.full(T, -1, np.int64)
    clen = np.ones(T, np.int64)
    alive = np.zeros(T, np.bool_)
    w = np.empty(K + 1)
    for t in range(T):
        s = sign[t]
        tot = 0.0
        n_open = 0
        lo = max(t - K, 0)
        for j in range(t - 1, lo - 1, -1):
            w[t - j] = 0.0
            if alive[j] and succ[j] < 0:
                n_open += 1
                if sign[j] == s:
                    w[t - j] = lam[t - j]
                    tot += lam[t - j]
        c = n_open if n_open < NMAX else NMAX
        wn = 0.5 * nu[c]
        u = np.random.random() * (tot + wn)
        if u >= tot:
            clen[t] = 1
        else:
            acc = 0.0
            ch = -1
            for j in range(t - 1, lo - 1, -1):
                acc += w[t - j]
                if w[t - j] > 0.0 and u < acc:
                    ch = j
                    break
            if ch < 0:
                clen[t] = 1
            else:
                pred[t] = ch
                succ[ch] = t
                clen[t] = clen[ch] + 1
        k = clen[t] if clen[t] < LMAX else LMAX
        alive[t] = np.random.random() >= haz[k]
    return pred


@njit(cache=True)
def _chain_ids(pred):
    T = len(pred)
    cid = np.empty(T, np.int64)
    nxt = 0
    for t in range(T):
        if pred[t] < 0:
            cid[t] = nxt
            nxt += 1
        else:
            cid[t] = cid[pred[t]]
    return cid


# ----------------------------------------------------------------------------
# Parametri: tabelle a bin
# ----------------------------------------------------------------------------
def _make_bins(n_max, n_single, ratio):
    """Bin per interi 1..n_max: singoli fino a n_single, poi geometrici. Ritorna (bin_of[0..n_max], nb, width)."""
    edges = [1]
    a = 1
    while a <= n_max:
        nxt = a + 1 if a < n_single else max(a + 1, int(round(a * ratio)))
        edges.append(nxt)
        a = nxt
    bin_of = np.zeros(n_max + 1, dtype=np.int64)
    width = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], min(edges[k + 1], n_max + 1)
        bin_of[lo:hi] = k
        width.append(hi - lo)
    return bin_of, len(edges) - 1, np.array(width, dtype=float)


class McmcParams:
    """Parametri nonparametrici: lam(eta') a bin, nu[n], pmf f(L) a bin."""

    def __init__(self, K=100, NMAX=12, LMAX=2000, pareto_exp=None):
        self.K, self.NMAX, self.LMAX = int(K), int(NMAX), int(LMAX)
        self.fix_f = pareto_exp is not None
        self.age_bin, self.nb_lam, self.age_w = _make_bins(self.K, 8, 1.25)
        self.len_bin, self.nb_len, self.len_w = _make_bins(self.LMAX, 12, 1.3)
        self.lam_bin = np.ones(self.nb_lam)
        self.nu = np.ones(self.NMAX + 1)
        r = 0.4
        cnt = np.array([sum((1 - r) ** (n - 1) * r for n in range(1, self.LMAX + 1)
                            if self.len_bin[n] == b) for b in range(self.nb_len)])
        cnt = np.maximum(cnt, 1e-12)
        self.f_bin = cnt / self.len_w / cnt.sum()           # pmf per intero (normalizzata)
        if pareto_exp is not None:                          # pmf Pareto: P(L >= n) = n^-alpha
            n = np.arange(1, self.LMAX + 1, dtype=float)
            pm = n ** -pareto_exp - (n + 1.0) ** -pareto_exp
            pm[-1] += (self.LMAX + 1.0) ** -pareto_exp
            fb = np.array([pm[self.len_bin[1:] == b].sum() for b in range(self.nb_len)])
            self.f_bin = np.maximum(fb, 1e-300) / self.len_w / fb.sum()

    def tables(self):
        lam = np.ones(self.K + 1)
        lam[1:] = self.lam_bin[self.age_bin[1:]]
        logf = np.zeros(self.LMAX + 1)
        logf[1:] = np.log(np.maximum(self.f_bin[self.len_bin[1:]], 1e-12))
        return lam, np.log(lam), self.nu.copy(), np.log(self.nu), logf

    def hazard(self):
        """h(k) = f(k) / P(L >= k) dalla pmf corrente, k = 0..LMAX."""
        f = np.zeros(self.LMAX + 1)
        f[1:] = self.f_bin[self.len_bin[1:]]
        surv = 1.0 - np.concatenate([[0.0], np.cumsum(f[1:])[:-1]])
        h = np.zeros(self.LMAX + 1)
        h[1:] = np.clip(f[1:] / np.maximum(surv, 1e-12), 0.0, 1.0)
        h[self.LMAX] = 1.0
        return h

    @staticmethod
    def _shrink(num, den, old, strength=0.5, reach=2):
        """Stima MM regolarizzata: il rapporto num/den dei bin poco esposti (den piccolo) e' molto
        rumoroso, quindi viene tirato verso la media locale (bin entro `reach`, pesati per den)."""
        nb = len(num)
        ok = den > 1e-9
        raw = np.where(ok, num / np.maximum(den, 1e-300), old)
        pr = strength * (np.median(den[ok]) if ok.any() else 1.0)
        loc = np.empty(nb)
        for b in range(nb):
            lo, hi = max(b - reach, 0), min(b + reach + 1, nb)
            w = den[lo:hi] * np.exp(-0.7 * np.abs(np.arange(lo, hi) - b))
            loc[b] = (w * raw[lo:hi]).sum() / w.sum() if w.sum() > 0 else old[b]
        return (num + pr * loc) / (den + pr)

    def mstep(self, lam_num, lam_den, nu_num, nu_den, len_cnt, prior=0.2):
        self.lam_bin = np.maximum(self._shrink(lam_num, lam_den, self.lam_bin, reach=2), 1e-9)
        self.nu = np.maximum(self._shrink(nu_num, nu_den, self.nu, reach=1), 1e-9)
        # scala libera (solo i rapporti contano): normalizza sulla media di lam pesata per l'esposizione
        w = np.maximum(lam_den, 0.0)
        sc = (w * self.lam_bin).sum() / w.sum() if w.sum() > 0 else self.lam_bin.mean()
        self.lam_bin = self.lam_bin / sc
        self.nu = self.nu / sc
        if not self.fix_f:
            c = len_cnt + 0.5
            self.f_bin = c / c.sum() / self.len_w


def run_mcmc(signs, day_codes, K=100, n_em=20, n_sweeps=4, n_stat=2, final_sweeps=6,
             p_new=0.15, rho=0.6, anneal=1.0, pareto_exp=None, params=None, seed=None, verbose=True, check=False, init_q=0.5, true_pred=None):
    """EM stocastico + campionamento finale della partizione.
    signs: segni (+-1) in ordine temporale; day_codes: indice del giorno di ogni trade.
    Ritorna (chain_id per trade [globale], params)."""
    signs = np.asarray(signs)
    n = len(signs)
    sgn = np.where(signs > 0, 1, -1).astype(np.int8)
    d_starts = np.flatnonzero(np.r_[True, day_codes[1:] != day_codes[:-1]])
    d_ends = np.r_[d_starts[1:], n]
    if seed is not None:
        _seed(int(seed))
    par = params if params is not None else McmcParams(K=K, pareto_exp=pareto_exp)
    K = par.K
    days = []
    lam0, _, nu0, _, _ = par.tables()
    haz0 = par.hazard()
    for a, b in zip(d_starts, d_ends):
        sg = sgn[a:b].copy()
        days.append((a, b, sg, _forward_init(sg, K, lam0, nu0, par.NMAX, haz0, par.LMAX)))
    idx = [_sign_index(sg, K) for (_, _, sg, _) in days]

    def rebuild(tabs):
        lam, loglam, nu, lognu, logf = tabs
        return [_init_state(pred, K, lam) for (_, _, _, pred) in days]

    hist = []
    for it in range(n_em):
        lam, loglam, nu, lognu, logf = tabs = par.tables()
        states = rebuild(tabs)
        lam_num = np.zeros(par.nb_lam); lam_den = np.zeros(par.nb_lam)
        nu_num = np.zeros(par.NMAX + 1); nu_den = np.zeros(par.NMAX + 1)
        len_cnt = np.zeros(par.nb_len)
        accs = []
        for di, ((a, b, sg, pred), (succ, cnt, S)) in enumerate(zip(days, states)):
            T = b - a
            sp, gidx, mwin = idx[di]
            for sw in range(n_sweeps):
                # annealing: temperatura alta nelle prime iterazioni EM, poi 1
                beta = 1.0 / (1.0 + (anneal - 1.0) * max(0.0, 1.0 - it / max(n_em / 2.0, 1.0)))
                ac, _ = _sweep(sg, pred, succ, cnt, S, K, lam, loglam, nu, lognu, par.NMAX, logf, par.LMAX, p_new, T,
                               sp, gidx, mwin, rho, beta)
                accs.append(ac / max(T, 1))
                if sw >= n_sweeps - n_stat:
                    _stats(pred, succ, cnt, S, K, par.age_bin, nu, par.NMAX, par.LMAX, par.len_bin,
                           lam_num, lam_den, nu_num, nu_den, len_cnt)
        if true_pred is not None and verbose and (it % 5 == 0 or it == n_em - 1):
            tp = np.asarray(true_pred, dtype=np.int64)
            ts, tc, tS = _init_state(tp, K, lam)
            ll_true = _total_loglik(tp, ts, tc, tS, loglam, nu, lognu, par.NMAX, logf, par.LMAX)
            (a, b, sg, pred), (succ, cnt, S) = days[0], states[0]
            ll_cur = _total_loglik(pred, succ, cnt, S, loglam, nu, lognu, par.NMAX, logf, par.LMAX)
            print(f"[mcmc-diag] iter {it + 1}: logP(partizione vera) = {ll_true:.1f}, logP(campione) = {ll_cur:.1f}")
        par.mstep(lam_num, lam_den, nu_num, nu_den, len_cnt)
        if it >= n_em - max(n_em // 2, 1):
            hist.append((par.lam_bin.copy(), par.nu.copy()))
        if verbose and (it % 5 == 0 or it == n_em - 1):
            nchains = len_cnt.sum() / n_stat
            print(f"[mcmc] EM {it + 1}/{n_em}: accettazione {np.mean(accs):.3f}, catene {int(nchains)}, "
                  f"L medio {n / max(nchains, 1):.2f}")
        # warm start dello stato tra le iterazioni
        days = [(a, b, sg, pred) for (a, b, sg, pred) in days]

    if hist:                                                # media sulle ultime iterazioni EM
        par.lam_bin = np.mean([h[0] for h in hist], axis=0)
        par.nu = np.mean([h[1] for h in hist], axis=0)
    lam, loglam, nu, lognu, logf = tabs = par.tables()
    states = rebuild(tabs)
    cid_all = np.empty(n, np.int64)
    base = 0
    for di, ((a, b, sg, pred), (succ, cnt, S)) in enumerate(zip(days, states)):
        T = b - a
        tot0 = _total_loglik(pred, succ, cnt, S, loglam, nu, lognu, par.NMAX, logf, par.LMAX) if check else 0.0
        dsum = 0.0
        sp, gidx, mwin = idx[di]
        for sw in range(final_sweeps):
            ac, d = _sweep(sg, pred, succ, cnt, S, K, lam, loglam, nu, lognu, par.NMAX, logf, par.LMAX, p_new, T,
                           sp, gidx, mwin, rho, 1.0)
            dsum += d
        if check:
            tot1 = _total_loglik(pred, succ, cnt, S, loglam, nu, lognu, par.NMAX, logf, par.LMAX)
            print(f"[mcmc-check] log P: {tot0:.3f} + {dsum:.3f} = {tot0 + dsum:.3f} vs ricalcolato {tot1:.3f}")
        cid = _chain_ids(pred)
        cid_all[a:b] = cid + base
        base += int(cid.max()) + 1 if T else 0
    return cid_all, par


_MCMC_CACHE = {}


def _assign_mcmc(signs, day_codes, pareto_exp, K, n_em, n_sweeps, n_stat, n_final, rho, relearn):
    """Partizione per linking MCMC. I parametri (lam, nu) appresi alla prima chiamata sono
    riutilizzati nelle successive (stessi K e pareto_exp) se relearn=False.
    Ritorna l'id di catena (metaordine) di ogni trade."""
    key = (int(K), float(pareto_exp))
    par = _MCMC_CACHE.get(key)
    learn = relearn or par is None
    if par is None:
        par = McmcParams(K=K, pareto_exp=pareto_exp)
    cid, par = run_mcmc(signs, day_codes, K=K, n_em=(n_em if learn else 0), n_sweeps=n_sweeps,
                        n_stat=n_stat, final_sweeps=n_final, rho=rho, pareto_exp=pareto_exp, params=par)
    _MCMC_CACHE[key] = par
    if learn:
        lam_show = np.round(par.lam_bin[:10], 3)
        print(f"[mcmc] parametri appresi: lam(eta'=1..) = {lam_show}, nu[n_aperti] = {np.round(par.nu, 3)}")
    return cid


def learn_mcmc_params(paths, data_dir, pareto_exp, K=100, n_em=30, n_sweeps=40, n_stat=10, rho=0.6):
    """Apprende (lam, nu) UNA volta su piu' giorni insieme e li salva nella cache: le chiamate
    successive a generate(mode='mcmc', mcmc_relearn=False) li riutilizzano senza riapprendere.
    paths: lista di file (relativi a data_dir), tipicamente giorni distribuiti nel periodo."""
    sg_all, dc_all = [], []
    for k, pth in enumerate(paths):
        tr = pd.read_csv(f"{data_dir}\\{pth}", header=None)
        order = np.argsort(np.array(tr[0]), kind='stable')
        sg = np.sign(np.array(tr[3])[order])
        sg_all.append(sg)
        dc_all.append(np.full(len(sg), k, dtype=np.int64))
    sg_all, dc_all = np.concatenate(sg_all), np.concatenate(dc_all)
    print(f"[mcmc] apprendimento su {len(paths)} file, {len(sg_all)} trade")
    key = (int(K), float(pareto_exp))
    par = McmcParams(K=K, pareto_exp=float(pareto_exp))
    _, par = run_mcmc(sg_all, dc_all, K=K, n_em=n_em, n_sweeps=n_sweeps, n_stat=n_stat,
                      final_sweeps=0, rho=rho, pareto_exp=float(pareto_exp), params=par)
    _MCMC_CACHE[key] = par
    print(f"[mcmc] parametri appresi: lam(eta'=1..) = {np.round(par.lam_bin, 3)}, "
          f"nu[n_aperti] = {np.round(par.nu, 3)}")
    return par


# ----------------------------------------------------------------------------
# GENERATE
# ----------------------------------------------------------------------------
def generate(path, nb_traders, pareto_exp, start_id=0, data_dir='database\\data',
             mode='hmm_pareto', drop_prob=0.0, pre_window=1000, n_types=4,
             eps=0.0, lam=0.3, block_size=None, max_pool=30,
             mcmc_K=100, mcmc_em=30, mcmc_sweeps=40, mcmc_stat=10, mcmc_final=200,
             mcmc_rho=0.6, mcmc_relearn=False):
    """
    Genera i metaordini da un file di trade e calcola impatto/volume parziali e totali.

    Parameters
    ----------
    path : str            file CSV (relativo a data_dir)
    nb_traders : int or list   N trader (modelli a N fisso); lista = griglia per verosimiglianza
    pareto_exp : float    esponente della Pareto della lunghezza (n. di figli) dei metaordini
    start_id : int        offset per gli id dei metaordini
    data_dir : str        cartella dei dati
    mode : 'hmm' | 'hmm_pareto' | 'hmm_lambda' | 'mcmc' | 'truth'   (vedi docstring del modulo)
    mcmc_K : int          (mcmc) distanza massima, in trade, tra due figli consecutivi
    mcmc_em, mcmc_sweeps, mcmc_stat : (mcmc) iterazioni EM, sweep per iterazione e sweep finali usati
                          per le statistiche; l'apprendimento avviene alla prima chiamata
    mcmc_final : int      (mcmc) sweep finali di campionamento della partizione
    mcmc_rho : float      (mcmc) parametro della proposta geometrica sul rango dei candidati
    mcmc_relearn : bool   (mcmc) se True riapprende i parametri a ogni chiamata (di default li riusa)
    drop_prob : float in [0, 1]
        probabilita' di scartare un metaordine con MetaImpact < 0 (0 = nessuno scarto)
    pre_window : int      n. di trade di mercato PRIMA del primo figlio per PreImbalance
    n_types : int         (hmm_pareto) n. di componenti della miscela di geometriche
    eps : float or list   frazione di trade di rumore; lista = griglia per verosimiglianza
    lam : float or list   (hmm_lambda) probabilita' di arrivo di un nuovo metaordine per passo
    block_size : int or None   parametri scelti per blocchi di block_size trade (None = per giorno)
    max_pool : int        (hmm_lambda) massimo numero di metaordini attivi nello stato nascosto

    Returns
    -------
    metaorders_agg : pd.DataFrame   statistiche aggregate per metaordine
    sorted_trades : pd.DataFrame    trade con le feature calcolate
    """
    if mode not in ('hmm', 'hmm_pareto', 'hmm_lambda', 'mcmc', 'truth'):
        raise ValueError("mode deve essere 'hmm', 'hmm_pareto', 'hmm_lambda', 'mcmc' o 'truth'")

    trades = pd.read_csv(f"{data_dir}\\{path}", header=None)
    t = np.array(trades[0])

    date_match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', path)
    if date_match:
        year, month, day = date_match.groups()
        date_string = f"{year}-{month}-{day}"
    else:
        date_string = "2020-01-01"

    start_of_day_dt = pd.to_datetime(date_string)
    timestamp = start_of_day_dt + pd.to_timedelta(t, unit='s')

    cols = {
        'timestamp': timestamp,
        'sign': np.array(trades[3]),
        'quantity': np.array(trades[2]),
        'price': np.array(trades[1])
    }
    if mode == 'truth':
        if 4 not in trades.columns:
            raise ValueError(f"mode='truth' richiede la quinta colonna (id metaordine) in {path}")
        cols['truth_id'] = np.array(trades[4]).astype(np.int64)
    trades = pd.DataFrame(cols)

    trades['day'] = trades['timestamp'].dt.date
    trades.sort_values('timestamp', inplace=True, kind='stable')

    daily_stats = trades.groupby('day').agg(
        DailySigma=('price', lambda x: (x.max() - x.min()) / x.iloc[0]),
        DailyVolume=('quantity', 'sum')
    ).reset_index()

    trades = trades.merge(daily_stats, on='day', how='left')
    trades['BeginMid'] = np.log(trades['price']) / trades['DailySigma']
    trades['EndTime'] = trades['timestamp']
    trades['EndMid'] = trades['BeginMid'].shift(-1)

    trades.drop(trades.tail(1).index, inplace=True)

    trades['Volume'] = trades['quantity'] / trades['DailyVolume']
    trades['EndTransactionTime'] = trades['Volume'].cumsum()
    trades['BeginTransactionTime'] = trades['EndTransactionTime'] - trades['Volume']

    trades.drop(columns=['price', 'quantity', 'DailySigma'], inplace=True)
    trades = trades.rename(columns={'timestamp': 'BeginTime'})
    trades = trades.reset_index(drop=True)
    trades['tidx'] = np.arange(len(trades))          # posizione nel flusso di mercato

    # somme cumulate del flusso di mercato (per PreImbalance)
    vol_mkt = trades['Volume'].to_numpy(dtype=float)
    sgn_mkt = np.sign(trades['sign'].to_numpy(dtype=float))
    cs0 = np.concatenate([[0.0], np.cumsum(sgn_mkt * vol_mkt)])
    cv0 = np.concatenate([[0.0], np.cumsum(vol_mkt)])
    dcodes = pd.factorize(trades['day'])[0]
    d_starts = np.flatnonzero(np.r_[True, dcodes[1:] != dcodes[:-1]])
    day_start = np.repeat(d_starts, np.diff(np.r_[d_starts, len(dcodes)]))

    # ---- partizione in metaordini ----
    if mode == 'truth':
        raw = trades['truth_id'].to_numpy().astype(np.int64).copy()
        noise = raw < 0
        if noise.any():
            raw[noise] = raw.max() + 1 + np.arange(noise.sum())
        trades['trader'] = ('Meta ' + pd.Series(raw).astype(str)).to_numpy()
        trades['_mid'] = raw
        drop_cols = ['_mid', 'truth_id']
    else:
        if mode == 'mcmc':
            mid_raw = _assign_mcmc(trades['sign'].to_numpy(), dcodes, float(pareto_exp), mcmc_K, mcmc_em,
                                   mcmc_sweeps, mcmc_stat, mcmc_final, mcmc_rho, mcmc_relearn)
            tr_idx = np.zeros(len(mid_raw), dtype=np.int64)
            noise = np.zeros(len(mid_raw), dtype=bool)
        else:
            tr_idx, mid_raw, noise = _assign_hmm(
                trades['sign'].to_numpy(), dcodes, mode, float(pareto_exp), n_types,
                _as_list(nb_traders), _as_list(eps), _as_list(lam), block_size, max_pool)
        if mode in ('hmm_lambda', 'mcmc'):
            labels = 'Meta ' + pd.Series(mid_raw).astype(str)
        else:
            labels = 'Trader ' + pd.Series(tr_idx + 1).astype(str)
        labels = labels.where(~noise, 'Noise')
        trades['trader'] = labels.to_numpy()
        trades['_mid'] = mid_raw
        drop_cols = ['_mid']

    sorted_trades = trades.sort_values(['_mid', 'BeginTime'], kind='stable').reset_index(drop=True)
    meta_ids = pd.factorize(sorted_trades['_mid'])[0].astype(np.int64)
    sorted_trades.drop(columns=drop_cols, inplace=True)
    sorted_trades['metaid'] = meta_ids + 1 + start_id

    # ---- proprieta' dei metaordini (come in methods.generate) ----
    start_t = sorted_trades.groupby('metaid')['BeginTransactionTime'].transform('first')
    sorted_trades['TradedVolume'] = sorted_trades['EndTransactionTime'] - start_t

    start_prices = sorted_trades.groupby('metaid')['BeginMid'].transform('first')
    sorted_trades['PartialImpact'] = (sorted_trades['EndMid'] - start_prices) * sorted_trades['sign']

    sorted_trades['PartialVolume'] = sorted_trades.groupby('metaid')['Volume'].cumsum()

    metaorders_agg = sorted_trades.groupby('metaid').agg({
        'sign': 'first',
        'BeginMid': 'first',
        'EndMid': 'last',
        'trader': 'first',
        'BeginTime': 'first',
        'EndTime': 'last',
        'Volume': 'sum',
        'TradedVolume': 'last',
        'DailyVolume': 'first',
        'BeginTransactionTime': 'first',
        'EndTransactionTime': 'last'
    }).reset_index()

    # n. di trade di mercato (di tutti i trader) nella finestra del metaordine
    g_t = sorted_trades.groupby('metaid')['tidx']
    metaorders_agg['MetaDurationTrades'] = (g_t.max() - g_t.min() + 1).to_numpy()

    # PreImbalance = s * D_pre: sbilanciamento del flusso di mercato (pesato per volume) nei
    # pre_window trade PRECEDENTI al primo figlio. NaN se il metaordine parte a inizio giorno.
    a0 = g_t.min().to_numpy()
    lo = np.maximum(a0 - int(pre_window), day_start[a0])
    den = cv0[a0] - cv0[lo]
    d_pre = np.where(a0 > lo, (cs0[a0] - cs0[lo]) / np.where(den > 0, den, 1.0), np.nan)
    metaorders_agg['PreImbalance'] = metaorders_agg['sign'].to_numpy() * d_pre

    sorted_trades['TradedVolume'] = sorted_trades['TradedVolume'] / sorted_trades.groupby('metaid')['TradedVolume'].transform('last')

    nb_child = sorted_trades.groupby('metaid').size().reset_index(name='NbChild')
    metaorders_agg = metaorders_agg.merge(nb_child, on='metaid')

    sorted_trades['NbChild'] = sorted_trades.groupby('metaid')['trader'].transform('count')
    sorted_trades['MetaVolume'] = sorted_trades.groupby('metaid')['PartialVolume'].transform('last')

    begintime = sorted_trades.groupby('metaid')['BeginTime'].transform('first')
    endtime = sorted_trades.groupby('metaid')['EndTime'].transform('last')

    sorted_trades['MetaDuration'] = (endtime - begintime).dt.total_seconds()
    sorted_trades['ElapsedTime'] = (sorted_trades['BeginTime'] - begintime).dt.total_seconds()

    sorted_trades['Ratio'] = sorted_trades['PartialImpact'] / np.sqrt(sorted_trades['MetaVolume'])

    for col in ['EndTime', 'day', 'EndTransactionTime']:
        if col in sorted_trades.columns:
            del sorted_trades[col]

    metaorders_agg['MetaImpact'] = (metaorders_agg['EndMid'] - metaorders_agg['BeginMid']) * metaorders_agg['sign']

    metaorders_agg = metaorders_agg.rename(columns={
        'Volume': 'MetaVolume', 'Impact': 'MetaImpact', 'TransactionTime': 'BeginTransactionTime'
    })

    metaorders_agg['Ratio'] = metaorders_agg['MetaImpact'] / np.sqrt(metaorders_agg['MetaVolume'])

    if drop_prob > 0:
        negative = (metaorders_agg['MetaImpact'] < 0).to_numpy()
        discard = negative & (np.random.random(len(metaorders_agg)) < drop_prob)
        keep = pd.Series(~discard, index=metaorders_agg.index)
        keep_ids = metaorders_agg.loc[keep, 'metaid']
        metaorders_agg = metaorders_agg[keep].reset_index(drop=True)
        sorted_trades = sorted_trades[sorted_trades['metaid'].isin(keep_ids)].reset_index(drop=True)

    return metaorders_agg, sorted_trades