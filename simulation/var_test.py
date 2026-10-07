"""Verifica di var_fit e simulate_var.

Test 1: recupero dei parametri. Si simula da un modello con parametri NOTI,
        si ristima con var_fit e si confrontano coefficienti e sigma2.
Test 2: coerenza stima -> simulazione. Si simula dal modello stimato, si ristima
        e si confronta con il modello stimato; si confrontano anche le medie.
Test 3: struttura ricorsiva. I residui dei due modelli sono incorrelati tra loro
        e il residuo del volume e' incorrelato con V_t.
Test 4: aggiornamento dei prezzi (rendimenti semplici vs logaritmici) e vincolo
        sulla lunghezza degli input iniziali.
"""
import numpy as np

from var import var_fit, simulate_var

P = 5
BURN = 1000
N = 1_000_000
SEED = 42


def true_model(p=P):
    """Parametri veri (ordine: const, [V_curr], V_l1..V_lp, R_l1..R_lp)."""
    a = np.array([0.40, 0.20, 0.10, 0.05, 0.02])      # V_{t-i} -> V_t
    b = np.array([0.50, -0.30, 0.10, 0.00, 0.05])     # R_{t-i} -> V_t
    c = np.array([-0.002, 0.001, 0.0, 0.0, 0.001])    # V_{t-i} -> R_t
    d = np.array([0.05, -0.03, 0.02, 0.0, 0.01])      # R_{t-i} -> R_t
    return {
        "p": p,
        "constant": True,
        "volume_model": {
            "params": np.concatenate([[0.5], a, b]),
            "sigma2": 0.25,
        },
        "return_model": {
            "params": np.concatenate([[0.0005], [0.003], c, d]),   # gamma = 0.003
            "sigma2": 1e-4,
        },
    }


def is_stable(model):
    """Stabilita' del sistema ridotto: autovalori della matrice companion < 1.

    Il sistema ridotto per (V, R) si ottiene sostituendo V_t nell'equazione di R_t.
    """
    p = model["p"]
    pv = model["volume_model"]["params"]
    pr = model["return_model"]["params"]
    a, b = pv[1:1+p], pv[1+p:]
    g, c, d = pr[1], pr[2:2+p], pr[2+p:]
    A = [np.array([[a[i], b[i]],
                   [g*a[i] + c[i], g*b[i] + d[i]]]) for i in range(p)]
    comp = np.zeros((2*p, 2*p))
    comp[:2, :] = np.hstack(A)
    comp[2:, :-2] = np.eye(2*(p-1))
    return np.max(np.abs(np.linalg.eigvals(comp))), None


def draw(model, n, seed):
    rng = np.random.default_rng(seed)
    p = model["p"]
    # I prezzi non servono qui: con rendimento medio > 0 su 100k passi andrebbero in
    # overflow, quindi si silenziano gli avvisi numerici.
    with np.errstate(over="ignore", invalid="ignore"):
        _, v, r = simulate_var(model, n + BURN, np.full(p, 1.0), np.zeros(p), rng=rng)
    return r[BURN:], v[BURN:]


ZMAX = 4.0   # soglia in unita' di errore standard (22 coefficienti -> 4 sigma e' prudente)


def report(title, labels, true, est, bse):
    """Confronta in unita' di errore standard: un errore assoluto fisso non ha senso,
    perche' la precisione dipende dalla scala dei regressori (R ~ 0.01, V ~ 1)."""
    print(f"\n{title}")
    print(f"{'coef':>8} {'vero':>10} {'stimato':>10} {'diff':>10} {'z':>7}")
    z = (est - true) / bse
    for l, t, e, zi in zip(labels, true, est, z):
        print(f"{l:>8} {t:10.5f} {e:10.5f} {e - t:10.5f} {zi:7.2f}")
    zm = np.max(np.abs(z))
    print(f"  max |z| = {zm:.2f} (soglia {ZMAX})")
    return zm


def main():
    truth = true_model()
    rho, _ = is_stable(truth)
    print(f"Raggio spettrale del modello vero: {rho:.4f}")
    assert rho < 1, "modello vero non stazionario"

    # ---------------- Test 1: recupero parametri ----------------
    r, v = draw(truth, N, SEED)
    fit = var_fit(r, v, p=P, constant=True)

    ok = True
    e1 = report("Test 1 - equazione del volume", fit["volume_model"]["labels"],
                truth["volume_model"]["params"], fit["volume_model"]["params"],
                fit["volume_model"]["bse"])
    e2 = report("Test 1 - equazione dei rendimenti", fit["return_model"]["labels"],
                truth["return_model"]["params"], fit["return_model"]["params"],
                fit["return_model"]["bse"])
    s1 = fit["volume_model"]["sigma2"] / truth["volume_model"]["sigma2"] - 1
    s2 = fit["return_model"]["sigma2"] / truth["return_model"]["sigma2"] - 1
    print(f"\n  sigma2 volume: vero {truth['volume_model']['sigma2']:.4f}, "
          f"stimato {fit['volume_model']['sigma2']:.4f} ({s1:+.2%})")
    print(f"  sigma2 rendimenti: vero {truth['return_model']['sigma2']:.6f}, "
          f"stimato {fit['return_model']['sigma2']:.6f} ({s2:+.2%})")
    sig_tol = ZMAX * np.sqrt(2.0 / N)   # sd relativa di sigma2 stimato ~ sqrt(2/N)
    t1 = e1 < ZMAX and e2 < ZMAX and abs(s1) < sig_tol and abs(s2) < sig_tol
    ok &= t1
    print(f"[Test 1] {'OK' if t1 else 'FALLITO'}")

    # ---------------- Test 2: stima -> simulazione ----------------
    r2, v2 = draw(fit, N, SEED + 1)
    fit2 = var_fit(r2, v2, p=P, constant=True)
    d1 = report("Test 2 - equazione del volume (ristima vs modello stimato)",
                fit["volume_model"]["labels"], fit["volume_model"]["params"],
                fit2["volume_model"]["params"], fit2["volume_model"]["bse"])
    d2 = report("Test 2 - equazione dei rendimenti (ristima vs modello stimato)",
                fit["return_model"]["labels"], fit["return_model"]["params"],
                fit2["return_model"]["params"], fit2["return_model"]["bse"])

    # Medie incondizionate teoriche: m = (I - sum A_i)^-1 * mu_ridotto
    p = P
    pv, pr = fit["volume_model"]["params"], fit["return_model"]["params"]
    a, b = pv[1:1+p], pv[1+p:]
    g, c, d = pr[1], pr[2:2+p], pr[2+p:]
    A = sum(np.array([[a[i], b[i]], [g*a[i] + c[i], g*b[i] + d[i]]]) for i in range(p))
    mu = np.array([pv[0], pr[0] + g*pv[0]])
    m = np.linalg.solve(np.eye(2) - A, mu)
    print(f"  media teorica   V={m[0]:.4f}, R={m[1]:.5f}")
    print(f"  media simulata  V={v2.mean():.4f}, R={r2.mean():.5f}")
    t2 = d1 < ZMAX and d2 < ZMAX \
        and abs(v2.mean() - m[0]) < 0.05 and abs(r2.mean() - m[1]) < 1e-3
    ok &= t2
    print(f"[Test 2] {'OK' if t2 else 'FALLITO'}")

    # ---------------- Test 3: struttura ricorsiva ----------------
    rn = len(r) - P
    v_lags = np.column_stack([v[P-i:len(v)-i] for i in range(1, P+1)])
    r_lags = np.column_stack([r[P-i:len(r)-i] for i in range(1, P+1)])
    Xv = np.column_stack([np.ones(rn), v_lags, r_lags])
    Xr = np.column_stack([np.ones(rn), v[P:], v_lags, r_lags])
    ev = v[P:] - Xv @ fit["volume_model"]["params"]
    er = r[P:] - Xr @ fit["return_model"]["params"]
    corr_ee = np.corrcoef(ev, er)[0, 1]
    corr_ev_v = np.corrcoef(ev, v[P:])[0, 1]   # atteso != 0 (V_t contiene e_v)
    corr_er_v = np.corrcoef(er, v[P:])[0, 1]   # atteso ~ 0 (V_t e' regressore)
    print(f"\nTest 3 - corr(e_v, e_r) = {corr_ee:+.4f}, corr(e_r, V_t) = {corr_er_v:+.4f}, "
          f"media residui: {ev.mean():+.2e}, {er.mean():+.2e}")
    t3 = abs(corr_ee) < 0.02 and abs(corr_er_v) < 0.02 \
        and abs(ev.mean()) < 1e-6 and abs(er.mean()) < 1e-6
    ok &= t3
    print(f"[Test 3] {'OK' if t3 else 'FALLITO'}")

    # ---------------- Test 4: prezzi e input ----------------
    rng = np.random.default_rng(7)
    pr_s, _, r_s = simulate_var(truth, 50, np.ones(P), np.zeros(P), initial_price=100.0, rng=rng)
    exp_s = 100.0 * np.cumprod(1 + r_s)
    rng = np.random.default_rng(7)
    pr_l, _, r_l = simulate_var(truth, 50, np.ones(P), np.zeros(P), initial_price=100.0,
                                rng=rng, log_returns=True)
    exp_l = 100.0 * np.exp(np.cumsum(r_l))
    # Riproducibilita' con stesso seed
    rng = np.random.default_rng(7)
    pr_s2, _, _ = simulate_var(truth, 50, np.ones(P), np.zeros(P), rng=rng)
    try:
        simulate_var(truth, 10, np.ones(P - 1), np.zeros(P))
        bad_len = False
    except ValueError:
        bad_len = True
    t4 = (np.allclose(pr_s, exp_s) and np.allclose(pr_l, exp_l)
          and np.allclose(pr_s, pr_s2) and bad_len)
    ok &= t4
    print(f"\nTest 4 - prezzi semplici: {np.allclose(pr_s, exp_s)}, "
          f"log: {np.allclose(pr_l, exp_l)}, riproducibilita': {np.allclose(pr_s, pr_s2)}, "
          f"controllo lunghezza: {bad_len}")
    print(f"[Test 4] {'OK' if t4 else 'FALLITO'}")

    print("\n" + ("TUTTI I TEST SUPERATI" if ok else "ALCUNI TEST FALLITI"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()