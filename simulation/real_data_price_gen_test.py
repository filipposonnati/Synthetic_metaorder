"""
verify_models.py
================
Verifica dei due modelli di real_data_price_gen.py (fit + simulazione):
  - 'regression' (OLS su p+1 lag di sign*|v|^delta)
  - 'tim'        (Transient Impact Model, propagatore G(l) = (l+1)^(-beta))

Test eseguiti
-------------
 A. Test deterministici (rumore = 0, impulso singolo): verificano che la
    simulazione ponga l'impatto ESATTAMENTE dove deve, per impact_lag = 0 e 1.
 B. Recupero dei parametri su dati sintetici con verita' nota:
      - OLS: const, theta_k, sigma (per impact_lag 0/1 e delta 1.0/0.5)
      - TIM: const, sigma_f, sigma_eta + scansione di beta (il minimo del
        residuo deve cadere sul beta vero)
 C. Round-trip: fit -> simulate -> refit sui prezzi simulati ritrova i parametri.
 D. Pipeline run(): timestamp/volumi/segni preservati, prezzi cambiati e positivi,
    tick_size rispettato, giorni troppo corti saltati.
 E. (opzionale) Dati reali: R^2 in-sample e confronto statistico reale vs simulato.

Uso:
    python verify_models.py
    python verify_models.py --data-dir ../database/data --file 20200102.csv \
                            --p 100 --beta 0.25 --delta 0.5 --L 500
"""

import argparse
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.cwd()))
import real_data_price_gen as g  # noqa: E402

RESULTS = []


# ---------------------------------------------------------------------------
# Mini framework
# ---------------------------------------------------------------------------

def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def section(title):
    print(f"\n=== {title} ===")


def run_test(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as e:  # un'eccezione e' un FAIL, non deve fermare gli altri test
        check(f"{fn.__name__} (eccezione)", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Dati sintetici
# ---------------------------------------------------------------------------

def make_trades(n, rng, p_stay=0.7):
    """Segni persistenti (catena di Markov) e volumi lognormali."""
    flips = np.where(rng.random(n) < p_stay, 1.0, -1.0)
    signs = np.cumprod(flips) * rng.choice([-1.0, 1.0])
    volumes = rng.lognormal(mean=0.0, sigma=0.5, size=n)
    return signs, volumes


def tol_coef(sigma, x_std, n, k=8.0):
    """Tolleranza ~ k errori standard OLS per un regressore con std x_std."""
    return k * sigma / (x_std * np.sqrt(n))


# ---------------------------------------------------------------------------
# A. Test deterministici
# ---------------------------------------------------------------------------

def test_kernel():
    section("A0. Kernel dei rendimenti del TIM")
    beta, L = 0.4, 50
    K = g._return_kernel(beta, L)
    G = (np.arange(L) + 1.0) ** (-beta)
    check("lunghezza kernel = L+1", len(K) == L + 1, f"{len(K)}")
    check("K(0) = G(0) = 1", np.isclose(K[0], 1.0))
    check("K(l) = G(l) - G(l-1) per 1<=l<L", np.allclose(K[1:L], np.diff(G)))
    check("K(L) = -G(L-1) (troncatura)", np.isclose(K[L], -G[-1]))
    check("somma(K) = 0 (G(-1)=G(L)=0, telescopica)", abs(K.sum()) < 1e-12, f"{K.sum():.2e}")


def _impulse_trades(N=60, t0=20, vol=4.0):
    signs = np.ones(N)
    volumes = np.zeros(N)
    volumes[t0] = vol
    return signs, volumes, t0


def test_ols_impulse():
    section("A1. OLS: risposta a un impulso (rumore = 0)")
    theta = np.array([1e-3, 5e-4, 2e-4])
    for impact_lag in (0, 1):
        for delta in (1.0, 0.5):
            signs, volumes, t0 = _impulse_trades()
            rng = np.random.default_rng(0)
            prices = g.simulate_ols(100.0, volumes, signs, const=0.0, theta=theta, sigma=0.0,
                                    rng=rng, delta=delta, impact_lag=impact_lag)
            r = np.diff(np.log(prices))
            idx = t0 - 1 + impact_lag
            expected = np.zeros(len(r))
            expected[idx: idx + len(theta)] = theta * volumes[t0] ** delta
            check(f"impulso al posto giusto (impact_lag={impact_lag}, delta={delta})",
                  np.allclose(r, expected, atol=1e-12),
                  f"max err {np.max(np.abs(r - expected)):.2e}")


def test_tim_impulse():
    section("A2. TIM: livello del log-prezzo = sigma_f * d * G(l) (rumore = 0)")
    sigma_f, beta, L = 0.01, 0.5, 10
    for impact_lag in (0, 1):
        for delta in (1.0, 0.5):
            signs, volumes, t0 = _impulse_trades(N=60)
            rng = np.random.default_rng(0)
            prices = g.simulate_tim(100.0, volumes, signs, const=0.0, sigma_f=sigma_f,
                                    sigma_eta=0.0, beta=beta, delta=delta, kernel_L=L,
                                    rng=rng, impact_lag=impact_lag)
            idx = t0 - 1 + impact_lag           # indice del rendimento colpito
            expected = np.zeros(len(prices))
            Gl = (np.arange(L) + 1.0) ** (-beta)
            expected[idx + 1: idx + 1 + L] = sigma_f * volumes[t0] ** delta * Gl
            got = np.log(prices / prices[0])
            check(f"propagatore sul livello (impact_lag={impact_lag}, delta={delta})",
                  np.allclose(got, expected, atol=1e-10),
                  f"max err {np.max(np.abs(got - expected)):.2e}")
            check(f"impatto decade a 0 dopo L passi (impact_lag={impact_lag}, delta={delta})",
                  np.allclose(got[idx + 1 + L:], 0.0, atol=1e-10))


# ---------------------------------------------------------------------------
# B. Recupero parametri
# ---------------------------------------------------------------------------

def test_ols_recovery():
    section("B1. OLS: recupero dei parametri su dati sintetici")
    N, p = 200_000, 10
    const_true, sigma_true = 2e-3, 1e-3
    theta_true = 2e-4 * (np.arange(p + 1) + 1.0) ** (-0.5)
    for impact_lag in (0, 1):
        for delta in (1.0, 0.5):
            rng = np.random.default_rng(10 + impact_lag)
            signs, volumes = make_trades(N, rng)
            prices = g.simulate_ols(100.0, volumes, signs, const_true, theta_true,
                                    sigma_true, rng, delta=delta, impact_lag=impact_lag)
            c, th, s = g.fit_ols(prices, volumes, signs, p, delta=delta, impact_lag=impact_lag)
            d_std = np.std(g._signed_impact(signs, volumes, delta))
            tol = tol_coef(sigma_true, d_std, N)
            tag = f"(impact_lag={impact_lag}, delta={delta})"
            check(f"theta_k recuperati {tag}", np.all(np.abs(th - theta_true) < tol),
                  f"max|err|={np.max(np.abs(th - theta_true)):.2e}, tol={tol:.2e}")
            check(f"costante recuperata {tag}", abs(c - const_true) < 8 * sigma_true / np.sqrt(N),
                  f"{c:.2e} vs {const_true:.2e}")
            check(f"sigma recuperato {tag}", abs(s / sigma_true - 1) < 0.01,
                  f"{s:.3e} vs {sigma_true:.3e}")

    # Un lag sbagliato in calibrazione deve peggiorare il fit (sanity sull'allineamento)
    rng = np.random.default_rng(99)
    signs, volumes = make_trades(N, rng)
    prices = g.simulate_ols(100.0, volumes, signs, const_true, theta_true, sigma_true,
                            rng, delta=1.0, impact_lag=1)
    _, _, s_ok = g.fit_ols(prices, volumes, signs, p, impact_lag=1)
    _, _, s_bad = g.fit_ols(prices, volumes, signs, p, impact_lag=0)
    check("impact_lag errato => residuo maggiore", s_bad > s_ok * 1.001,
          f"sigma ok={s_ok:.4e}, sigma lag errato={s_bad:.4e}")


def test_tim_recovery():
    section("B2. TIM: recupero dei parametri su dati sintetici")
    N, L = 100_000, 100
    const_true, sigma_f_true, sigma_eta_true, beta_true = 1e-3, 5e-4, 1e-3, 0.3
    for impact_lag in (0, 1):
        for delta in (1.0, 0.5):
            rng = np.random.default_rng(20 + impact_lag)
            signs, volumes = make_trades(N, rng)
            prices = g.simulate_tim(100.0, volumes, signs, const_true, sigma_f_true,
                                    sigma_eta_true, beta_true, delta, L, rng,
                                    impact_lag=impact_lag)
            c, sf, se = g.fit_tim(prices, volumes, signs, beta_true, delta, L, impact_lag)
            X = g._build_impact_signal(signs, volumes, beta_true, delta, L, impact_lag)
            tol = tol_coef(sigma_eta_true, np.std(X), len(X))
            tag = f"(impact_lag={impact_lag}, delta={delta})"
            check(f"sigma_f recuperato {tag}", abs(sf - sigma_f_true) < tol,
                  f"{sf:.3e} vs {sigma_f_true:.3e}, tol={tol:.1e}")
            check(f"costante recuperata {tag}", abs(c - const_true) < 8 * sigma_eta_true / np.sqrt(N),
                  f"{c:.2e} vs {const_true:.2e}")
            check(f"sigma_eta recuperato {tag}", abs(se / sigma_eta_true - 1) < 0.01,
                  f"{se:.3e} vs {sigma_eta_true:.3e}")

    # Scansione di beta: il residuo deve essere minimo al beta vero
    rng = np.random.default_rng(31)
    signs, volumes = make_trades(N, rng)
    prices = g.simulate_tim(100.0, volumes, signs, const_true, sigma_f_true, sigma_eta_true,
                            beta_true, 1.0, L, rng, impact_lag=1)
    grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    sig = [g.fit_tim(prices, volumes, signs, b, 1.0, L, 1)[2] for b in grid]
    best = grid[int(np.argmin(sig))]
    check("scansione beta: minimo del residuo al beta vero", best == beta_true,
          "sigma_eta: " + ", ".join(f"b={b}:{s:.4e}" for b, s in zip(grid, sig)))


# ---------------------------------------------------------------------------
# C. Round-trip fit -> simulate -> refit
# ---------------------------------------------------------------------------

def test_roundtrip():
    section("C. Round-trip: fit -> simulate -> refit")
    N = 150_000
    rng = np.random.default_rng(5)
    signs, volumes = make_trades(N, rng)
    # "Dati reali" sintetici
    p, L, beta, delta = 10, 100, 0.3, 0.5
    theta_true = 2e-4 * (np.arange(p + 1) + 1.0) ** (-0.5)
    real = g.simulate_ols(100.0, volumes, signs, 0.0, theta_true, 1e-3, rng, delta=delta)

    # OLS
    c1, th1, s1 = g.fit_ols(real, volumes, signs, p, delta=delta)
    sim = g.calibrate_and_simulate_ols(real, volumes, signs, 100.0, p,
                                       np.random.default_rng(1), delta=delta)
    c2, th2, s2 = g.fit_ols(sim, volumes, signs, p, delta=delta)
    tol = tol_coef(s1, np.std(g._signed_impact(signs, volumes, delta)), N, k=10)
    check("OLS refit coerente col fit originale", np.all(np.abs(th2 - th1) < tol),
          f"max|d theta|={np.max(np.abs(th2 - th1)):.2e}, tol={tol:.2e}")
    check("OLS sigma coerente", abs(s2 / s1 - 1) < 0.01, f"{s2:.3e} vs {s1:.3e}")

    # TIM (prezzi reali generati con TIM)
    real_t = g.simulate_tim(100.0, volumes, signs, 0.0, 5e-4, 1e-3, beta, delta, L, rng)
    c1, sf1, se1 = g.fit_tim(real_t, volumes, signs, beta, delta, L)
    sim_t = g.calibrate_and_simulate_tim(real_t, volumes, signs, 100.0, beta, delta, L,
                                         np.random.default_rng(2))
    c2, sf2, se2 = g.fit_tim(sim_t, volumes, signs, beta, delta, L)
    X = g._build_impact_signal(signs, volumes, beta, delta, L, 1)
    tol = tol_coef(se1, np.std(X), len(X), k=10)
    check("TIM refit coerente (sigma_f)", abs(sf2 - sf1) < tol,
          f"{sf2:.3e} vs {sf1:.3e}, tol={tol:.1e}")
    check("TIM sigma_eta coerente", abs(se2 / se1 - 1) < 0.01, f"{se2:.3e} vs {se1:.3e}")

    # Stessa seed => stessi prezzi (riproducibilita')
    a = g.calibrate_and_simulate_tim(real_t, volumes, signs, 100.0, beta, delta, L,
                                     np.random.default_rng(7))
    b = g.calibrate_and_simulate_tim(real_t, volumes, signs, 100.0, beta, delta, L,
                                     np.random.default_rng(7))
    check("riproducibilita' con stessa seed", np.array_equal(a, b))


# ---------------------------------------------------------------------------
# D. Pipeline run()
# ---------------------------------------------------------------------------

def test_pipeline():
    section("D. Pipeline run() su file temporanei")
    rng = np.random.default_rng(3)
    N = 5_000
    signs, volumes = make_trades(N, rng)
    ts = np.cumsum(rng.exponential(1.0, N))
    real_prices = g.simulate_ols(100.0, volumes, signs, 0.0, np.array([2e-4, 1e-4]),
                                 5e-4, rng)
    real_prices = np.round(real_prices, 2)

    for model in ("regression", "tim"):
        with tempfile.TemporaryDirectory() as tmp:
            din, dout = Path(tmp) / "in", Path(tmp) / "out"
            din.mkdir()
            pd.DataFrame({0: ts, 1: real_prices, 2: volumes, 3: signs.astype(int)}) \
                .to_csv(din / "day1.csv", index=False, header=False)
            # file troppo corto (2 trade < 3): deve essere saltato da entrambi i modelli
            pd.DataFrame({0: ts[:2], 1: real_prices[:2], 2: volumes[:2], 3: signs[:2].astype(int)}) \
                .to_csv(din / "short.csv", index=False, header=False)

            tick = 0.01
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g.run(data_dir=din, out_dir=dout, price_model=model, vol_lags_for_ret=5,
                      beta=0.3, delta=0.5, kernel_L=20, impact_lag=1,
                      tick_size=tick, seed=1)

            out = dout / "day1.csv"
            check(f"[{model}] file di output creato", out.exists())
            check(f"[{model}] nessun file extra nella cartella di output",
                  sorted(p_.name for p_ in dout.iterdir()) == ["day1.csv"])
            check(f"[{model}] giorno troppo corto saltato", not (dout / "short.csv").exists())
            if not out.exists():
                continue
            o = pd.read_csv(out, header=None).to_numpy()
            check(f"[{model}] timestamp preservati", np.allclose(o[:, 0], ts))
            check(f"[{model}] volumi preservati", np.allclose(o[:, 2], np.abs(volumes)))
            check(f"[{model}] segni preservati", np.array_equal(o[:, 3], signs.astype(int)))
            check(f"[{model}] primo prezzo = P0 reale", np.isclose(o[0, 1], real_prices[0]))
            check(f"[{model}] prezzi finiti e positivi", np.all(np.isfinite(o[:, 1])) and np.all(o[:, 1] > 0))
            check(f"[{model}] prezzi diversi dai reali", not np.allclose(o[:, 1], real_prices))
            check(f"[{model}] prezzi su griglia di tick",
                  np.allclose(o[:, 1] / tick, np.round(o[:, 1] / tick), atol=1e-6))

    section("D2. Validazione input")
    ok = True
    for bad_prices, label in ((np.array([100.0, -1.0, 100.0, 100.0]), "prezzi <= 0"),
                              (np.array([100.0, np.nan, 100.0, 100.0]), "prezzi NaN")):
        try:
            g.fit_tim(bad_prices, np.ones(4), np.ones(4), 0.3, 1.0, 3)
            ok = False
        except ValueError:
            pass
    check("input invalidi => ValueError", ok)
    try:
        g.fit_ols(np.full(20, 100.0), np.ones(20), np.ones(20), p=30)
        raised = False
    except ValueError:
        raised = True
    check("p troppo grande => ValueError", raised)
    try:
        g._align(np.arange(5), 2)
        raised = False
    except ValueError:
        raised = True
    check("impact_lag non valido => ValueError", raised)


# ---------------------------------------------------------------------------
# E. Dati reali (opzionale)
# ---------------------------------------------------------------------------

def test_real(data_dir, fname, p, beta, delta, L, impact_lag, seed):
    section(f"E. Dati reali: {fname}")
    ts, prices, vols, signs = g.open_real_data(fname, data_dir)
    y = g._log_returns(prices)
    print(f"  N trade = {len(prices)},  std rendimenti reali = {y.std():.3e}")

    def r2(sigma, k):
        return 1.0 - (sigma ** 2) * (len(y) - k) / ((y - y.mean()) ** 2).sum()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c, th, s = g.fit_ols(prices, vols, signs, p, delta=delta, impact_lag=impact_lag)
        c2, sf, se = g.fit_tim(prices, vols, signs, beta, delta, L, impact_lag)
    print(f"  OLS: sigma={s:.3e}  R2~{r2(s, p + 2):.4f}  theta_0={th[0]:.3e}")
    print(f"  TIM: sigma_eta={se:.3e}  R2~{r2(se, 2):.4f}  sigma_f={sf:.3e}")

    rng = np.random.default_rng(seed)
    sims = {
        "regression": g.calibrate_and_simulate_ols(prices, vols, signs, prices[0], p, rng,
                                                   delta=delta, impact_lag=impact_lag),
        "tim": g.calibrate_and_simulate_tim(prices, vols, signs, prices[0], beta, delta, L, rng,
                                            impact_lag=impact_lag),
    }
    sign_r = lambda x: np.sign(x[x != 0]) if False else x
    for name, sp in sims.items():
        ys = g._log_returns(sp)
        ac = lambda z, k=1: np.corrcoef(z[:-k], z[k:])[0, 1]
        # Correlazione con il flusso d'ordini firmato (stessa che nei dati reali)
        d = g._align(g._signed_impact(signs, vols, delta), impact_lag)
        print(f"  [{name}] std r: sim={ys.std():.3e} reale={y.std():.3e} | "
              f"AC1: sim={ac(ys):+.3f} reale={ac(y):+.3f} | "
              f"corr(r, sign*|v|^delta): sim={np.corrcoef(ys, d)[0, 1]:+.3f} "
              f"reale={np.corrcoef(y, d)[0, 1]:+.3f}")
        check(f"[{name}] std rendimenti simulati entro 10% dai reali (in-sample)",
              abs(ys.std() / y.std() - 1) < 0.10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Verifica dei modelli regression e tim")
    ap.add_argument("--data-dir", default=None, help="cartella dati reali (test E)")
    ap.add_argument("--file", default=None, help="file giornaliero (default: primo *.csv)")
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--L", type=int, default=500)
    ap.add_argument("--impact-lag", type=int, default=1, choices=(0, 1))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for t in (test_kernel, test_ols_impulse, test_tim_impulse, test_ols_recovery,
              test_tim_recovery, test_roundtrip, test_pipeline):
        run_test(t)

    if args.data_dir:
        dd = Path(args.data_dir)
        fname = args.file or (sorted(f.name for f in dd.glob("*.csv")) or [None])[0]
        if fname is None:
            print(f"\n(nessun file .csv in {dd}: test E saltato)")
        else:
            run_test(test_real, dd, fname, args.p, args.beta, args.delta, args.L,
                     args.impact_lag, args.seed)

    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{'=' * 50}\nRISULTATO: {n_ok}/{len(RESULTS)} test superati")
    for name, ok in RESULTS:
        if not ok:
            print(f"  FALLITO: {name}")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()