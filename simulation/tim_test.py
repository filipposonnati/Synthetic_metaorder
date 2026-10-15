"""Script di verifica per regression_delta.py

Controlli:
  1. power_transform: casi noti (delta = 1, 0, 0.5) e simmetria
  2. _lags: allineamento temporale della matrice dei ritardi
  3. impact_fit: recupero dei parametri veri su dati sintetici (DGP noto)
  4. impact_simulate: forma, riproducibilita' col seed, prezzi positivi
  5. impact_simulate: confronto con ricorsione manuale a rumore zero
  6. Round trip: simula -> rifitta -> i parametri devono coincidere
  7. Validazione input (errori attesi)
"""
import sys
import numpy as np

from regression_delta import power_transform, _lags, impact_fit, impact_simulate

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ----------------------------------------------------------------------------
# Parametri veri del processo generatore (DGP)
# ----------------------------------------------------------------------------
P = 3
DELTA = 0.5
V_PARAMS = np.array([0.1, 0.4, 0.2, 0.1])           # const, phi_1..phi_p
V_SIGMA = 1.0
R_PARAMS = np.array([0.0005, 0.003, 0.002, 0.001, 0.0005])  # const, b_0..b_p
R_SIGMA = 0.002


def generate_data(n, seed):
    """Genera volumi AR(p) e rendimenti dal modello vero."""
    rng = np.random.default_rng(seed)
    v = np.zeros(n)
    for t in range(P, n):
        lag = v[t - P:t][::-1]
        v[t] = V_PARAMS[0] + V_PARAMS[1:] @ lag + rng.normal(0, V_SIGMA)
    tv = power_transform(v, DELTA)
    r = np.zeros(n)
    for t in range(P, n):
        window = tv[t - P:t + 1][::-1]  # V_t, V_{t-1}, ..., V_{t-p}
        r[t] = R_PARAMS[0] + R_PARAMS[1:] @ window + rng.normal(0, R_SIGMA)
    return r, v


# ----------------------------------------------------------------------------
# 1. power_transform
# ----------------------------------------------------------------------------
x = np.array([-4.0, -1.0, 0.0, 2.0, 9.0])
check("power_transform delta=1 e' l'identita'",
      np.allclose(power_transform(x, 1.0), x))
check("power_transform delta=0 restituisce il segno",
      np.allclose(power_transform(x, 0.0), [-1, -1, 0, 1, 1]) or
      np.allclose(power_transform(x, 0.0), np.sign(x) * np.abs(x) ** 0.0))
check("power_transform delta=0.5 valori attesi",
      np.allclose(power_transform(x, 0.5), [-2, -1, 0, np.sqrt(2), 3]))
check("power_transform e' dispari: f(-x) = -f(x)",
      np.allclose(power_transform(-x, 0.7), -power_transform(x, 0.7)))

# ----------------------------------------------------------------------------
# 2. _lags
# ----------------------------------------------------------------------------
a = np.arange(10, dtype=float)
L = _lags(a, 3)
check("_lags: shape (n-p, p)", L.shape == (7, 3), str(L.shape))
check("_lags: riga t contiene [x_{t-1}, x_{t-2}, x_{t-3}]",
      all(np.allclose(L[i], [a[i + 2], a[i + 1], a[i]]) for i in range(L.shape[0])))

# ----------------------------------------------------------------------------
# 3. impact_fit: recupero parametri
# ----------------------------------------------------------------------------
N = 200_000
r, v = generate_data(N, seed=1)
res = impact_fit(r, v, delta=DELTA, p=P)

vp = res["volume_model"]["params"]
rp = res["return_model"]["params"]
check("fit volumi: parametri recuperati",
      np.allclose(vp, V_PARAMS, atol=0.02),
      f"{vp} {V_PARAMS}")
check("fit volumi: sigma2 recuperata",
      abs(res["volume_model"]["sigma2"] / V_SIGMA ** 2 - 1) < 0.02,
      f"{res['volume_model']['sigma2']:.4f} vs {V_SIGMA ** 2:.4f}")
check("fit rendimenti: parametri recuperati",
      np.allclose(rp, R_PARAMS, atol=2e-4),
      f"{rp} {R_PARAMS}")
check("fit rendimenti: sigma2 recuperata",
      abs(res["return_model"]["sigma2"] / R_SIGMA ** 2 - 1) < 0.02,
      f"{res['return_model']['sigma2']:.3e} vs {R_SIGMA ** 2:.3e}")
check("fit: dimensioni dei parametri",
      vp.shape == (P + 1,) and rp.shape == (P + 2,),
      f"vol {vp.shape}, ret {rp.shape}")

# ----------------------------------------------------------------------------
# 4. impact_simulate: proprieta' di base
# ----------------------------------------------------------------------------
init_v = v[-P:]
n_steps = 1000
prices, sv, sr = impact_simulate(res, n_steps, init_v, initial_price=100.0, seed=7)
check("simulate: lunghezze di output",
      len(prices) == n_steps and len(sv) == n_steps and len(sr) == n_steps)
check("simulate: nessun NaN/inf",
      np.all(np.isfinite(prices)) and np.all(np.isfinite(sv)) and np.all(np.isfinite(sr)))
check("simulate: prezzi positivi", np.all(prices > 0))

p2, v2, r2 = impact_simulate(res, n_steps, init_v, initial_price=100.0, seed=7)
check("simulate: stesso seed -> stesso risultato",
      np.array_equal(prices, p2) and np.array_equal(sv, v2) and np.array_equal(sr, r2))
p3, _, _ = impact_simulate(res, n_steps, init_v, initial_price=100.0, seed=8)
check("simulate: seed diverso -> risultato diverso", not np.array_equal(prices, p3))

check("simulate: prezzo allineato (P_0 = initial, P_k = P_{k-1}(1+r_{k-1}))",
      np.isclose(prices[0], 100.0) and
      np.allclose(prices[1:], prices[:-1] * (1 + sr[:-1])))

# ----------------------------------------------------------------------------
# 5. Confronto con ricorsione manuale a rumore zero
# ----------------------------------------------------------------------------
det = {
    "p": P, "delta": DELTA,
    "volume_model": {"params": V_PARAMS, "sigma2": 0.0},
    "return_model": {"params": R_PARAMS, "sigma2": 0.0},
}
init = np.array([1.0, -0.5, 2.0])  # dal piu' vecchio al piu' recente
pd_, vd_, rd_ = impact_simulate(det, 20, init, initial_price=50.0, seed=0)

hist = list(init)
man_v, man_r, man_p = [], [], []
price = 50.0
for _ in range(20):
    lag = np.array(hist[-P:][::-1])  # V_{t-1}, ..., V_{t-p}
    vt = V_PARAMS[0] + V_PARAMS[1:] @ lag
    window = np.concatenate([[vt], lag])
    rt = R_PARAMS[0] + R_PARAMS[1:] @ power_transform(window, DELTA)
    price *= 1 + rt
    hist.append(vt)
    man_v.append(vt); man_r.append(rt); man_p.append(price)

def manual_recursion(init, n, shock_v=0.0, shock_r=0.0, price0=50.0):
    hist = list(init)
    man_v, man_r, man_p = [], [], []
    price = price0
    for _ in range(n):
        lag = np.array(hist[-P:][::-1])  # V_{t-1}, ..., V_{t-p}
        vt = V_PARAMS[0] + V_PARAMS[1:] @ lag + shock_v
        window = np.concatenate([[vt], lag])
        rt = R_PARAMS[0] + R_PARAMS[1:] @ power_transform(window, DELTA) + shock_r
        man_p.append(price)          # P_t, prima di applicare r_t
        price *= 1 + rt
        hist.append(vt)
        man_v.append(vt); man_r.append(rt)
    return man_v, man_r, man_p


man_v, man_r, man_p = manual_recursion(init, 20)

check("deterministico: volumi = ricorsione manuale", np.allclose(vd_, man_v))
check("deterministico: rendimenti = ricorsione manuale", np.allclose(rd_, man_r))
check("deterministico: prezzi = ricorsione manuale (allineati)", np.allclose(pd_, man_p))
check("deterministico: primo prezzo = initial_price", np.isclose(pd_[0], 50.0))

# ----------------------------------------------------------------------------
# 6. Round trip: simula -> rifitta
# ----------------------------------------------------------------------------
true_like = {
    "p": P, "delta": DELTA,
    "volume_model": {"params": V_PARAMS, "sigma2": V_SIGMA ** 2},
    "return_model": {"params": R_PARAMS, "sigma2": R_SIGMA ** 2},
}
big_p, big_v, big_r = impact_simulate(true_like, 200_000, np.zeros(P), seed=3)
res2 = impact_fit(big_r, big_v, delta=DELTA, p=P)
check("round trip: parametri volume",
      np.allclose(res2["volume_model"]["params"], V_PARAMS, atol=0.02),
      f"max err = {np.max(np.abs(res2['volume_model']['params'] - V_PARAMS)):.4f}")
check("round trip: parametri rendimento",
      np.allclose(res2["return_model"]["params"], R_PARAMS, atol=2e-4),
      f"max err = {np.max(np.abs(res2['return_model']['params'] - R_PARAMS)):.2e}")

# ----------------------------------------------------------------------------
# 7. Validazione input
# ----------------------------------------------------------------------------
def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


check("impact_fit: lunghezze diverse -> ValueError",
      raises(lambda: impact_fit(r[:100], v[:99], delta=DELTA, p=P)))
check("impact_simulate: initial_v di lunghezza errata -> ValueError",
      raises(lambda: impact_simulate(res, 10, np.zeros(P + 1))))

# ----------------------------------------------------------------------------
# 8. Bootstrap dei residui
# ----------------------------------------------------------------------------
check("fit: residui salvati con lunghezza N-p",
      len(res["volume_model"]["resid"]) == N - P and
      len(res["return_model"]["resid"]) == N - P)

# residui costanti -> il bootstrap equivale a uno shock costante
c_v, c_r = 0.3, -0.001
boot_det = {
    **det,
    "volume_model": {**det["volume_model"], "resid": np.full(5, c_v)},
    "return_model": {**det["return_model"], "resid": np.full(5, c_r)},
}
pb, vb, rb = impact_simulate(boot_det, 20, init, initial_price=50.0, seed=0, bootstrap=True)
mv, mr, mp = manual_recursion(init, 20, shock_v=c_v, shock_r=c_r)
check("bootstrap: residui costanti = ricorsione manuale con shock",
      np.allclose(vb, mv) and np.allclose(rb, mr) and np.allclose(pb, mp))

# parametri nulli -> l'output coincide con gli shock; verifica l'accoppiamento
pair = {
    "p": P, "delta": DELTA,
    "volume_model": {"params": np.zeros(P + 1), "sigma2": 0.0,
                     "resid": np.arange(5) * 0.1},
    "return_model": {"params": np.zeros(P + 2), "sigma2": 0.0,
                     "resid": np.arange(5) * 0.001},
}
_, vp_, rp_ = impact_simulate(pair, 200, np.zeros(P), seed=5, bootstrap=True)
check("bootstrap: volume e rendimento estratti dallo stesso istante",
      np.allclose(rp_, vp_ * 0.01))

# riproducibilita' e round trip
_, vb1, rb1 = impact_simulate(res, 500, init_v, seed=11, bootstrap=True)
_, vb2, rb2 = impact_simulate(res, 500, init_v, seed=11, bootstrap=True)
check("bootstrap: stesso seed -> stesso risultato",
      np.array_equal(vb1, vb2) and np.array_equal(rb1, rb2))

bp_, bv_, br_ = impact_simulate(res, 200_000, init_v, seed=3, bootstrap=True)
res3 = impact_fit(br_, bv_, delta=DELTA, p=P)
check("bootstrap round trip: parametri volume",
      np.allclose(res3["volume_model"]["params"], V_PARAMS, atol=0.02))
check("bootstrap round trip: parametri rendimento",
      np.allclose(res3["return_model"]["params"], R_PARAMS, atol=2e-4))

# ----------------------------------------------------------------------------
print()
print(f"{sum(RESULTS)}/{len(RESULTS)} controlli superati")
sys.exit(0 if all(RESULTS) else 1)