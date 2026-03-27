import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.optimize import curve_fit
from scipy.stats import sem
from statsmodels.tsa.stattools import acf

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14
})

# --- Fit Models ---
def power_law(x, A, delta):
    return A * x**delta

# --- Pooled ACF ---
def pooled_acf(series_list, nlags):
    """
    Stima l'ACF aggregando le autocovarianze di tutte le serie
    prima di calcolare il rapporto. Più corretta statisticamente
    rispetto alla media delle singole ACF.
    """
    sum_autocov = np.zeros(nlags + 1)

    for s in series_list:
        s = np.asarray(s, dtype=float)
        s = s - s.mean()
        n = len(s)
        for k in range(nlags + 1):
            if k < n:
                sum_autocov[k] += np.dot(s[k:], s[:n - k])

    return sum_autocov / sum_autocov[0]

# --- Data Loading ---
data_dir = 'database\\data'
paths = listdir(data_dir)
max_lag = 500
all_signs = []
all_daily_corrs = []

for path in paths:
    trades = pd.read_csv(f"{data_dir}\\{path}", header=None)
    signs = trades[3].values.astype(float)
    all_signs.append(signs)

    daily_corr = acf(signs, nlags=max_lag, fft=True)
    all_daily_corrs.append(daily_corr)

# --- Pooled ACF ---
pooled = pooled_acf(all_signs, nlags=max_lag)

# --- Barre di errore: SEM tra le ACF giornaliere, lag per lag ---
data_matrix = np.array(all_daily_corrs)
std_err = sem(data_matrix, axis=0)

lags = np.arange(1, max_lag + 1)
pooled_vals = pooled[1:]
err_vals    = std_err[1:]

# --- Fit power law: lag 1-20 e lag 21-max_lag ---
fit_range_early = np.arange(1, 21)
popt_early, pcov_early = curve_fit(power_law, fit_range_early, pooled[1:21])
perr_early = np.sqrt(np.diag(pcov_early))   # std dev dei parametri
print(f"Fit lag  1-20:")
print(f"  A     = {popt_early[0]:.6f} ± {perr_early[0]:.6f}")
print(f"  delta = {popt_early[1]:.6f} ± {perr_early[1]:.6f}")

fit_range_late = np.arange(21, max_lag + 1)
popt_late, pcov_late = curve_fit(power_law, fit_range_late, pooled[21:])
perr_late = np.sqrt(np.diag(pcov_late))
print(f"Fit lag 21-{max_lag}:")
print(f"  A     = {popt_late[0]:.6f} ± {perr_late[0]:.6f}")
print(f"  delta = {popt_late[1]:.6f} ± {perr_late[1]:.6f}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Pannello 1: prime 5 ACF giornaliere
ax = axes[0]
for i, daily_corr in enumerate(all_daily_corrs[:5]):
    ax.plot(np.arange(1, max_lag + 1), daily_corr[1:],
            alpha=0.8, linewidth=1.0, label=f'Day {i+1}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Lag')
ax.set_ylabel('ACF')
ax.set_title('Daily ACF')
ax.set_xlim([1.0, 50.0])

# Pannello 2: pooled ACF con banda di errore e due fit
ax = axes[1]

ax.fill_between(lags,
                pooled_vals - err_vals,
                pooled_vals + err_vals,
                color='steelblue', alpha=0.35, label='SEM')

ax.plot(lags, pooled_vals, color='steelblue', linewidth=1.5, label='Pooled ACF')

ax.plot(fit_range_early, power_law(fit_range_early, *popt_early),
        color='tomato', linewidth=2, linestyle='--',
        label=(f'Fit lag 1-20:\n'
               f'  A={popt_early[0]:.3f}±{perr_early[0]:.3f}\n'
               f'  δ={popt_early[1]:.3f}±{perr_early[1]:.3f}'))

ax.plot(fit_range_late, power_law(fit_range_late, *popt_late),
        color='darkorange', linewidth=2, linestyle='--',
        label=(f'Fit lag 21-{max_lag}:\n'
               f'  A={popt_late[0]:.3f}±{perr_late[0]:.3f}\n'
               f'  δ={popt_late[1]:.3f}±{perr_late[1]:.3f}'))

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Lag')
ax.set_ylabel('ACF')
ax.set_title('Pooled ACF')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('images\\correlation_trade_sign.png', dpi=300, bbox_inches='tight')
plt.show()