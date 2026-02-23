import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.optimize import curve_fit
from scipy.stats import sem, ks_2samp

# --- Fit Models ---
def power_law(tau, L, gamma):
    """Simple Power Law: f(x) = L * x^(-gamma)"""
    return L * np.power(tau, -gamma)

# --- Function for automatic tau_min (Tail) search ---
def find_optimal_tau_min_tail(x, y, y_err , model_func, p0):
    """
    Finds the minimum lag (tau_min) that minimizes the KS distance 
    between the data tail and the model.
    """
    # Test tau_min between the first lag and 50% of the range
    start_search = 1
    end_search = int(len(x) * 0.5)
    
    best_tau_min = x[0]
    min_ks = np.inf
    
    for i in range(start_search, end_search):
        tau_test = x[i]
        # Select data for the tail: x >= tau_test
        mask = x >= tau_test
        if len(x[mask]) < 10: continue  # Ensure enough points for fit
        
        try:
            popt, _ = curve_fit(model_func, x[mask], y[mask], sigma=y_err[mask], absolute_sigma=True, p0=p0, maxfev=2000)
            y_model = model_func(x[mask], *popt)
            
            # KS test on normalized shape (treating it like a probability distribution)
            stat, _ = ks_2samp(y[mask]/np.sum(y[mask]), y_model/np.sum(y_model))
            
            if stat < min_ks:
                min_ks = stat
                best_tau_min = tau_test
        except:
            continue
    return best_tau_min

# --- Function for automatic tau_max (Head) search ---
def find_optimal_tau_max_head(x, y, y_err, model_func, p0, limit_idx):
    """
    Finds the maximum lag (tau_max) that minimizes the KS distance 
    between the data head and the model.
    """
    # We search for a cutoff point for the head.
    # We start from a minimum window (e.g., 5 points) up to the provided limit (start of tail).
    start_search = 5 
    end_search = limit_idx 
    
    best_tau_max = x[start_search]
    min_ks = np.inf
    
    for i in range(start_search, end_search):
        tau_test = x[i]
        # Select data for the head: x <= tau_test
        mask = x <= tau_test
        
        # Skip if too few points or if range is empty
        if len(x[mask]) < 5: continue 

        try:
            popt, _ = curve_fit(model_func, x[mask], y[mask], sigma=y_err[mask], absolute_sigma=True, p0=p0, maxfev=2000)
            y_model = model_func(x[mask], *popt)
            
            # KS test on normalized shape
            stat, _ = ks_2samp(y[mask]/np.sum(y[mask]), y_model/np.sum(y_model))
            
            # We want to minimize KS distance
            if stat < min_ks:
                min_ks = stat
                best_tau_max = tau_test
        except:
            continue
            
    return best_tau_max

plt.rcParams.update({
    'font.size': 12,          # Dimensione base per tutto il testo
    'axes.titlesize': 20,     # Titolo
    'axes.labelsize': 16,     # Etichette assi
    'xtick.labelsize': 12,    # Numeri asse X
    'ytick.labelsize': 12,    # Numeri asse Y
    'legend.fontsize': 14     # Legenda
})

# --- Data Loading ---
data_dir = 'database\\data'
paths = listdir(data_dir)
max_lag = 1000
all_daily_corrs = []

print("Loading data...")
for path in paths:
    try:
        trades = pd.read_csv(f"{data_dir}\\{path}", header=None)
        signs = trades[3].values.astype(float)
        # Calculate sign autocorrelation
        daily_corr = [np.mean(signs[lag:] * signs[:-lag]) for lag in range(1, max_lag + 1)]
        all_daily_corrs.append(daily_corr)
    except Exception as e:
        print(f"Skipping file {path}: {e}")

if not all_daily_corrs:
    raise ValueError("No data loaded. Check directory path.")

data_matrix = np.array(all_daily_corrs)
avg_corr = np.mean(data_matrix, axis=0)
#std_error = np.sqrt(np.var(data_matrix, axis=0))
std_error = sem(data_matrix, axis=0)
lags = np.arange(1, max_lag + 1)

# --- Optimized Tail Analysis (Power Law) ---
print("Analyzing Tail...")
# Find best start point for the tail
tau_min_tail = find_optimal_tau_min_tail(lags, avg_corr, std_error, power_law, [0.1, 0.5])
mask_tail = lags >= tau_min_tail
popt_tail, pcov_tail = curve_fit(power_law, lags[mask_tail], avg_corr[mask_tail], sigma=std_error[mask_tail], absolute_sigma=True)

# --- Optimized Head Analysis (Power Law) ---
print("Analyzing Head...")
# Find the index where the tail starts to use as an upper limit for the head search
tail_start_idx = np.where(lags == tau_min_tail)[0][0]

# Find best end point for the head (searching from start up to tail_start)
tau_max_head = find_optimal_tau_max_head(lags, avg_corr, std_error, power_law, [0.1, 0.2], tail_start_idx)
mask_head = lags <= tau_max_head
popt_head, pcov_head = curve_fit(power_law, lags[mask_head], avg_corr[mask_head], sigma=std_error[mask_head], absolute_sigma=True)

# --- Plotting ---
fig = plt.figure(figsize=(14, 10))
grid = plt.GridSpec(4, 2, hspace=0.4)

# Main Plot
ax1 = fig.add_subplot(grid[:3, :])

# Plot Data with Error bars
ax1.plot(lags, avg_corr, color='gray', alpha=0.6, linestyle='-', marker='', label='Data')

# Plot Tail Fit
x_fine_tail = np.geomspace(tau_min_tail, max_lag, 200)
ax1.plot(x_fine_tail, power_law(x_fine_tail, *popt_tail), color='crimson', lw=3, label=f'Tail Fit')

# Plot Head Fit
x_fine_head = np.geomspace(1, tau_max_head, 100)
ax1.plot(x_fine_head, power_law(x_fine_head, *popt_head), color='forestgreen', lw=3, 
         label=f'Head Fit')

# Add vertical lines to show regions
ax1.axvline(tau_max_head, color='forestgreen', linestyle='--', alpha=0.5)
ax1.axvline(tau_min_tail, color='crimson', linestyle='--', alpha=0.5)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel('$C(\\tau)$')
ax1.legend()
ax1.grid(True, which="both", ls="-", alpha=0.1)

# Residuals Plot
ax2 = fig.add_subplot(grid[3, :])

ax2.set_xlabel('$\u03C4$ (lag)')

# Tail Residuals
res_tail = (avg_corr[mask_tail] - power_law(lags[mask_tail], *popt_tail)) / std_error[mask_tail]
ax2.scatter(lags[mask_tail], res_tail, color='crimson', s=15, alpha=0.6)

# Head Residuals
res_head = (avg_corr[mask_head] - power_law(lags[mask_head], *popt_head)) / std_error[mask_head]
ax2.scatter(lags[mask_head], res_head, color='forestgreen', s=15, alpha=0.6)

ax2.axhline(0, color='black', lw=1)
ax2.set_xscale('log') # Log scale for residuals x-axis often helps visibility

#plt.tight_layout()

plt.savefig('images\\correlation_trade_sign.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Print Results ---
print("-" * 30)
print(f"HEAD FIT (lags 1 to {tau_max_head}):")
print(f"Gamma: {popt_head[1]:.4f} +- {np.sqrt(pcov_head[1][1]):.4f}")
print(f"Constant: {popt_head[0]:.4f} +- {np.sqrt(pcov_head[0][0]):.4f}")
print("-" * 30)
print(f"TAIL FIT (lags {tau_min_tail} to {max_lag}):")
print(f"Gamma: {popt_tail[1]:.4f} +- {np.sqrt(pcov_tail[1][1]):.4f}")
print(f"Constant: {popt_tail[0]:.4f} +- {np.sqrt(pcov_tail[0][0]):.4f}")
print("-" * 30)