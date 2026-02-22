import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import methods
import os
from os import listdir
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp

# --- Modelli di Fit ---
def power_law_dist(x, C, alpha):
    return C * np.power(x, -alpha)

def exp_dist(x, C, lambda_val):
    return C * np.exp(-lambda_val * x)

# --- Funzione per ricerca automatica x_min ---
def find_optimal_xmin(x, y, model_func, p0):
    """Trova il punto di inizio della coda che minimizza la distanza KS per un dato modello."""
    start_idx = max(1, int(len(x) * 0.05))
    end_idx = int(len(x) * 0.7)
    best_xmin = x[start_idx]
    min_ks = np.inf
    
    for i in range(start_idx, end_idx):
        xmin_test = x[i]
        mask = x >= xmin_test
        if len(x[mask]) < 8: continue 
        try:
            popt, _ = curve_fit(model_func, x[mask], y[mask], p0=p0)
            y_model = model_func(x[mask], *popt)
            stat, _ = ks_2samp(y[mask]/np.sum(y[mask]), y_model/np.sum(y_model))
            if stat < min_ks:
                min_ks = stat
                best_xmin = xmin_test
        except:
            continue
    return best_xmin

# --- Configurazione e Directory ---
data_dir = 'database\\data'
iterations = 100
nb_traders = 20
kind = 'uniform'
exponent = 0.0
csv_name = f'database\\dist_meta_child_{iterations}_{nb_traders}_{kind}_{exponent}.csv'

# --- 1. GENERAZIONE O CARICAMENTO DATI ---
if os.path.exists(csv_name):
    print(f"File '{csv_name}' trovato. Caricamento dati...")
    df_results = pd.read_csv(csv_name)
    bin_centers = df_results['NbChild'].values
    mean_density = df_results['Mean_Density'].values
    std_density = df_results['Std_Dev'].values
else:
    print(f"File '{csv_name}' non trovato. Avvio simulazioni...")
    all_densities = []
    paths = np.array(listdir(data_dir))

    for i in range(iterations):
        print(f'Iterazione: {i + 1}/{iterations}')
        meta_tot = pd.DataFrame()
        for path in paths:
            l = len(meta_tot)
            # Utilizzo del tuo metodo 'methods.generate'
            meta, _ = methods.generate(path, nb_traders, kind, exponent, l, data_dir)
            meta_tot = pd.concat([meta_tot, meta])
        
        data = meta_tot['NbChild']
        if i == 0:
            min_v, max_v = int(data.min()), int(data.max())
            bins = np.arange(min_v - 0.5, max_v + 1.5, 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2

        counts, _ = np.histogram(data, bins=bins, density=True)
        all_densities.append(counts)

    all_densities = np.array(all_densities)
    mean_density = np.mean(all_densities, axis=0)
    std_density = np.std(all_densities, axis=0)

    df_results = pd.DataFrame({
        'NbChild': bin_centers,
        'Mean_Density': mean_density,
        'Std_Dev': std_density
    })
    df_results.to_csv(csv_name, index=False)

# --- 2. ANALISI AUTOMATICA DELLE CODE ---
# Power Law
xmin_p = find_optimal_xmin(bin_centers, mean_density, power_law_dist, [mean_density[0], 2.0])
mask_p = bin_centers >= xmin_p
popt_p, pcov_p = curve_fit(power_law_dist, bin_centers[mask_p], mean_density[mask_p], 
                           sigma=std_density[mask_p], absolute_sigma=True)

# Esponenziale
xmin_e = find_optimal_xmin(bin_centers, mean_density, exp_dist, [mean_density[0], 0.5])
mask_e = bin_centers >= xmin_e
popt_e, pcov_e = curve_fit(exp_dist, bin_centers[mask_e], mean_density[mask_e], 
                           sigma=std_density[mask_e], absolute_sigma=True)


fig = plt.figure(figsize=(15, 12))
grid = plt.GridSpec(5, 2, hspace=0.45, wspace=0.3)

# Grafico Principale (Log-Log)
ax_main = fig.add_subplot(grid[:3, :])
ax_main.fill_between(bin_centers, mean_density - std_density, mean_density + std_density, color='gray', alpha=0.15, label='Incertezza ($\sigma$)')
ax_main.scatter(bin_centers, mean_density, color='black', s=15, alpha=0.4)

# Plot Fit Power Law
x_fine_p = np.geomspace(xmin_p, bin_centers.max(), 200)
ax_main.plot(x_fine_p, power_law_dist(x_fine_p, *popt_p), color='tab:red', lw=3, label=f'PL Fit ($\\alpha$={popt_p[1]:.2f}, $x_{{min}}$={xmin_p})')

# Plot Fit Esponenziale
x_fine_e = np.linspace(xmin_e, bin_centers.max(), 200)
ax_main.plot(x_fine_e, exp_dist(x_fine_e, *popt_e), color='tab:blue', lw=3, ls='--', label=f'Exp Fit ($\lambda$={popt_e[1]:.2f}, $x_{{min}}$={xmin_e})')

ax_main.set_xscale('log')
ax_main.set_yscale('log')
ax_main.set_title('Tail Fitting')
ax_main.set_ylabel('Density', fontsize=12)
ax_main.legend(fontsize=11)
ax_main.grid(True, which="both", ls="-", alpha=0.1)

# Residui Power Law
ax_res_p = fig.add_subplot(grid[3:, 0])
res_p = (mean_density[mask_p] - power_law_dist(bin_centers[mask_p], *popt_p)) / std_density[mask_p]
ax_res_p.scatter(bin_centers[mask_p], res_p, color='tab:red', s=25, alpha=0.6, edgecolor='k')
ax_res_p.axhline(0, color='black', lw=1.5)
ax_res_p.set_title('Residuals: Power Law')
#ax_res_p.set_xscale('log')

# Residui Esponenziale
ax_res_e = fig.add_subplot(grid[3:, 1])
res_e = (mean_density[mask_e] - exp_dist(bin_centers[mask_e], *popt_e)) / std_density[mask_e]
ax_res_e.scatter(bin_centers[mask_e], res_e, color='tab:blue', s=25, alpha=0.6, edgecolor='k')
ax_res_e.axhline(0, color='black', lw=1.5)
ax_res_e.set_title('Residuals: exponential')
#ax_res_e.set_xscale('log')

if not os.path.exists('images'): os.makedirs('images')
plt.savefig('images\\dist_meta_child.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"A: {popt_p[0]:.3f} +- {np.sqrt(pcov_p[0][0]):.3f}, Alpha: {popt_p[1]:.3f} +- {np.sqrt(pcov_p[1][1]):.3f}")
print(f"A: {popt_e[0]:.3f} +- {np.sqrt(pcov_e[0][0]):.3f}, Lambda: {popt_e[1]:.3f} +- {np.sqrt(pcov_e[1][1]):.3f}")