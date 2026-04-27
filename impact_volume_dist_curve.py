import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit

# Global Plot Configuration
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14
})

def power_law(x, Y, delta):
    return Y * x**delta

def robust_power_law_fit(x, y, x_err, y_err, sample_count):
    """
    Performs an iterative power-law fit on the 'core' high-density data.
    """
    # 1. Identify high-density core
    core_mask = sample_count > (0.5 * sample_count.max())
    xc, yc = x[core_mask], y[core_mask]
    xerr_c, yerr_c = x_err[core_mask], y_err[core_mask]

    if len(xc) < 2:
        return None

    try:
        # Initial fit ignoring errors
        popt, _ = curve_fit(power_law, xc, yc, maxfev=5000)
        Y, delta = popt

        # Iterative fit propagating X and Y errors into effective sigma
        for _ in range(10):
            eff_err = np.sqrt(yerr_c**2 + (Y * delta * xc**(delta - 1) * xerr_c)**2)
            popt, pcov = curve_fit(power_law, xc, yc, sigma=eff_err, absolute_sigma=True, maxfev=5000)
            Y, delta = popt
        
        return Y, delta, np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])
    except:
        return None

def bin_data(df, n_bins=51):
    """Common logic to bin MetaVolume and calculate stats."""
    v_min, v_max = df['MetaVolume'].min(), df['MetaVolume'].max()
    if v_min <= 0 or v_max <= 0 or v_min == v_max:
        return None
    
    bins = np.logspace(np.log10(v_min), np.log10(v_max), n_bins)
    df = df.copy()
    df['bin'] = pd.cut(df['MetaVolume'], bins=bins, include_lowest=True)
    
    grouped = df.groupby('bin', observed=True).agg({
        'MetaVolume': ['mean', 'std'],
        'MetaImpact': ['mean', 'std', 'count']
    }).dropna()
    
    grouped.columns = ['x', 'x_std', 'y', 'y_std', 'count']
    # Calculate Standard Error of the Mean
    grouped['x_err'] = grouped['x_std'] / np.sqrt(grouped['count'])
    grouped['y_err'] = grouped['y_std'] / np.sqrt(grouped['count'])
    
    return grouped, bins

def plot_aggregate_impact(df, image_name, n_bins=51):
    """Part 1: Single aggregate plot."""
    res = bin_data(df, n_bins)
    if res is None: return
    grouped, bins = res
    
    fit = robust_power_law_fit(grouped['x'], grouped['y'], grouped['x_err'], grouped['y_err'], grouped['count'])
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel(r'$Q$'); ax1.set_ylabel(r'$I(Q)$')
    ax2.set_ylabel('Frequency')
    
    # Data and Hist
    ax2.hist(df['MetaVolume'], bins=bins, color='lightgrey', alpha=0.6)
    ax1.errorbar(grouped['x'], grouped['y'], xerr=grouped['x_err'], yerr=grouped['y_err'], 
                 marker='o', linestyle="", color='C0', label="Binned data")
    
    if fit:
        Y, delta, Y_err, delta_err = fit
        x_line = np.logspace(np.log10(grouped['x'].min()), np.log10(grouped['x'].max()), 100)
        ax1.plot(x_line, power_law(x_line, Y, delta), 'k--', label='Fitted curve')
        print(f"Aggregate Fit: Y={Y:.4e}, delta={delta:.4f}")

    ax1.legend()
    plt.tight_layout()
    plt.savefig(f'images\\{image_name}.png')
    plt.show()

def plot_stratified_impact(df, image_name, n_bins=51):
    """Part 2: 10 subplots (2 columns x 5 rows) stratified by NbChild."""
    child_range = range(2, 12)
    colors = dict(zip(child_range, cm.tab10(np.linspace(0, 1, len(child_range)))))
    
    fig, axes = plt.subplots(5, 2, figsize=(14, 22))
    axes_flat = axes.flatten()
    
    for i, nb in enumerate(child_range):
        ax1 = axes_flat[i]
        subset = df[df['NbChild'] == nb]
        
        if subset.empty or len(subset) < 10:
            ax1.text(0.5, 0.5, f"NbChild={nb}: Insufficient Data", ha='center')
            continue
            
        res = bin_data(subset, n_bins)
        if res is None: continue
        grouped, bins = res
        
        ax2 = ax1.twinx()
        ax1.set_xscale('log'); ax1.set_yscale('log')
        #ax1.set_title(rf'$N_c = {nb}$ ($n={len(subset):,}$, bins={len(grouped)})$', fontsize=13)
        
        # Plotting
        ax2.hist(subset['MetaVolume'], bins=bins, color='lightgrey', alpha=0.6)
        ax1.errorbar(grouped['x'], grouped['y'], xerr=grouped['x_err'], yerr=grouped['y_err'],
                     marker='o', linestyle='', color=colors[nb], markersize=4, label='Binned data')
        
        fit = robust_power_law_fit(grouped['x'], grouped['y'], grouped['x_err'], grouped['y_err'], grouped['count'])
        if fit:
            Y, delta, Ye, de = fit
            xl = np.logspace(np.log10(grouped['x'].min()), np.log10(grouped['x'].max()), 100)
            ax1.plot(xl, power_law(xl, Y, delta), 'k--', linewidth=1.5, 
                     label=rf'$Y={Y:.2e}, \delta={delta:.3f} \pm {de:.3f}$')
            
            print(f'{nb} {Y:.3f} +- {Ye:.3f}, {delta:.2f} +- {de:.2f}')
        
        ax1.legend(loc='upper left', fontsize=9)

        # KEY CHANGE: Only show x-label on the bottom row (indices 8 and 9)
        if i >= 8:
            ax1.set_xlabel(r'$Q$')
        else:
            ax1.set_xlabel('')
            
        # Optional: Hide y-labels on the right column for even more space
        if i % 2 != 0:
            ax1.set_ylabel('')
        else:
            ax1.set_ylabel(r'$I(Q)$')

    #fig.suptitle('Market Impact vs. Volume — Stratified by $N_c$', fontsize=20, y=1.01)
    plt.tight_layout()
    plt.savefig(f'images\\{image_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 1. Configurazione nomi
    model = '' 
    file_name_con_estensione = '20_power_2.0.csv'
    # Rimuoviamo .csv per i titoli dei grafici e i nomi dei file immagine
    function_clean = file_name_con_estensione.replace('.csv', '')

    # 2. Path per il CARICAMENTO (deve avere il .csv)
    folder_prefix = f"meta_{model}" if model else "meta"
    path_caricamento = f"database\\{folder_prefix}\\meta_{file_name_con_estensione}"

    # 3. Path per il SALVATAGGIO (regola: prefisso_nbchild_nome)
    img_prefix = f"impact_volume_dist_curve_{model + '_' if model else ''}"
    
    # Nome per il plot aggregato: impact_volume_dist_curve_20_power_2.0.png
    nome_img_aggregato = f"{img_prefix}{function_clean}"
    
    # Nome per il plot stratificato: impact_volume_dist_curve_nbchild_20_power_2.0.png
    nome_img_stratificato = f"{img_prefix}nbchild_{function_clean}"

    # Esecuzione
    try:
        data = pd.read_csv(path_caricamento, parse_dates=['BeginTime', 'EndTime'])
        df_clean = data[data['NbChild'] > 1].copy()

        # Parte 1
        print(f"Generazione grafico aggregato: {nome_img_aggregato}")
        plot_aggregate_impact(df_clean, nome_img_aggregato)

        # Parte 2
        print(f"Generazione grafico stratificato: {nome_img_stratificato}")
        plot_stratified_impact(df_clean, nome_img_stratificato)
        
    except FileNotFoundError:
        print(f"Errore: Non trovo il file CSV al percorso: {path_caricamento}")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")