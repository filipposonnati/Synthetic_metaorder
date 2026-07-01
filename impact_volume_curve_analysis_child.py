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

def linear_func(x, m, q):
    """Helper function for linear trend fitting."""
    return m * x + q

def robust_power_law_fit(x, y, x_err, y_err, sample_count):
    """
    Performs an iterative power-law fit on the 'core' high-density data.
    """
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

def compute_all_fits(df, n_bins=51):
    """Runs the power-law fit for every NbChild value present in the data
    and returns a dict {nb: {'Y':..., 'delta':..., 'Ye':..., 'de':...}}.
    """
    all_nb_values = sorted(df['NbChild'].unique())
    all_results = {}

    for nb in all_nb_values:
        subset = df[df['NbChild'] == nb]
        if len(subset) < 10:
            continue
        res = bin_data(subset, n_bins)
        if res is None:
            continue
        grouped, _ = res
        fit = robust_power_law_fit(grouped['x'], grouped['y'], grouped['x_err'], grouped['y_err'], grouped['count'])
        if fit:
            Y, delta, Ye, de = fit
            all_results[nb] = {'Y': Y, 'delta': delta, 'Ye': Ye, 'de': de}

    return all_results


def plot_stratified_impact(df, image_name, plot_range, n_bins=51):
    """Part 2: 10 subplots (2 columns x 5 rows) stratified by NbChild.

    Plots are restricted to the NbChild values in `plot_range`.
    """
    colors = dict(zip(plot_range, cm.tab10(np.linspace(0, 1, len(plot_range)))))

    fig, axes = plt.subplots(5, 2, figsize=(14, 22))
    axes_flat = axes.flatten()

    for i, nb in enumerate(plot_range):
        ax1 = axes_flat[i]
        subset = df[df['NbChild'] == nb]

        if subset.empty or len(subset) < 40:
            ax1.text(0.5, 0.5, f"NbChild={nb}: Insufficient Data", ha='center')
            continue

        res = bin_data(subset, n_bins)
        if res is None:
            continue
        grouped, bins = res

        ax2 = ax1.twinx()
        ax1.set_xscale('log'); ax1.set_yscale('log')

        # Plotting
        ax2.hist(subset['MetaVolume'], bins=bins, color='lightgrey', alpha=0.6)
        
        # MIGLIORAMENTO: Aggiunto il numero di figli (nb) alla legenda del binned data
        ax1.errorbar(grouped['x'], grouped['y'], xerr=grouped['x_err'], yerr=grouped['y_err'],
                     marker='o', linestyle='', color=colors[nb], markersize=4, label=f'Binned data (NbChild={nb})')

        fit = robust_power_law_fit(grouped['x'], grouped['y'], grouped['x_err'], grouped['y_err'], grouped['count'])
        if fit:
            Y, delta, Ye, de = fit
            xl = np.logspace(np.log10(grouped['x'].min()), np.log10(grouped['x'].max()), 100)
            ax1.plot(xl, power_law(xl, Y, delta), 'k--', linewidth=1.5,
                     label=rf'$Y={Y:.2e}, \delta={delta:.3f} \pm {de:.3f}$')

        ax1.legend(loc='upper left', fontsize=9)

        # Only show x-label on the bottom row (indices 8 and 9)
        if i >= 8:
            ax1.set_xlabel(r'$Q$')
        else:
            ax1.set_xlabel('')

        # Hide y-labels on the right column for even more space
        if i % 2 != 0:
            ax1.set_ylabel('')
        else:
            ax1.set_ylabel(r'$I(Q)$')

    plt.tight_layout()
    plt.savefig(f'images\\{image_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    model = ''
    file_name_con_estensione = '20_power_2.0.csv'
    function_clean = file_name_con_estensione.replace('.csv', '')

    folder_prefix = f"meta_tail_{model}" if model else "meta_tail"
    path_caricamento = f"database\\{folder_prefix}\\meta_{file_name_con_estensione}"
    img_prefix = f"impact_volume_curve_analysis_child/{model + '_' if model else ''}"

    nome_img_stratificato = f"{img_prefix}{function_clean}_nbchild"

    try:
        data = pd.read_csv(path_caricamento)
        df_clean = data[data['NbChild'] > 1].copy()

        # Compute the power-law fit once for every NbChild value present in the data
        all_results = compute_all_fits(df_clean)

        # Three stratified figures, each covering 10 consecutive NbChild values
        stratified_ranges = [
            (range(2, 12), nome_img_stratificato),
            (range(12, 22), f"{nome_img_stratificato}_12_21"),
            (range(22, 32), f"{nome_img_stratificato}_22_31"),
        ]

        for plot_range, img_name in stratified_ranges:
            print(f"Generazione grafico 1 (stratificato): {img_name}.png")
            plot_stratified_impact(df_clean, img_name, plot_range)

        res_df = pd.DataFrame.from_dict(all_results, orient='index').sort_index()
        res_df = res_df[(res_df.index > 1) & (res_df.index <= 31)]

        x_data = res_df.index.values

        # --- Graph 2: Parameter Plots ---
        print(f"Generazione grafico 2 (fit parametri originario): {img_prefix}{function_clean}_nbchild_fit.png")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # MIGLIORAMENTO: Barre d'errore rese più intense con colori saturi, spessore maggiore e capsize
        ax1.errorbar(x_data, res_df['Y'], yerr=res_df['Ye'], fmt='.', color='blue', 
                     ecolor='blue', elinewidth=1.5, capsize=3, label='Y')
        ax1.set_xlabel('Number of Children')
        ax1.set_ylabel('Y')
        ax1.set_xticks(x_data)
        ax1.set_xlim(1.5, max(x_data) + 0.5)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # MIGLIORAMENTO: Barre d'errore rese più intense con colori saturi, spessore maggiore e capsize
        ax2.errorbar(x_data, res_df['delta'], yerr=res_df['de'], fmt='.', color='tomato', 
                     ecolor='tomato', elinewidth=1.5, capsize=3, label=r'$\delta$')
        ax2.set_xlabel('Number of Children')
        ax2.set_ylabel(r'$\delta$')
        ax2.set_xticks(x_data)
        ax2.set_xlim(1.5, max(x_data) + 0.5)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"images/{img_prefix}{function_clean}_nbchild_fit.png")
        plt.close()
        
        print("Operazione completata con successo.")

    except FileNotFoundError:
        print(f"Errore: Non trovo il file CSV al percorso: {path_caricamento}")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")