import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.odr import ODR, Model, RealData

# Global Plot Configuration
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11
})

show_q_dist = True  # True: mostra la distribuzione di Q sull'asse destro (come impact_volume_curve.py)

def power_law(x, Y, delta):
    return Y * x**delta

def linear_func(p, x):
    """Modello lineare per ODR (spazio log-log): p = [slope, intercept]."""
    return p[0] * x + p[1]

def robust_power_law_fit(x, y, x_err, y_err, sample_count):
    """
    Fit ODR in spazio log-log sui bin ad alta frequenza, identico a
    impact_volume_curve.py (x_err, y_err sono gli SEM dei bin).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    x_err, y_err = np.asarray(x_err, float), np.asarray(y_err, float)
    counts = np.asarray(sample_count)

    # Solo bin con medie positive, poi bin ad alta frequenza (>50% del massimo)
    pos = (x > 0) & (y > 0)
    x, y, x_err, y_err, counts = x[pos], y[pos], x_err[pos], y_err[pos], counts[pos]
    if len(counts) == 0:
        return None
    core = counts > 0.5 * counts.max()
    xc, yc, xerr_c, yerr_c, cc = x[core], y[core], x_err[core], y_err[core], counts[core]

    if len(xc) <= 2:
        return None

    try:
        x_sem = np.where(cc > 1, xerr_c, 1e-8)
        y_sem = np.where(cc > 1, yerr_c, 1e-8)

        log_x = np.log10(xc)
        log_y = np.log10(yc)

        sigma_log_x = np.maximum(x_sem / (xc * np.log(10)), 1e-6)
        sigma_log_y = np.maximum(y_sem / (yc * np.log(10)), 1e-6)

        model_odr = Model(linear_func)
        data = RealData(log_x, log_y, sx=sigma_log_x, sy=sigma_log_y)
        out = ODR(data, model_odr, beta0=[1.0, 0.0]).run()

        slope, intercept = out.beta
        slope_err, intercept_err = out.sd_beta

        # R^2 in log-log space, come in impact_volume_curve.py
        y_pred = slope * log_x + intercept
        ss_res = np.sum((log_y - y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        Y = 10**intercept
        Y_err = np.log(10) * Y * intercept_err

        # intervallo di x usato nel fit (solo bin ad alta frequenza), per tracciare la retta
        return Y, slope, Y_err, slope_err, r2, (xc.min(), xc.max())
    except Exception:
        return None

def bin_data(df, n_bins=51):
    """Common logic to bin MetaVolume and calculate stats."""
    df = df[df['MetaVolume'] > 0]   # come in impact_volume_curve.py
    if df.empty:
        return None
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
    grouped = grouped[(grouped['x'] > 0) & (grouped['y'] > 0)]
    
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
            Y, delta, Ye, de, r2, _ = fit
            all_results[nb] = {'Y': Y, 'delta': delta, 'Ye': Ye, 'de': de, 'r2': r2}

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

        ax1.set_xscale('log'); ax1.set_yscale('log')
        ax1.grid(True, which="both", ls="-", alpha=0.2)

        # Distribuzione di Q sull'asse destro (lineare), come show_q_dist in impact_volume_curve.py
        if show_q_dist:
            ax2 = ax1.twinx()
            counts, bin_edges = np.histogram(subset['MetaVolume'], bins=bins)
            ax2.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge',
                    color='gray', edgecolor='gray', alpha=0.2)
            ax2.set_yscale('linear')
            ax2.set_ylabel('Frequency', color='black')

        # Tutti i bin come semplici marker (nessuna barra d'errore)
        ax1.plot(grouped['x'], grouped['y'], linestyle='', marker='.', markersize=5,
                 color=colors[nb], label=f'Data n={nb}')

        fit = robust_power_law_fit(grouped['x'], grouped['y'], grouped['x_err'], grouped['y_err'], grouped['count'])
        if fit:
            Y, delta, Ye, de, r2, (fx_min, fx_max) = fit
            xl = np.logspace(np.log10(fx_min), np.log10(fx_max), 100)
            ax1.plot(xl, power_law(xl, Y, delta), linestyle='--', linewidth=1.8, color=colors[nb],
                     label=rf'Fit: ${delta:.3f} \pm {de:.3f}$' + '\n' + rf'($R^2 = {r2:.4f}$)')

        ax1.legend(loc='upper left', fontsize=9, frameon=True)

        # Only show x-label on the bottom row (indices 8 and 9)
        if i >= 8:
            ax1.set_xlabel(r'$Q$')
        else:
            ax1.set_xlabel('')

        # Hide y-labels on the right column for even more space
        if i % 2 != 0:
            ax1.set_ylabel('')
        else:
            ax1.set_ylabel(r'$I$')

    plt.tight_layout()
    plt.savefig(f'images\\{image_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    model = ''
    file_name_con_estensione = '20_uniform.csv'
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
            (range(2, 12), f"{nome_img_stratificato}_1"),
            (range(12, 22), f"{nome_img_stratificato}_2"),
        ]

        for plot_range, img_name in stratified_ranges:
            print(f"Generazione grafico 1 (stratificato): {img_name}.png")
            plot_stratified_impact(df_clean, img_name, plot_range)

        res_df = pd.DataFrame.from_dict(all_results, orient='index').sort_index()
        res_df = res_df[(res_df.index > 1) & (res_df.index <= 21)]

        x_data = res_df.index.values

        # --- Graph 2: Parameter Plots ---
        print(f"Generazione grafico 2 (fit parametri originario): {img_prefix}{function_clean}_nbchild_fit.png")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        # MIGLIORAMENTO: Barre d'errore rese più intense con colori saturi, spessore maggiore e capsize
        ax1.errorbar(x_data, res_df['Y'], yerr=res_df['Ye'], fmt='.', color='blue', 
                     ecolor='blue', elinewidth=1.5, capsize=3, label='Y')
        ax1.set_ylabel('Y')
        ax1.set_xticks(x_data)
        ax1.set_xlim(1.5, max(x_data) + 0.5)
        ax1.set_ylim(0.0, 2.0)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.7)
        #ax1.legend()

        # MIGLIORAMENTO: Barre d'errore rese più intense con colori saturi, spessore maggiore e capsize
        ax2.errorbar(x_data, res_df['delta'], yerr=res_df['de'], fmt='.', color='tomato', 
                     ecolor='tomato', elinewidth=1.5, capsize=3, label=r'$\delta$')
        ax2.set_xlabel('n')
        ax2.set_ylabel(r'$\delta$')
        ax2.set_xticks(x_data)
        ax2.set_xlim(1.5, max(x_data) + 0.5)
        ax2.set_ylim(0.0, 0.5)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.7)
        #ax2.legend()

        plt.tight_layout()
        plt.savefig(f"images/{img_prefix}{function_clean}_nbchild_fit.png")
        plt.close()
        
        print("Operazione completata con successo.")

    except FileNotFoundError:
        print(f"Errore: Non trovo il file CSV al percorso: {path_caricamento}")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")