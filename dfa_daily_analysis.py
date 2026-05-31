import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from scipy.stats import linregress

# ══════════════════════════════════════════════════════════════════════════════
# SETUP GRAFICI
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

# ══════════════════════════════════════════════════════════════════════════════
# DFA CON DOPPIO REGIME (FIT BI-LINEARE)
# ══════════════════════════════════════════════════════════════════════════════

def dfa_two_regimes(x, n_crossover=15):
    """
    Esegue la DFA su una singola serie calcolando due esponenti dfa (Alpha)
    separati, divisi dal valore di scala 'n_crossover'.
    """
    N = len(x)
    Y = np.cumsum(x - np.mean(x))
    
    # Range di finestre da scale piccole (6) fino a N // 4
    n_vals = np.unique(np.logspace(np.log10(6), np.log10(N // 4), num=25, dtype=int))
    F_n = []
    
    for n in n_vals:
        num_segments = N // n
        if num_segments == 0:
            F_n.append(np.nan)
            continue
            
        squared_fluct = 0.0
        t = np.arange(n)
        
        # Forward e Backward detrending
        for m in range(num_segments):
            start, end = m * n, m * n + n
            poly = np.polyfit(t, Y[start:end], 1)
            squared_fluct += np.sum((Y[start:end] - np.polyval(poly, t)) ** 2)
            
            start_b, end_b = N - (m + 1) * n, N - (m + 1) * n + n
            poly_b = np.polyfit(t, Y[start_b:end_b], 1)
            squared_fluct += np.sum((Y[start_b:end_b] - np.polyval(poly_b, t)) ** 2)
            
        F_n.append(np.sqrt(squared_fluct / (2 * num_segments * n)))
        
    n_vals, F_n = np.array(n_vals), np.array(F_n)
    mask = (F_n > 0) & ~np.isnan(F_n)
    
    n_vals = n_vals[mask]
    F_n = F_n[mask]
    
    short_mask = n_vals <= n_crossover
    long_mask = n_vals > n_crossover
    
    alpha_short, alpha_long = np.nan, np.nan
    
    if np.sum(short_mask) >= 3:
        slope_s, _, _, _, _ = linregress(np.log10(n_vals[short_mask]), np.log10(F_n[short_mask]))
        alpha_short = slope_s
        
    if np.sum(long_mask) >= 3:
        slope_l, _, _, _, _ = linregress(np.log10(n_vals[long_mask]), np.log10(F_n[long_mask]))
        alpha_long = slope_l
        
    return alpha_short, alpha_long, n_vals, F_n

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    data_dir = os.path.join('database', 'data')
    paths = sorted(listdir(data_dir))
    
    alphas_short = []
    alphas_long = []
    example_data = None
    
    print("\n" + "═"*75)
    print(f"{'FILE (GIORNO)':<25} | {'LUNGH. (N)':<10} | {'α_short (n<=15)':<15} | {'α_long (n>15)':<15}")
    print("═"*75)
    
    for path in paths:
        file_path = os.path.join(data_dir, path)
        trades = pd.read_csv(file_path, header=None)
        signs = trades[3].values.astype(float)
        length = len(signs)
        
        # Se il file è troppo corto per estrarre statisticamente il lungo termine
        if length < 100:
            print(f"{path:<25} | {length:<10} | {'[SALTATO: Serie troppo corta]':<33}")
            continue
            
        a_short, a_long, n_v, fn_v = dfa_two_regimes(signs, n_crossover=15)
        
        if not np.isnan(a_short) and not np.isnan(a_long):
            alphas_short.append(a_short)
            alphas_long.append(a_long)
            
            # Stampa giorno per giorno in formato tabellare pulito
            print(f"{path:<25} | {length:<10} | {a_short:<15.4f} | {a_long:<15.4f}")
            
            if example_data is None and length > 1000:
                example_data = (path, a_short, a_long, n_v, fn_v)
        else:
            print(f"{path:<25} | {length:<10} | {'[ERRORE: Fit non riuscito]':<33}")

    # Controllo di sicurezza prima del calcolo delle medie complessive
    if len(alphas_short) == 0:
        print("\n[ATTENZIONE] Nessun file ha superato i criteri di fit. Verifica i tuoi dati.")
        exit()

    mean_short = np.mean(alphas_short)
    mean_long = np.mean(alphas_long)
    
    # ══════════════════════════════════════════════════════════════════════════
    # GENERAZIONE GRAFICI
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # GRAFICO 1: Crossover su singolo giorno tipo
    if example_data:
        name, a_s, a_l, n_v, fn_v = example_data
        ax = axes[0]
        ax.loglog(n_v, fn_v, 'bo-', label='F(n) Empirico')
        ax.axvline(15, color='purple', linestyle='--', label='Crossover Impostato (n=15)')
        
        ns = n_v[n_v <= 15]
        ax.loglog(ns, (ns**a_s) * (fn_v[0] / (ns[0]**a_s)), 'r--', label=f'Breve termine ($\\alpha$={a_s:.2f})')
        
        nl = n_v[n_v > 15]
        ax.loglog(nl, (nl**a_l) * (fn_v[-1] / (nl[-1]**a_l)), 'g--', label=f'Lungo termine ($\\alpha$={a_l:.2f})')
        
        ax.set_xlabel('Scala n (Dimensione Finestra)')
        ax.set_ylabel('Fluuttuazione F(n)')
        ax.set_title(f'Verifica Crossover DFA ({name})')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # GRAFICO 2: Istogramma complessivo dei due regimi
    ax = axes[1]
    ax.hist(alphas_short, bins=15, alpha=0.5, color='crimson', edgecolor='black', 
            label=f'Breve Termine (Media={mean_short:.2f})')
    ax.hist(alphas_long, bins=15, alpha=0.5, color='forestgreen', edgecolor='black', 
            label=f'Lungo Termine (Media={mean_long:.2f})')
    
    ax.axvline(0.5, color='gray', linestyle=':', label='Rumore Bianco (\\alpha=0.5)')
    ax.set_xlabel('Esponente DFA ($\\alpha$)')
    ax.set_ylabel('Frequenza (Numero di Giorni)')
    ax.set_title('Distribuzione di $\\alpha$ nei Due Regimi')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.join('images', 'acf'), exist_ok=True)
    output_plot = os.path.join('images', 'acf', 'dfa_crossover_analysis.png')
    plt.savefig(output_plot, dpi=300)
    plt.close()
    
    # ══════════════════════════════════════════════════════════════════════════
    # REPORT RIASSUNTIVO FINALE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═"*50)
    print("               REPORT MEDIE GLOBALI              ")
    print("═"*50)
    print(f"Giorni analizzati con successo: {len(alphas_short)}")
    print("-"*50)
    print(f"Regime BREVE TERMINE (n <= 15):")
    print(f"  > Alpha Medio (α_short): {mean_short:.4f}")
    print(f"  > Gamma ACF Equivalente:  {2 - 2*mean_short:.4f}")
    print("-"*50)
    print(f"Regime LUNGO TERMINE (n > 15):")
    print(f"  > Alpha Medio (α_long):  {mean_long:.4f}")
    print(f"  > Gamma ACF Equivalente:  {2 - 2*mean_long:.4f}")
    print("═"*50)
    print(f"Grafico riassuntivo salvato in: {output_plot}\n")