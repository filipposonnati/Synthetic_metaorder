import numpy as np
import matplotlib.pyplot as plt
from series_creation import generate_ffm

# =============================================================================
# DETRENDED FLUCTUATION ANALYSIS (DFA)
# =============================================================================
def compute_dfa(series, min_scale=10, max_scale=None, num_scales=30):
    """
    Calcola la DFA di ordine 1 (detrend lineare).
    Ritorna le scale n e le fluttuazioni F(n).
    """
    N = len(series)
    if max_scale is None:
        max_scale = N // 10 # Regola d'oro: non superare 1/10 della lunghezza totale
        
    # 1. Profilo integrato (cumulativa)
    y = np.cumsum(series - np.mean(series))
    
    # Scale logaritmiche per avere punti equidistanti nel plot log-log
    scales = np.unique(np.geomspace(min_scale, max_scale, num_scales).astype(int))
    F_n = np.zeros(len(scales))
    
    for i, n in enumerate(scales):
        # Suddividiamo in finestre non sovrapposte di lunghezza n
        n_windows = N // n
        windowed_y = y[:n_windows * n].reshape((n_windows, n))
        
        # Asse x locale per il fit
        x = np.arange(n)
        
        rms_sum = 0
        for w in range(n_windows):
            # Fit lineare locale (DFA1)
            p = np.polyfit(x, windowed_y[w], 1)
            trend = np.polyval(p, x)
            
            # Varianza dei residui (fluttuazione rispetto al trend locale)
            rms_sum += np.mean((windowed_y[w] - trend)**2)
            
        # Radice quadrata della media delle varianze
        F_n[i] = np.sqrt(rms_sum / n_windows)
        
    return scales, F_n

# =============================================================================
# 3. TEST E VISUALIZZAZIONE
# =============================================================================
if __name__ == "__main__":
    n_points = 1000000
    gamma_target = 0.3 # Esponente target: C(t) ~ t^-0.3
    
    print(f"Generazione serie binaria (n={n_points}, gamma={gamma_target})...")
    serie_binaria = generate_ffm(n_points, gamma_target)
    
    print("Calcolo DFA in corso...")
    scales, F_n = compute_dfa(serie_binaria, min_scale=10, max_scale=10000, num_scales=40)
    
    # Fit della DFA per trovare l'esponente alpha (F(n) ~ n^alpha)
    # Fittiamo in scala logaritmica: log(F(n)) = alpha * log(n) + c
    log_scales = np.log(scales)
    log_Fn = np.log(F_n)
    alpha, intercept = np.polyfit(log_scales, log_Fn, 1)
    
    # Conversione da alpha a gamma
    gamma_stimato = 2.0 - 2.0 * alpha
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.loglog(scales, F_n, 'bo', markersize=5, alpha=0.7, label='Dati DFA')
    
    # Linea di fit
    fit_line = np.exp(intercept) * (scales ** alpha)
    plt.loglog(scales, fit_line, 'r-', linewidth=2, 
               label=f'Fit lineare ($\\alpha$={alpha:.3f})')
    
    plt.title(f"DFA Analysis - Target $\\gamma$={gamma_target}, Stimato $\\gamma$={gamma_stimato:.3f}")
    plt.xlabel("Scala $n$ (Dimensione finestra)")
    plt.ylabel("Fluttuazione Detrendizzata $F(n)$")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()