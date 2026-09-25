import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import del modulo interno 'methods'
import methods

# ---------------------------------------------------------------------------
# 1. Modelli di Fit, Calcolo ACF e Generazione Sequenza Binaria AR(p)
# ---------------------------------------------------------------------------

def log_exp_func(L, ln_A, lmbda):
    """ Modello esponenziale in spazio logaritmico: ln(P(L)) = ln(A) - lambda * L """
    return ln_A - lmbda * L


def exp_func(L, A, lmbda):
    """ Modello esponenziale standard per il plot: P(L) = A * exp(-lambda * L) """
    return A * np.exp(-lmbda * L)


def log_exp_power_law_func(L, ln_A, gamma, lmbda):
    """
    Modello exponential power law in spazio logaritmico:
    ln(P(L)) = ln(A) - gamma * ln(L) - lambda * L
    """
    return ln_A - gamma * np.log(L) - lmbda * L


def exp_power_law_func(L, A, gamma, lmbda):
    """ Modello exponential power law standard: P(L) = A * (L**(-gamma)) * exp(-lambda * L) """
    return A * (L**(-gamma)) * np.exp(-lmbda * L)


def compute_sample_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Calcola l'autocorrelazione campionaria (ACF) fino al lag specificato
    sfruttando la Trasformata Rapida di Fourier (FFT) per la massima efficienza.
    """
    n = len(x)
    x_centered = x - np.mean(x)
    var = np.var(x)
    
    # Zero-padding a 2*n per evitare l'autocorrelazione circolare
    f = np.fft.fft(x_centered, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:max_lag + 1] / (var * n)
    return acf


def generate_ar_p_binary_sequence(size: int, target_rhos: list[float]) -> np.ndarray:
    """
    Genera una sequenza binaria {-1, +1} avente correlazioni target ai lag 1..p.
    Usa il teorema di Van Vleck per convertire le correlazioni binarie in gaussiane.
    """
    p = len(target_rhos)
    
    # 1. Conversione Van Vleck (binario -> gaussiano)
    rhos_g = np.sin(np.pi / 2.0 * np.array(target_rhos))
    
    # 2. Matrice di covarianza Toeplitz e risoluzione equazioni di Yule-Walker
    R = np.eye(p)
    for i in range(p):
        for j in range(p):
            if i != j:
                lag = abs(i - j)
                R[i, j] = rhos_g[lag - 1]
    
    phi = np.linalg.solve(R, rhos_g)
    
    # 3. Variazione dell'innovazione per varianza unitaria
    var_e = 1.0 - np.dot(phi, rhos_g)
    if var_e <= 0:
        raise ValueError("La struttura di correlazione non è stazionaria / definita positiva.")
    std_e = np.sqrt(var_e)
    
    # 4. Generazione processo Gaussiano AR(p)
    burn_in = 1000
    x = np.zeros(size + burn_in)
    e = np.random.normal(0, std_e, size + burn_in)
    
    for t in range(p, len(x)):
        x[t] = np.dot(phi, x[t-p:t][::-1]) + e[t]
        
    x = x[burn_in:]  # Scarto burn-in
    
    # 5. Binarizzazione tramite funzione segno
    return np.where(x >= 0, 1, -1)


def process_one_file(trades: pd.DataFrame, signs: np.ndarray, nb_traders: int, kind: str, alpha: float) -> pd.Series:
    """
    Elabora i dati dei trade e la sequenza dei segni per calcolare le lunghezze dei meta-ordini.
    """
    traders = methods.mapping_function(trades, nb_traders, kind, alpha)
    db = pd.DataFrame({'sign': signs, 'trader': traders})
    sorted_trades = db.sort_values(['trader']).reset_index(drop=True)

    sorted_trades['metaid'] = np.where(
        (sorted_trades['trader'] != sorted_trades['trader'].shift()) |
        (sorted_trades['sign'].shift() != sorted_trades['sign']),
        1, 0
    ).cumsum()

    return sorted_trades.groupby('metaid')['trader'].count()


# ---------------------------------------------------------------------------
# 2. Esecuzione e Simulazione
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    
    # Creazione cartella di output
    output_dir = 'images/meta_child_dist'
    os.makedirs(output_dir, exist_ok=True)

    # Caricamento della funzione di autocorrelazione target (ACF)
    acf_file = 'database/acf_binary.npy'
    if os.path.exists(acf_file):
        acf_data = np.load(acf_file)
    elif os.path.exists('database/acf.npy'):
        acf_data = np.load('database/acf.npy')
    else:
        raise FileNotFoundError("Impossibile trovare il file acf_binary.npy o database/acf.npy")

    # Estrazione dei lag target (es. p = 1000 lag)
    p_lags = 1000
    target_rhos = acf_data[1:p_lags + 1].tolist()

    # Configurazioni dei trader
    configurations = [
        {'nb_traders': 1,  'kind': 'uniform', 'exponent': 0.0},
        {'nb_traders': 4,  'kind': 'uniform', 'exponent': 0.0},
        {'nb_traders': 20, 'kind': 'uniform', 'exponent': 0.0}
    ]

    # Parametri di simulazione
    num_events = 2_000_000   # Eventi per run
    num_runs   = 5           # Iterazioni

    print("==================================================")
    print(f" Running Simulation with ACF AR({p_lags}) Process")
    print("==================================================")

    results = {}
    reconstructed_acfs = []

    for cfg in configurations:
        nb_traders = cfg['nb_traders']
        kind       = cfg['kind']
        exponent   = float(cfg['exponent'])
        label      = f"N={nb_traders}, kind={kind}, exp={exponent}"
        
        print(f"Processing: {label}...")
        lengths_acc = []

        for _ in range(num_runs):
            # Simulazione DataFrame trades sintetico
            trades = pd.DataFrame(np.random.rand(num_events, 4))
            
            # Generazione sequenza binaria con autocorrelazione reale
            signs = generate_ar_p_binary_sequence(num_events, target_rhos=target_rhos)
            
            # Calcolo dell'ACF ricostruita dalla serie binaria generata
            sample_acf = compute_sample_acf(signs, max_lag=p_lags)
            reconstructed_acfs.append(sample_acf[1:])  # Lag 1..p_lags
            
            # Calcolo run lengths
            run_lengths = process_one_file(trades, signs, nb_traders, kind, exponent)
            lengths_acc.append(run_lengths.to_numpy())

        results[label] = np.concatenate(lengths_acc)

    # Calcolo dell'ACF ricostruita media su tutte le iterazioni
    mean_reconstructed_acf = np.mean(reconstructed_acfs, axis=0)
    lags = np.arange(1, p_lags + 1)

    # ---------------------------------------------------------------------------
    # 3. FIGURA 1: Distribuzione Meta-Ordini + Residui + Chi2 Ridotto
    # ---------------------------------------------------------------------------
    
    fig1, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
    )
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    r1 = target_rhos[0]
    p_same_approx = (1.0 + r1) / 2.0
    lambda_init = -np.log(p_same_approx)

    for idx, (label, lengths) in enumerate(results.items()):
        values, counts = np.unique(lengths, return_counts=True)
        pmf = counts / counts.sum()
        color = colors[idx % len(colors)]

        # Plot dati originali
        ax_main.plot(values, pmf, 'o', linestyle='none', color=color, alpha=0.5, label=f"Data: {label}")

        # Maschera dati validi per lo spazio logaritmico
        valid_mask = pmf > 0
        x_data = values[valid_mask]
        y_data_log = np.log(pmf[valid_mask])
        counts_data = counts[valid_mask]

        # Stima dell'errore Poissoniano sui dati logaritmici: sigma_ln(y) = 1 / sqrt(N_counts)
        sigma_y_log = 1.0 / np.sqrt(counts_data)

        x_fit = np.linspace(values.min(), values.max(), 300)

        # --- 1. Fit Esponenziale Standard ---
        p0_exp = [0.0, lambda_init]
        popt_exp, pcov_exp = curve_fit(
            log_exp_func, x_data, y_data_log, p0=p0_exp, sigma=sigma_y_log, absolute_sigma=True
        )
        ln_A_exp, lambda_exp = popt_exp
        A_exp = np.exp(ln_A_exp)

        # Calcolo Residui e Chi-Quadrato Ridotto (Esponenziale)
        y_exp_pred_log = log_exp_func(x_data, *popt_exp)
        res_exp_log = y_data_log - y_exp_pred_log
        dof_exp = len(x_data) - len(p0_exp)
        chi2_red_exp = np.sum((res_exp_log / sigma_y_log) ** 2) / dof_exp

        y_fit_exp = exp_func(x_fit, A_exp, lambda_exp)
        ax_main.plot(
            x_fit, y_fit_exp, linestyle='--', color=color, linewidth=1.5,
            label=f"Exp Config {idx+1} (λ={lambda_exp:.3f}, $\\chi^2_{{red}}$={chi2_red_exp:.2f})"
        )

        # --- 2. Fit Exponential Power Law ---
        p0_power = [0.0, 1.0, lambda_init]
        try:
            popt_pow, pcov_pow = curve_fit(
                log_exp_power_law_func, x_data, y_data_log, p0=p0_power, sigma=sigma_y_log, absolute_sigma=True
            )
            ln_A_pow, gamma_pow, lambda_pow = popt_pow
            A_pow = np.exp(ln_A_pow)

            # Calcolo Residui e Chi-Quadrato Ridotto (Exp Power Law)
            y_pow_pred_log = log_exp_power_law_func(x_data, *popt_pow)
            res_pow_log = y_data_log - y_pow_pred_log
            dof_pow = len(x_data) - len(p0_power)
            chi2_red_pow = np.sum((res_pow_log / sigma_y_log) ** 2) / dof_pow

            y_fit_pow = exp_power_law_func(x_fit, A_pow, gamma_pow, lambda_pow)
            ax_main.plot(
                x_fit, y_fit_pow, linestyle=':', color=color, linewidth=2.2,
                label=f"Exp Power Config {idx+1} (γ={gamma_pow:.2f}, λ={lambda_pow:.3f}, $\\chi^2_{{red}}$={chi2_red_pow:.2f})"
            )

            # Plot Residui nel sottografetto
            ax_res.plot(x_data, res_exp_log, linestyle='--', marker='s', markersize=3, color=color, alpha=0.5)
            ax_res.plot(x_data, res_pow_log, linestyle=':', marker='o', markersize=3, color=color, alpha=0.8)

            print(f"\n--- Fit Results for '{label}' ---")
            print(f"Esponenziale:      Chi2_red = {chi2_red_exp:.4f}")
            print(f"Exp Power Law:    Chi2_red = {chi2_red_pow:.4f}")
            if chi2_red_pow < chi2_red_exp:
                print("--> Il modello 'Exp Power Law' fornisce un fit migliore.")
            else:
                print("--> Il modello 'Esponenziale' fornisce un fit migliore.")

        except Exception as e:
            print(f"Fit Exp Power Law fallito per {label}: {e}")

    # Formattazione Pannello Principale
    ax_main.set_yscale('log')
    ax_main.set_ylabel("Probability $P(L)$")
    #ax_main.set_title("Meta-order Length Distribution & Fits Comparison")
    ax_main.grid(True, which="both", ls="--", alpha=0.5)
    ax_main.legend(fontsize='small', loc='best')

    # Formattazione Pannello Residui
    ax_res.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.7)
    ax_res.set_xlabel("Meta-order Length")
    ax_res.set_ylabel("Residuals\n$[\\ln P_{data} - \\ln P_{fit}]$")
    ax_res.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "series_ar.png")
    fig1.savefig(fig1_path, dpi=300)
    print(f"\nFigure 1 saved to '{fig1_path}'")

    # ---------------------------------------------------------------------------
    # 4. FIGURA 2: Confronto ACF Reale vs Ricostruita
    # ---------------------------------------------------------------------------
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    ax2.plot(lags, target_rhos, label="ACF Reale (Target)", color='black', linewidth=2.0)
    ax2.plot(lags, mean_reconstructed_acf, label="ACF Ricostruita (Simulata)", color='crimson', linestyle='--', linewidth=1.8)
    
    ax2.set_xlabel("Lag $k$")
    ax2.set_ylabel("Autocorrelation $\\rho(k)$")
    ax2.set_title("ACF Reale vs ACF Ricostruita")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "series_ar_acf_comparison.png")
    fig2.savefig(fig2_path, dpi=300)
    print(f"Figure 2 saved to '{fig2_path}'")