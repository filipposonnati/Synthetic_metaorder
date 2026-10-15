import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw

# Import internal module 'methods' for trader mapping
import methods

# ---------------------------------------------------------------------------
# 1. Main Functions & Fit Model
# ---------------------------------------------------------------------------

def generate_correlated_binary_sequence(size: int, rho: float = 0.0) -> np.ndarray:
    """
    Generate a binary sequence {-1, +1} (Markov Chain order 1) 
    with lag-1 autocorrelation equal to `rho`.
    """
    if rho == 0.0:
        return np.random.choice([-1, 1], size=size, p=[0.5, 0.5])
    
    p_same = (1.0 + rho) / 2.0
    
    first_sign = np.random.choice([-1, 1])
    same_mask = np.random.rand(size - 1) < p_same
    flips = np.where(same_mask, 1, -1)
    
    signs = np.empty(size, dtype=np.int8)
    signs[0] = first_sign
    signs[1:] = first_sign * np.cumprod(flips)
    
    return signs


def generate_markov_order2_binary_sequence(size: int, r1: float = 0.5, r2: float = 0.3) -> np.ndarray:
    """
    Generate a binary sequence {-1, +1} (Markov Chain order 2)
    with fixed lag-1 autocorrelation r1 and lag-2 autocorrelation r2.
    """
    A = np.array([[1.0, r1], [r1, 1.0]])
    b = np.array([r1, r2])
    phi1, phi2 = np.linalg.solve(A, b)
    
    a = phi1 / 2.0
    b_coef = phi2 / 2.0
    
    signs = np.empty(size, dtype=np.int8)
    signs[0] = np.random.choice([-1, 1])
    signs[1] = np.random.choice([-1, 1])
    
    rand_vals = np.random.rand(size)
    for t in range(2, size):
        p_plus = 0.5 + a * signs[t-1] + b_coef * signs[t-2]
        signs[t] = 1 if rand_vals[t] < p_plus else -1
        
    return signs


def process_one_file(signs: np.ndarray,
                     nb_traders: int, kind: str, alpha: float) -> pd.Series:
    """
    Process trade data to compute meta-order run lengths.
    """
    trades = pd.DataFrame(np.random.rand(len(signs), 4))

    traders = methods.mapping_function(trades, nb_traders, kind, alpha)

    db = pd.DataFrame({'sign': signs, 'trader': traders})

    sorted_trades = db.sort_values(['trader'], kind='stable').reset_index(drop=True)

    sorted_trades['metaid'] = np.where(
        (sorted_trades['trader'] != sorted_trades['trader'].shift()) |
        (sorted_trades['sign'].shift() != sorted_trades['sign']),
        1, 0
    ).cumsum()

    return sorted_trades.groupby('metaid')['trader'].count()


# ---------------------------------------------------------------------------
# 2. Main Execution
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    
    output_dir = 'images/meta_child_dist'
    os.makedirs(output_dir, exist_ok=True)

    # Configurazioni aggiornate con l'aggiunta del coefficiente teorico 'lambda_theo'
    configurations = [
        {'nb_traders': 1,  'kind': 'uniform', 'exponent': 0.0, 'lambda_theo': -np.log(2/3)},
        {'nb_traders': 2,  'kind': 'uniform', 'exponent': 0.0, 'lambda_theo': np.log(1.5)},
        #{'nb_traders': 2,  'kind': 'power',   'exponent': 2.0, 'lambda_theo': 0.3}
    ]

    num_events = 1_000_000      
    
    rho_configs = [
        #{'type': 'markov_1', 'rho1': 0.0, 'rho2': 0.0, 'label_file': 'markov_0'},
        {'type': 'markov_1', 'rho1': 0.5, 'rho2': 0.0, 'label_file': 'markov_1'},
        #{'type': 'markov_2', 'rho1': 0.5, 'rho2': 0.8, 'label_file': 'markov_2'}
    ]

    for cfg_rho in rho_configs:
        r1 = cfg_rho['rho1']
        r2 = cfg_rho['rho2']
        sim_type = cfg_rho['type']
        file_tag = cfg_rho['label_file']

        print(f"\n==================================================")
        print(f" Running Simulation for {file_tag} (type: {sim_type})")
        print(f"==================================================")

        results = {}
        config_meta = {} # Dizionario per mappare l'etichetta alla sua configurazione teorica

        for cfg in configurations:
            nb_traders = cfg['nb_traders']
            kind       = cfg['kind']
            exponent   = float(cfg['exponent'])
            label      = f"N={nb_traders}, kind={kind}, exp={exponent}"
            
            config_meta[label] = cfg
            
            print(f"Processing: {label}...")
            lengths_acc = []
            
            if sim_type == 'markov_1':
                signs = generate_correlated_binary_sequence(num_events, rho=r1)
            elif sim_type == 'markov_2':
                signs = generate_markov_order2_binary_sequence(num_events, r1=r1, r2=r2)

            #print(signs)
            
            run_lengths = process_one_file(signs, nb_traders, kind, exponent)
            #print(f"Lunghezza media empirica: {np.mean(run_lengths):.4f} (attesa: 4.0)")

            #print(run_lengths)
            
            lengths_acc.append(run_lengths.to_numpy())

            results[label] = np.concatenate(lengths_acc)

            #exit()

        # ---------------------------------------------------------------------------
        # 3. Semi-Log Plot: Dati empirici, Fit e Curva Teorica
        # ---------------------------------------------------------------------------
        
        XMIN_MANUALE = 1

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for idx, (label, lengths) in enumerate(results.items()):
            values, counts = np.unique(lengths, return_counts=True)
            pmf = counts / counts.sum()
            color = colors[idx % len(colors)]

            # A. Scatter plot dei dati empirici
            ax.plot(values, pmf, 'o', linestyle='none', color=color, alpha=0.7, label=f"Emp. {label}")

            # B. Fit esponenziale con xmin manuale tramite powerlaw
            fit = powerlaw.Fit(lengths, discrete=True, xmin=XMIN_MANUALE, verbose=False)
            lambda_fit = fit.exponential.parameter1
            
            # B1. Calcolo analitico alternativo di lambda_hat basato sulla media della coda (MLE geometrico)
            tail_lengths = lengths[lengths >= XMIN_MANUALE]
            if len(tail_lengths) > 0:
                mean_tail = np.mean(tail_lengths - XMIN_MANUALE)
                # Per una geometrica P(L) ~ exp(-lambda * L), la stima MLE è: lambda_hat = -log( mean / (mean + 1) )
                # Oppure basata direttamente sulla relazione p_same: p_hat = mean / (mean + 1), lambda_hat = -log(p_hat)
                p_hat = mean_tail / (mean_tail + 1.0)
                lambda_hat = -np.log(p_hat)
            else:
                lambda_hat = np.nan

            n_samples_tail = len(tail_lengths)
            lambda_err = lambda_fit / np.sqrt(n_samples_tail) if n_samples_tail > 0 else 0.0
            
            x_fit = np.linspace(XMIN_MANUALE, values.max(), 200)
            
            idx_xmin = np.where(values == XMIN_MANUALE)[0]
            p_xmin = pmf[idx_xmin[0]] if len(idx_xmin) > 0 else pmf[0]
            y_fit = p_xmin * np.exp(-lambda_fit * (x_fit - XMIN_MANUALE))

            # C. Plot della curva di fit (linea tratteggiata)
            fit_label = f"Fit {idx+1} ($\\lambda_{{fit}}$={lambda_fit:.3f})"
            ax.plot(x_fit, y_fit, linestyle='--', color=color, linewidth=1.5, label=fit_label)

            idx_xmin_theo = np.where(values == 1)[0]
            p_xmin_theo = pmf[idx_xmin_theo[0]] if len(idx_xmin_theo) > 0 else pmf[0]
            x_theo = np.linspace(1, values.max(), 200)

            # D. Plot della curva teorica (linea continua) basata su 'lambda_theo' in configurazione
            cfg_current = config_meta[label]
            if 'lambda_theo' in cfg_current:
                lambda_theo = cfg_current['lambda_theo']
                y_theo = p_xmin_theo * np.exp(-lambda_theo * (x_theo - 1))
                theo_label = f"Theo {idx+1} ($\\lambda_{{theo}}$={lambda_theo:.3f})"
                ax.plot(x_theo, y_theo, linestyle='-', color=color, linewidth=2.0, alpha=0.8, label=theo_label)

            print(f"\n--- Results for '{label}' ({file_tag}) ---")
            print(f"λ (fit)    : {lambda_fit:.4f} ± {lambda_err:.4f}")
            print(f"λ (hat)    : {lambda_hat:.4f} (MLE geometrico su coda >= {XMIN_MANUALE})")
            if 'lambda_theo' in cfg_current:
                print(f"λ (theo)   : {cfg_current['lambda_theo']:.4f}")

        # Formatting Semi-Log Plot
        ax.set_yscale('log')
        ax.set_xlabel("Meta-order Length")
        ax.set_ylabel("Probability $P(L)$")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8, loc='upper right')

        plt.tight_layout()
        
        filename = f"series_{file_tag}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        print(f"\nFigure saved to '{filepath}'")