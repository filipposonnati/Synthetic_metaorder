import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import internal module 'methods' for trader mapping
import methods

# ---------------------------------------------------------------------------
# 1. Main Functions & Fit Model
# ---------------------------------------------------------------------------

def log_exp_func(L, ln_A, lmbda):
    """ Logarithmic exponential model: ln(P(L)) = ln(A) - lambda * L """
    return ln_A - lmbda * L


def exp_func(L, A, lmbda):
    """ Standard exponential model for plotting: P(L) = A * exp(-lambda * L) """
    return A * np.exp(-lmbda * L)


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


def process_one_file(trades: pd.DataFrame, signs: np.ndarray,
                     nb_traders: int, kind: str, alpha: float) -> pd.Series:
    """
    Process trade data to compute meta-order run lengths.
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
# 2. Main Execution
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    
    output_dir = 'images/meta_child_dist'
    os.makedirs(output_dir, exist_ok=True)

    configurations = [
        {'nb_traders': 1,  'kind': 'uniform', 'exponent': 0.0},
        {'nb_traders': 2,  'kind': 'uniform', 'exponent': 0.0},
        {'nb_traders': 2, 'kind': 'power', 'exponent': 2.0}
    ]

    num_events = 10_000_000   
    num_runs   = 5           
    
    rho_configs = [
        #{'type': 'markov_1', 'rho1': 0.0, 'rho2': 0.0, 'label_file': 'markov_0'},
        #{'type': 'markov_1', 'rho1': 0.5, 'rho2': 0.0, 'label_file': 'markov_1'},
        #{'type': 'markov_2', 'rho1': 0.5, 'rho2': 0.0, 'label_file': 'markov_2'},
        {'type': 'markov_2', 'rho1': 0.5, 'rho2': 0.25, 'label_file': 'markov_3'},
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

        for cfg in configurations:
            nb_traders = cfg['nb_traders']
            kind       = cfg['kind']
            exponent   = float(cfg['exponent'])
            label      = f"N={nb_traders}, kind={kind}, exp={exponent}"
            
            print(f"Processing: {label}...")
            lengths_acc = []

            for _ in range(num_runs):
                trades = pd.DataFrame(np.random.rand(num_events, 4))
                
                if sim_type == 'markov_1':
                    signs = generate_correlated_binary_sequence(num_events, rho=r1)
                elif sim_type == 'markov_2':
                    signs = generate_markov_order2_binary_sequence(num_events, r1=r1, r2=r2)
                
                run_lengths = process_one_file(trades, signs, nb_traders, kind, exponent)
                lengths_acc.append(run_lengths.to_numpy())

            results[label] = np.concatenate(lengths_acc)

        # ---------------------------------------------------------------------------
        # 3. Semi-Log Plot with Log-Space curve_fit & Saving
        # ---------------------------------------------------------------------------
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for idx, (label, lengths) in enumerate(results.items()):
            values, counts = np.unique(lengths, return_counts=True)
            pmf = counts / counts.sum()
            color = colors[idx % len(colors)]

            # Scatter plot for distributions
            ax.plot(values, pmf, 'o', linestyle='none', color=color, alpha=0.7, label=label)

            # Fit EACH configuration in log-space using curve_fit
            valid_mask = pmf > 0
            x_data = values[valid_mask]
            y_data_log = np.log(pmf[valid_mask])

            # Initial guesses for ln(A) and lambda
            p_same_theoretical = (1.0 + r1) / 2.0 if r1 > 0 else 0.5
            lambda_theoretical = -np.log(p_same_theoretical)
            p0 = [0.0, lambda_theoretical]  # ln(A) ~ 0 -> A ~ 1

            popt, pcov = curve_fit(log_exp_func, x_data, y_data_log, p0=p0)
            
            ln_A_fit, lambda_fit = popt
            A_fit = np.exp(ln_A_fit)
            
            perr = np.sqrt(np.diag(pcov))
            ln_A_err, lambda_err = perr
            A_err = A_fit * ln_A_err  # Error propagation: delta(A) = A * delta(ln A)

            # Fit curve evaluation
            x_fit = np.linspace(values.min(), values.max(), 200)
            y_fit = exp_func(x_fit, A_fit, lambda_fit)

            # Plot fit curve per configuration matching color
            fit_label = f"Log-Fit Config {idx+1} (λ={lambda_fit:.4f}±{lambda_err:.4f})"
            ax.plot(x_fit, y_fit, linestyle='--', color=color, linewidth=1.8, label=fit_label)

            print(f"\n--- Log-space curve_fit Results for '{label}' ({file_tag}) ---")
            print(f"Model: ln P(L) = ln A - λ * L")
            print(f"A          = {A_fit:.6f} ± {A_err:.6f}")
            print(f"λ          = {lambda_fit:.6f} ± {lambda_err:.6f}")

        # Formatting Semi-Log Plot
        ax.set_yscale('log')
        ax.set_xlabel("Meta-order Length")
        ax.set_ylabel("Probability $P(L)$")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

        plt.tight_layout()
        
        filename = f"series_{file_tag}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        print(f"\nFigure saved to '{filepath}'")