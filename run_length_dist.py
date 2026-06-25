from itertools import groupby
from typing import List, Dict, Union
from os import listdir
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from series_gaussian import generate_binary_sequence
from lmf import simulate_lmf, simulate_lmf_lambda

def compute_run_length_distribution(sequence: List[int]) -> Dict[int, int]:
    """
    Computes the run-length distribution of a binary sequence.
    
    Parameters:
    sequence (list): A list of binary elements (0s and 1s).
    
    Returns:
    dict: A dictionary where keys are run lengths and values are their frequencies.
    """
    if len(sequence) == 0:
        return {}
        
    distribution = {}
    
    # groupby groups consecutive identical elements together
    for key, group in groupby(sequence):
        run_length = len(list(group))
        
        # Increment the count for this specific run length
        distribution[run_length] = distribution.get(run_length, 0) + 1
        
    return dict(sorted(distribution.items()))

def plot_run_length_distributions(
    distributions: Union[Dict[int, int], List[Dict[int, int]]], 
    labels: Union[str, List[str]] = None,
    log_scale: Union[bool, str] = 'both',
    density: bool = True
):
    """
    Plots one or multiple run-length distributions as raw counts or density
    with an aligned log-ratio subplot underneath comparing all models to the first dataset.
    
    Supports log_scale=True (log-log), log_scale=False (semi-log y), or log_scale='both'.
    """
    if isinstance(distributions, dict):
        distributions = [distributions]
        labels = [labels] if labels else ["Distribution"]
    elif labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(distributions))]
        
    # --- Pre-process distributions once to compute frequencies/densities ---
    processed_data = []
    for dist in distributions:
        sorted_lengths = sorted(dist.keys())
        frequencies = [dist[length] for length in sorted_lengths]
        
        if density:
            total_runs = sum(frequencies)
            if total_runs > 0:
                frequencies = [f / total_runs for f in frequencies]
                
        processed_data.append((sorted_lengths, frequencies))
        
    # Use the first dataset (e.g., "Real Data") as the reference
    ref_lengths, ref_frequencies = processed_data[0]
    ref_lookup = dict(zip(ref_lengths, ref_frequencies))
    
    # Determine which scale configurations to iterate through
    scales_to_plot = [False, True] if log_scale == 'both' else [log_scale]
    
    for current_log_scale in scales_to_plot:
        # Create a 2-row layout: top for main distribution, bottom for log-ratios
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, 
            figsize=(10, 8), 
            sharex=True, 
            gridspec_kw={'height_ratios': [3, 1]}
        )
        
        # Draw a horizontal baseline at 0 for reference (log10(1) = 0 means Model == Real)
        ax_res.axhline(0, color='gray', linestyle='--', alpha=0.6)
        
        for i, (label, (sorted_lengths, frequencies)) in enumerate(zip(labels, processed_data)):
            # Plot main distribution
            ax_main.plot(sorted_lengths, frequencies, marker='o', linestyle='-', alpha=0.7, label=label)
            
            # Compute and plot log-ratio for all datasets compared to Real Data (index > 0)
            if i > 0:
                ratio_lengths = []
                log_ratios = []
                for length, freq in zip(sorted_lengths, frequencies):
                    ref_freq = ref_lookup.get(length, 0.0)
                    # Protect against division by zero and log of zero
                    if ref_freq > 0 and freq > 0:  
                        ratio_lengths.append(length)
                        log_ratios.append(np.log10(freq / ref_freq))
                
                # Fetch the color of the line from the main axes to match perfectly
                line_color = ax_main.lines[-1].get_color()
                ax_res.plot(ratio_lengths, log_ratios, marker='o', linestyle='-', alpha=0.7, color=line_color)
        
        # Formatting Top Main Plot
        y_label = "Probability Density" if density else "Frequency / Count"
        ax_main.set_ylabel(y_label, fontsize=12)
        ax_main.set_yscale('log')
        ax_main.grid(True, which="both", linestyle="--", alpha=0.5)
        ax_main.legend(fontsize=10)
        
        # Formatting Bottom Log-Ratio Plot
        ax_res.set_xlabel("Run Length", fontsize=12)
        ax_res.set_ylabel("Log Ratio\nlog10(Model / Real)", fontsize=10)
        ax_res.grid(True, which="both", linestyle="--", alpha=0.5)
        
        # Adjust X scale globally (propagates automatically because sharex=True)
        if current_log_scale:
            ax_main.set_xscale('log')
            
        fig.tight_layout()

        # Save to cross-platform paths based on the current scale mode
        if current_log_scale:
            fig.savefig(os.path.join('images', 'run_length_distribution_loglog.png'))
        else:
            fig.savefig(os.path.join('images', 'run_length_distribution.png'))
            
        plt.close(fig)

def read_data():
    data_dir = os.path.join('database', 'data')
    paths    = listdir(data_dir)

    # Master dictionary to aggregate all run length counts across all files
    rld_total = {}

    for path in paths:
        # Ensure we only read CSV files and ignore system files (like .DS_Store)
        if not path.endswith('.csv'):
            continue
            
        full_path = os.path.join(data_dir, path)
        trades = pd.read_csv(full_path, header=None)
        
        # 1. Cast as int instead of float for clean binary/discrete matching
        signs = trades[3].values.astype(int) 
        
        # 2. Compute RLD for the current file
        rld = compute_run_length_distribution(signs)
        
        # 3. Merge this file's distribution into the master rld_total dictionary
        for length, count in rld.items():
            rld_total[length] = rld_total.get(length, 0) + count

    # Sorted print of the combined distribution after processing ALL files
    sorted_rld_total = dict(sorted(rld_total.items()))

    return sorted_rld_total


if __name__ == "__main__":
    rld = read_data()

    pooled     = np.load('database/acf_binary.npy')
    p_plus     = float(np.load('database/p_plus.npy'))
    median_len = int(np.load('database/median_len.npy'))

    #gaussian_signs, _ = generate_binary_sequence(pooled, p_plus=p_plus, N=10_000_000, n_realizations=1, seed=42)
    #gaussian_rld = compute_run_length_distribution(gaussian_signs[0])

    lmf_signs = simulate_lmf(1.5, 4, 10_000_000)
    lmf_rld_4 = compute_run_length_distribution(lmf_signs)

    lmf_signs_lambda, _ = simulate_lmf_lambda(1.8, 0.2, 10_000_000)
    lmf_rld_lambda_18_02 = compute_run_length_distribution(lmf_signs_lambda)

    lmf_signs_lambda, _ = simulate_lmf_lambda(1.6, 0.3, 10_000_000)
    lmf_rld_lambda_16_03 = compute_run_length_distribution(lmf_signs_lambda)

    lmf_signs_lambda, _ = simulate_lmf_lambda(1.8, 0.3, 10_000_000)
    lmf_rld_lambda_18_03 = compute_run_length_distribution(lmf_signs_lambda)

    # Calling the method once handles plotting and saving both configurations with aligned ratio graphs
    plot_run_length_distributions(
        distributions=[rld, lmf_rld_lambda_18_02, lmf_rld_lambda_16_03, lmf_rld_lambda_18_03],
        labels=["Real Data", r"LMF $\lambda$ = 0.2 $\alpha$ = 1.8", r"LMF $\lambda$ = 0.3 $\alpha$ = 1.6", r"LMF $\lambda$ = 0.3 $\alpha$ = 1.8"],
        log_scale='both'
    )