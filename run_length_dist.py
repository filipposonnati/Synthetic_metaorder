from itertools import groupby
from typing import List, Dict, Union
from os import listdir
import pandas as pd
import os
import matplotlib.pyplot as plt

from corr import generate_gaussian_signs
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
    log_scale: bool = False,
    density: bool = True
):
    """
    Plots one or multiple run-length distributions as raw counts or density.
    """
    if isinstance(distributions, dict):
        distributions = [distributions]
        labels = [labels] if labels else ["Distribution"]
    elif labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(distributions))]
        
    plt.figure(figsize=(10, 6))
    
    for dist, label in zip(distributions, labels):
        sorted_lengths = sorted(dist.keys())
        frequencies = [dist[length] for length in sorted_lengths]
        
        # --- Density Normalization Logic ---
        if density:
            total_runs = sum(frequencies)
            # Avoid division by zero if an empty dict somehow slips through
            if total_runs > 0:
                frequencies = [f / total_runs for f in frequencies]
        # ------------------------------------
        
        plt.plot(sorted_lengths, frequencies, marker='o', linestyle='-', alpha=0.7, label=label)
    
    # Adjust y-axis label based on mode
    y_label = "Probability Density" if density else "Frequency / Count"

    plt.xlabel("Run Length", fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    plt.yscale('log')
    
    if log_scale:
        plt.xscale('log')
        
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if log_scale:
        plt.savefig('images\\run_length_distribution_loglog.png')
    else:
        plt.savefig('images\\run_length_distribution.png')
    
    plt.show()

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

    gaussian_signs = generate_gaussian_signs(10_000_000)
    gaussian_rld = compute_run_length_distribution(gaussian_signs)

    lmf_signs = simulate_lmf(1.5, 10, 10_000_000)
    lmf_rld = compute_run_length_distribution(lmf_signs)

    lmf_signs_lambda, _ = simulate_lmf_lambda(1.5, 0.2, 10_000_000)
    lmf_rld_lambda = compute_run_length_distribution(lmf_signs_lambda)

    plot_run_length_distributions(
        distributions=[rld, gaussian_rld, lmf_rld, lmf_rld_lambda],
        labels=["Real Data", "Gaussian", "LMF", r"LMF $\lambda$"],
        log_scale=True
    )

    plot_run_length_distributions(
        distributions=[rld, gaussian_rld, lmf_rld, lmf_rld_lambda],
        labels=["Real Data", "Gaussian", "LMF", r"LMF $\lambda$"],
        log_scale=False
    )