import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from scipy.optimize import curve_fit
import re
import statsmodels.api as sm
from statsmodels.nonparametric.kernel_regression import KernelReg

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Theoretical Scaling Baseline
def square_root_law(x, a):
    return a * np.sqrt(x)

# 2. Neural Network Operating on Log-Log Space
class ImpactCurveNet(nn.Module):
    def __init__(self, hidden: int = 32, depth: int = 2):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def fit_nn_on_binned_data(X_binned, Y_binned, n_eval=500, epochs=200):
    """
    Trains the MLP near-instantaneously by optimizing directly over 
    the clean binned summary coordinates rather than the raw noisy scatter.
    """
    log_q = np.log10(X_binned).astype(np.float32)
    log_i = np.log10(Y_binned).astype(np.float32)

    # Standardize arrays to guarantee numeric stability in the network layers
    mu_q, sig_q = log_q.mean(), log_q.std() + 1e-8
    mu_i, sig_i = log_i.mean(), log_i.std() + 1e-8
    log_q_n = (log_q - mu_q) / sig_q
    log_i_n = (log_i - mu_i) / sig_i

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImpactCurveNet(hidden=32, depth=2).to(device)

    X_tensor = torch.from_numpy(log_q_n[:, None])
    Y_tensor = torch.from_numpy(log_i_n[:, None])
    
    # Because there are only ~300 binned coordinates, we pass them all in a single batch
    loader = DataLoader(TensorDataset(X_tensor, Y_tensor), batch_size=512, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    loss_fn = nn.MSELoss()  # MSE is highly precise on already aggregated binned values

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

    # Generate visual evaluation curve over a flawless grid
    model.eval()
    q_min, q_max = log_q.min(), log_q.max()
    with torch.no_grad():
        eval_grid_n = torch.linspace((q_min - mu_q)/sig_q, (q_max - mu_q)/sig_q, n_eval, dtype=torch.float32).unsqueeze(1).to(device)
        log_i_pred_n = model(eval_grid_n).squeeze(1).cpu().numpy()

    x_eval = 10 ** np.linspace(q_min, q_max, n_eval)
    y_eval = 10 ** (log_i_pred_n * sig_i + mu_i)
    return x_eval, y_eval

# 3. Visual Layout Configuration
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 20, 'axes.labelsize': 16,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 13
})

def get_marker(kind):
    return '.' if kind == 'power' else ('x' if kind == 'uniform' else 's')

dir_path = 'database\\meta'
target_file = "meta_20_power_2.0.csv"

fig, ax1 = plt.subplots(figsize=(9, 6))

pattern = r"meta_(?:(?P<num_one>1)|(?P<num_others>\d+)_(?P<kind>\w+?)(?:_(?P<exp>[\d.]+))?)\.csv"
match = re.search(pattern, target_file)

if match:
    num_traders = match.group('num_one') or match.group('num_others')
    kind = match.group('kind') if match.group('kind') else ""
    exp = match.group('exp') if match.group('exp') else ""
    label = " ".join(filter(None, [str(num_traders), kind, exp]))

    image_name = label + '_smooth'

    try:
        synthetic_meta = pd.read_csv(f'{dir_path}\\{target_file}', sep=',', parse_dates=['BeginTime', 'EndTime'])
    except FileNotFoundError:
        print(f"File {target_file} not found inside {dir_path}.")
        exit()

    synthetic_meta['NbChild'] = pd.to_numeric(synthetic_meta['NbChild'], errors='coerce')
    df_res = synthetic_meta[synthetic_meta['NbChild'] > 1].dropna(subset=['MetaVolume', 'MetaImpact']).copy()

    if len(df_res) > 0:
        min_vol, max_vol = df_res['MetaVolume'].min(), df_res['MetaVolume'].max()

        # ---- BACKGROUND LAYER: METAORDER VOLUME DISTRIBUTION ----
        ax2 = ax1.twinx()
        dist_bins = np.logspace(np.log10(min_vol), np.log10(max_vol), 51)
        ax2.hist(df_res['MetaVolume'], bins=dist_bins, color='gray', alpha=0.20, label='Metaorder Distribution')
        ax2.set_ylabel('Relative Frequency')
        ax2.tick_params(axis='y')

        # ---- DATA PROCESSING: QUANTILE MICRO-BINNING ----
        num_bins = 300
        df_res['bin'] = pd.qcut(df_res['MetaVolume'], q=num_bins, labels=False, duplicates='drop')
        grouped = df_res.groupby('bin', observed=True).agg({'MetaVolume': 'mean', 'MetaImpact': 'mean'}).dropna()
        X_binned, Y_binned = grouped['MetaVolume'].to_numpy(), grouped['MetaImpact'].to_numpy()
        log_X_binned = np.log10(X_binned)

        # Plot empirical background markers
        ax1.scatter(X_binned, Y_binned, marker=get_marker(kind), color='gray', alpha=0.4, s=15, label=f"{label} (Binned Data)")

        # ---- MODEL 1: PYTORCH NEURAL NETWORK (Optimized on clean binned inputs) ----
        x_nn, y_nn = fit_nn_on_binned_data(X_binned, Y_binned, epochs=250)
        ax1.plot(x_nn, y_nn, color='magenta', linestyle='-.', linewidth=2.5, label='Binned Neural Network')

        # ---- MODEL 2: LOWESS LOCAL REGRESSION ----
        lowess_output = sm.nonparametric.lowess(Y_binned, log_X_binned, frac=0.12)
        ax1.plot(10**lowess_output[:, 0], lowess_output[:, 1], color='blue', linestyle="-", linewidth=2, label="LOWESS Expectation")

        # ---- MODEL 3: NADARAYA-WATSON KERNEL REGRESSION ----
        kr = KernelReg(endog=Y_binned, exog=log_X_binned, var_type='c', bw='cv_ls')
        log_x_grid = np.linspace(log_X_binned.min(), log_X_binned.max(), 500)
        y_smooth_kernel, _ = kr.fit(log_x_grid)
        ax1.plot(10**log_x_grid, y_smooth_kernel, color='red', linestyle="--", linewidth=2, label="Kernel Expectation")

        # ---- MODEL 4: SQUARE ROOT REFERENCE BASELINE ----
        try:
            popt, _ = curve_fit(square_root_law, X_binned, Y_binned, p0=[1.0])
            ax1.plot(10**log_x_grid, square_root_law(10**log_x_grid, 0.5), color='black', linestyle=':', linewidth=2, label=r"Square Root Law ($I \propto Q^{0.5}$)")
        except Exception:
            pass

# Output Layout Finalization
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r'$Q$')
ax1.set_ylabel(r'$I(Q)$')
ax1.tick_params(axis='y')
ax1.grid(True, which="both", ls="-")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(f'images\\impact_volume_curve_analysis\\{image_name}.png')
plt.show()