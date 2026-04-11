import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
import os
from scipy.optimize import curve_fit
import re

def post_impact_model(x, a, beta):
    return a * (x**(1 - beta) - (x - 1)**(1 - beta))

def der_post_impact(x, a, beta):
    return a * (1 - beta) * (x**(-beta) - (x - 1)**(-beta))

def sqrt_law(Q, Y):
    """Power law with delta fixed to 0.5: I(Q) = Y * sqrt(Q)"""
    return Y * np.sqrt(Q)

def save(index_time):
    paths = np.array(listdir('..\\database\\trades_20_power_2.0'))

    for path in paths:
        print(path)
        trades = pd.read_csv(
            f'..\\database\\trades_20_power_2.0\\{path}',
            sep=',',
            parse_dates=['BeginTime']
        )

        meta = pd.read_csv(
            f'..\\database\\meta_20_power_2.0\\meta{path[6:]}',
            sep=',',
            parse_dates=['BeginTime', 'EndTime']
        )

        meta = meta[meta['NbChild'] > 1]

        for index, meta_row in meta.iterrows():
            t_begin_meta  = meta_row['BeginTime']
            t_end_meta    = meta_row['EndTime']
            meta_duration = t_end_meta - t_begin_meta
            meta_sign     = meta_row['sign']
            meta_volume   = meta_row['MetaVolume']
            metaid        = meta_row['metaid']
            meta_trader   = meta_row['trader']

            # window [index_time, index_time+1] * meta_duration after t_end
            t_start_window = t_begin_meta + meta_duration * index_time
            t_end_window   = t_begin_meta + meta_duration * (index_time + 1)

            post_trades = trades[
                (trades['BeginTime'] >= t_start_window) &
                (trades['BeginTime'] <  t_end_window)
            ].copy()

            if post_trades.empty:
                continue

            impact = (post_trades['BeginMid'] - meta_row['BeginMid']) * meta_sign

            time_lag_seconds = (post_trades['BeginTime'] - t_begin_meta).dt.total_seconds()
            normalized_time  = time_lag_seconds / meta_duration.total_seconds()

            post_trades['NormalizedTime'] = normalized_time.values
            post_trades['Impact']         = impact.values
            post_trades['MetaVolume']     = meta_volume

            print(t_begin_meta, t_end_meta, meta_duration)
            print(post_trades[['NormalizedTime', 'Impact', 'MetaVolume']])

            file_path   = f'..\\database\\{dir}\\20_power_2.0_{index_time}.csv'
            file_exists = os.path.isfile(file_path)
            post_trades[['NormalizedTime', 'Impact', 'MetaVolume']].to_csv(
                file_path, mode='a', index=False, header=not file_exists
            )

# ── plotting config ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size':       12,
    'axes.titlesize':  20,
    'axes.labelsize':  16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
})

dir    = 'post_trades'
length = 3       # number of meta_duration windows after execution end
n_time_bins = 12 # fine time bins per window

# ── ensure CSVs exist ────────────────────────────────────────────────────────
for index_time in range(length):
    if not os.path.exists(f'..\\database\\{dir}\\20_power_2.0_{index_time}.csv'):
        save(index_time)

# ── per-window: subdivide into fine time bins, fit Y in each ─────────────────
Y_values     = []
Y_err_values = []
t_centers    = []

paths = np.array(listdir(f'..\\database\\{dir}'))

for path in sorted(paths[:length]):
    print(path)
    post_trades = pd.read_csv(f'..\\database\\{dir}\\{path}', sep=',')

    match = re.search(r"_(\d+)\.csv$", path)
    index = int(match.group(1))

    # normalized time for this file runs in [1+index, 2+index]
    t_lo = index
    t_hi = 1.0 + index
    if index == 0:
        time_bins = np.linspace(t_lo, t_hi, n_time_bins + 1)**3
    time_bins = np.linspace(t_lo, t_hi, n_time_bins + 1)

    post_trades['time_bin'] = pd.cut(
        post_trades['NormalizedTime'], bins=time_bins, include_lowest=True
    )

    # ── one figure per time window ────────────────────────────────────────────
    fig_ll, axes_ll = plt.subplots(4, 3, figsize=(18, 20))
    axes_ll = axes_ll.flatten()
    subplot_idx = 0

    for tb, group in post_trades.groupby('time_bin', observed=True):
        print(tb)
        if len(group) < 10:   # skip bins with too few points for a reliable fit
            continue

        t_mid = group['NormalizedTime'].mean()   # mean of actual data, not bin midpoint

        # ── log bins over MetaVolume ─────────────────────────────────────────
        min_vol = group['MetaVolume'].min()
        max_vol = group['MetaVolume'].max()
        if min_vol <= 0 or min_vol == max_vol:
            continue

        bins_vol = np.logspace(np.log10(min_vol), np.log10(max_vol), 51)
        group = group.copy()
        group['vol_bin'] = pd.cut(group['MetaVolume'], bins=bins_vol, include_lowest=True)

        vol_grouped = group.groupby('vol_bin', observed=True).agg(
            Q_mean  =('MetaVolume', 'mean'),
            I_mean  =('Impact',     'mean'),
            I_std   =('Impact',     'std'),
            count   =('Impact',     'count'),
        ).dropna()

        vol_grouped = vol_grouped[vol_grouped['count'] >= 3]
        if len(vol_grouped) < 3:
            continue

        Q     = vol_grouped['Q_mean'].to_numpy()
        I     = vol_grouped['I_mean'].to_numpy()
        I_err = vol_grouped['I_std'].to_numpy() / np.sqrt(vol_grouped['count'].to_numpy())

        # ── fit ────────────────────────────────────
        try:
            h = np.histogram(group['MetaVolume'], bins=bins_vol)
            max_h = np.max(h[0])

            mask = np.where(h[0] > max_h * 0.5)

            popt, pcov = curve_fit(
                sqrt_law, Q[mask], I[mask],
                sigma=I_err[mask], absolute_sigma=True,
                p0=[1.0]
            )
            Y     = popt[0]
            Y_err = np.sqrt(pcov[0][0])
        except RuntimeError:
            print(f'  Fit failed for time bin [{tb.left:.2f}, {tb.right:.2f}], skipping.')
            continue

        mask = np.where(I > 0.0)
        Q     = Q[mask]
        I     = I[mask]
        I_err = I_err[mask]

        print(Y)

        x_fit = np.geomspace(min_vol, max_vol, 200)
        y_fit = Y * np.sqrt(x_fit)

        # ── draw into the next subplot (matching impact_time_curve.py design) ─
        if subplot_idx < len(axes_ll):
            ax = axes_ll[subplot_idx]
            subplot_idx += 1

            ax.set_title(f'$t/T$ in [{tb.left:.3f}, {tb.right:.3f}]', fontsize=11)

            # Twin axis: histogram of MetaVolume (frequency) on the right
            ax_hist = ax.twinx()
            ax_hist.hist(group['MetaVolume'], bins=bins_vol, color='steelblue', alpha=0.25,
                         label='Volume freq.')
            ax_hist.set_ylabel('Frequency', fontsize=9)
            ax_hist.tick_params(axis='y', labelsize=8)
            ax_hist.set_xscale('log')
            ax_hist.set_zorder(ax.get_zorder() - 1)
            ax.set_facecolor('none')

            # Binned mean impact with error bars
            ax.errorbar(Q, I, yerr=I_err,
                        fmt='o', color='crimson', markersize=5,
                        linewidth=1.2, capsize=3, label='Binned mean', zorder=5)

            # Square root fit curve
            ax.plot(x_fit, y_fit,
                    linestyle='--', color='black', linewidth=1.5,
                    label=f'$Y\\sqrt{{Q}}$, $Y$={Y:.3f}±{Y_err:.3f}', zorder=6)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('$Q$')
            ax.set_ylabel(r'$I_i$')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, which='both', ls='--', alpha=0.3)

        t_centers.append(t_mid)
        Y_values.append(Y)
        Y_err_values.append(Y_err)

    # hide any unused subplots in this window's figure
    for idx in range(subplot_idx, len(axes_ll)):
        axes_ll[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'..\\images\\impact_time_curve_ext\\{dir}_impact_single_{index}.png', dpi=150, bbox_inches='tight')
    plt.close()

t_centers    = np.array(t_centers)
Y_values     = np.array(Y_values)
Y_err_values = np.array(Y_err_values)

print(f'\nTotal points collected: {len(t_centers)}')

# ── restrict to t > 1.0 for the decay fit ────────────────────────────────────
fit_mask     = t_centers > 1.0
t_fit        = t_centers[fit_mask]
Y_fit        = Y_values[fit_mask]
Y_err_fit    = Y_err_values[fit_mask]

print(f'Points used for decay fit (t > 1.0): {len(t_fit)}')

# ── fit post-impact decay to Y(t) ────────────────────────────────────────────
# iterative effective-error fit
popt, pcov = curve_fit(
    post_impact_model, t_fit, Y_fit,
    sigma=Y_err_fit, absolute_sigma=True,
    p0=[Y_fit[0], 0.5]
)
a_fit, beta_fit = popt

for _ in range(10):
    eff_err = Y_err_fit  # only y-errors here; x-errors negligible within a bin
    popt, pcov = curve_fit(
        post_impact_model, t_fit, Y_fit,
        sigma=eff_err, absolute_sigma=True,
        p0=popt
    )
    a_fit, beta_fit = popt

a_err    = np.sqrt(pcov[0][0])
beta_err = np.sqrt(pcov[1][1])
print(f'Post-impact fit: a = {a_fit:.4f} ± {a_err:.4f},  beta = {beta_fit:.4f} ± {beta_err:.4f}')

x_th = np.linspace(t_fit.min(), t_centers.max(), 300)
plt.plot(x_th, post_impact_model(x_th, a_fit, beta_fit),
            linestyle=':', color='black',
            label=rf'Fitted curve')

# ── plot ──────────────────────────────────────────────────────────────────────
plt.errorbar(t_centers, Y_values, yerr=Y_err_values,
             linestyle='', marker='o', color='C0', label=r'$Y(t/T)$')

plt.xlabel(r'$t / T$')
plt.ylabel(r'$I / \sqrt{Q}$)')
plt.legend()
plt.grid(True, which='both', ls='-')
plt.tight_layout()

plt.savefig(f'..\\images\\impact_time_curve_ext\\{dir}_impact_time_curve.png')
#plt.show()