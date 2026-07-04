import gc
import os
import re
from os import listdir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from tqdm import tqdm


def post_impact_model(x, a, beta):
    return a * (x**(1 - beta) - (x - 1)**(1 - beta))


def der_post_impact(x, a, beta):
    return a * (1 - beta) * (x**(-beta) - (x - 1)**(-beta))


def save_transactions(index_time):
    """
    Extract child-order impact data in the *transaction-count* window
    [index_time, index_time + 1] * meta_duration_transactions after the
    start of the meta-order, expressed in number of transactions rather
    than wall-clock time.

    Only the columns actually needed downstream are read/kept, and
    intermediate data is freed as soon as possible to keep RAM usage low.
    """
    paths = np.array(listdir('database\\trades_20_power_2.0'))

    meta_cols = [
        'BeginTransactionTime', 'EndTransactionTime',
        'sign', 'MetaVolume', 'metaid', 'trader', 'NbChild', 'BeginMid',
    ]
    # NOTE: TransactionTime is a continuous "volume time" (cumsum of
    # quantity / DailyVolume), NOT a discrete transaction count. It must
    # stay float64 throughout - casting it to int would collapse almost
    # every value to 0 and destroy all resolution.
    meta_dtypes = {
        'BeginTransactionTime': 'float64',
        'EndTransactionTime':   'float64',
        'sign':                 'int8',
        'MetaVolume':           'float32',
        'NbChild':              'int32',
        'BeginMid':             'float32',
    }

    trades_cols = ['TransactionTime', 'BeginMid']
    trades_dtypes = {
        'TransactionTime': 'float64',
        'BeginMid':         'float32',
    }

    out_dir = f'database\\{dir}'
    os.makedirs(out_dir, exist_ok=True)
    file_path = f'{out_dir}\\20_power_2.0_{index_time}.csv'
    file_exists = os.path.isfile(file_path)

    total_rows_written = 0
    empty_meta_files    = 0
    no_overlap_windows  = 0

    for path in tqdm(paths, desc=f'save_transactions[{index_time}]', unit='file'):
        trades = pd.read_csv(
            f'database\\trades_20_power_2.0\\{path}',
            sep=',',
            usecols=trades_cols,
            dtype=trades_dtypes,
        )
        trades = trades.dropna(subset=['TransactionTime', 'BeginMid'])

        meta = pd.read_csv(
            f'database\\meta_20_power_2.0\\meta{path[6:]}',
            sep=',',
            usecols=meta_cols,
            dtype=meta_dtypes,
        )
        meta = meta.dropna(subset=['BeginTransactionTime', 'EndTransactionTime'])

        # NOTE: no NbChild filter here on purpose. Every meta-order is
        # saved (with its NbChild value) so the filter can be applied
        # later, at analysis time, without needing to re-run this
        # (expensive) extraction step whenever the threshold changes.
        if meta.empty:
            empty_meta_files += 1
            del trades, meta
            continue

        # sort once so each meta window can be sliced with a cheap
        # binary search instead of re-scanning the whole file every time
        trades = trades.sort_values('TransactionTime')
        trans_time = trades['TransactionTime'].to_numpy()
        begin_mid  = trades['BeginMid'].to_numpy()
        del trades

        rows_out = []

        for meta_row in meta.itertuples(index=False):
            t_begin_meta  = meta_row.BeginTransactionTime
            t_end_meta    = meta_row.EndTransactionTime
            meta_duration = t_end_meta - t_begin_meta
            if meta_duration <= 0:
                continue

            meta_sign     = meta_row.sign
            meta_volume   = meta_row.MetaVolume
            meta_nbchild  = meta_row.NbChild
            meta_beginmid = meta_row.BeginMid

            # window [index_time, index_time + 1] * meta_duration, expressed
            # in the same continuous volume-time units as TransactionTime
            # (cumulative fraction of daily volume), not a transaction count
            t_start_window = t_begin_meta + meta_duration * index_time
            t_end_window   = t_begin_meta + meta_duration * (index_time + 1)

            lo = np.searchsorted(trans_time, t_start_window, side='left')
            hi = np.searchsorted(trans_time, t_end_window,   side='left')
            if hi <= lo:
                no_overlap_windows += 1
                continue

            sel_time = trans_time[lo:hi]
            sel_mid  = begin_mid[lo:hi]

            impact          = (sel_mid - meta_beginmid) * meta_sign
            normalized_time = (sel_time - t_begin_meta) / meta_duration

            rows_out.append(pd.DataFrame({
                'NormalizedTime':  normalized_time,
                'Impact':          impact,
                'MetaVolume':      meta_volume,
                'TransactionTime': sel_time,
                'NbChild':         meta_nbchild,
            }))

        if rows_out:
            out_df = pd.concat(rows_out, ignore_index=True)
            total_rows_written += len(out_df)
            out_df.to_csv(file_path, mode='a', index=False, header=not file_exists)
            file_exists = True
            del out_df

        del meta, rows_out, trans_time, begin_mid
        gc.collect()

    print(
        f'[save_transactions index_time={index_time}] '
        f'rows written: {total_rows_written}, '
        f'meta files with no usable rows: {empty_meta_files}, '
        f'meta windows with no matching trades: {no_overlap_windows}'
    )
    if total_rows_written == 0:
        print(
            f'  WARNING: no data was written to {file_path}. '
            'Check that BeginTransactionTime/EndTransactionTime in the meta '
            'files and TransactionTime in the trades files actually overlap '
            '(e.g. same unit/reference point), otherwise every window will '
            'be empty.'
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

dir           = 'post_transaction_trades'  # output folder for windowed CSVs
images_subdir = 'impact_transaction_time_curve'
length        = 4       # number of meta_duration windows
n_time_bins   = 10      # fine time bins per window
power_exp     = 0.5     # exponent in I(Q) = Y * Q**power_exp (was fit, now fixed & averaged)

min_nb_child = 5   # filter applied at analysis time, not at save time

# ── ensure CSVs exist and are non-empty ──────────────────────────────────────
def _csv_has_data_rows(path):
    """True if the CSV exists and has at least one row beyond the header."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) == 0:
        return False
    with open(path, 'r') as f:
        next(f, None)          # header
        return next(f, None) is not None

for index_time in range(length):
    csv_path = f'database\\{dir}\\20_power_2.0_{index_time}.csv'
    if not _csv_has_data_rows(csv_path):
        if os.path.exists(csv_path):
            print(f'{csv_path} exists but is empty, regenerating it.')
            os.remove(csv_path)
        save_transactions(index_time)

# ── per-window: subdivide into fine time bins, fit Y in each ─────────────────
Y_values     = []
Y_err_values = []
t_centers    = []

paths = np.array(listdir(f'database\\{dir}'))

for path in tqdm(sorted(paths[:length]), desc='windows', unit='file'):
    post_trades = pd.read_csv(f'database\\{dir}\\{path}', sep=',')
    post_trades = post_trades[post_trades['NbChild'] >= min_nb_child]
    if post_trades.empty:
        print(f'{path}: no rows left after NbChild >= {min_nb_child} filter, skipping.')
        continue

    match = re.search(r"_(\d+)\.csv$", path)
    index = int(match.group(1))

    # normalized (transaction) time for this file runs in [1+index, 2+index]
    t_lo = index
    t_hi = 1.0 + index
    time_bins = np.linspace(t_lo, t_hi, n_time_bins + 1)

    post_trades['time_bin'] = pd.cut(
        post_trades['NormalizedTime'], bins=time_bins, include_lowest=True
    )

    # ── one figure per time window ────────────────────────────────────────────
    fig_ll, axes_ll = plt.subplots(4, 3, figsize=(18, 20))
    axes_ll = axes_ll.flatten()
    subplot_idx = 0

    time_bin_groups = list(post_trades.groupby('time_bin', observed=True))
    for tb, group in tqdm(time_bin_groups, desc=f'fitting bins [{path}]', unit='bin'):
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
            Q_mean =('MetaVolume', 'mean'),
            I_mean =('Impact',     'mean'),
            I_std  =('Impact',     'std'),
            count  =('Impact',     'count'),
        ).dropna()

        vol_grouped = vol_grouped[vol_grouped['count'] >= 3]
        if len(vol_grouped) < 3:
            continue

        Q     = vol_grouped['Q_mean'].to_numpy()
        I     = vol_grouped['I_mean'].to_numpy()
        I_err = vol_grouped['I_std'].to_numpy() / np.sqrt(vol_grouped['count'].to_numpy())

        # ── average I / Q**power_exp (no fit) ─────────────────────────────
        h = np.histogram(group['MetaVolume'], bins=bins_vol)
        max_h = np.max(h[0])

        mask = np.where(h[0] > max_h * 0.5)

        if len(mask[0]) == 0:
            print(f'  No points above histogram threshold for time bin [{tb.left:.2f}, {tb.right:.2f}], skipping.')
            continue

        ratios     = I[mask] / Q[mask]**power_exp
        ratio_errs = I_err[mask] / Q[mask]**power_exp

        # inverse-variance weighted average of I / Q**power_exp
        weights = 1.0 / ratio_errs**2
        Y     = np.sum(weights * ratios) / np.sum(weights)
        Y_err = np.sqrt(1.0 / np.sum(weights))

        mask = np.where(I > 0.0)
        Q          = Q[mask]
        I          = I[mask]
        I_err      = I_err[mask]
        ratio      = I / Q**power_exp
        ratio_err  = I_err / Q**power_exp

        # ── draw into the next subplot ──────────────────────────────────────
        if subplot_idx < len(axes_ll):
            ax = axes_ll[subplot_idx]
            subplot_idx += 1

            ax.set_title(f'$t/T$ in [{tb.left:.3f}, {tb.right:.3f}] (transactions)', fontsize=11)

            # Twin axis: histogram of MetaVolume (frequency) on the right
            ax_hist = ax.twinx()
            ax_hist.hist(group['MetaVolume'], bins=bins_vol, color='steelblue', alpha=0.25,
                         label='Volume freq.')
            ax_hist.set_ylabel('Frequency', fontsize=9)
            ax_hist.tick_params(axis='y', labelsize=8)
            ax_hist.set_xscale('log')
            ax_hist.set_zorder(ax.get_zorder() - 1)
            ax.set_facecolor('none')

            # Binned mean of I / Q**power_exp with error bars (no fit)
            ax.errorbar(Q, ratio, yerr=ratio_err,
                        fmt='o', color='crimson', markersize=5,
                        linewidth=1.2, capsize=3, label='Binned mean', zorder=5)

            ax.axhline(Y, linestyle='--', color='black', linewidth=1.5,
                       label=f'$Y$={Y:.3f}±{Y_err:.3f}', zorder=6)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('$Q$')
            ax.set_ylabel(rf'$I_i / Q^{{{power_exp}}}$')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, which='both', ls='--', alpha=0.3)

        t_centers.append(t_mid)
        Y_values.append(Y)
        Y_err_values.append(Y_err)

    # hide any unused subplots in this window's figure
    for idx in range(subplot_idx, len(axes_ll)):
        axes_ll[idx].set_visible(False)

    plt.tight_layout()
    os.makedirs(f'images\\{images_subdir}', exist_ok=True)
    plt.savefig(f'images\\{images_subdir}\\{dir}_impact_single_{index}.png', dpi=150, bbox_inches='tight')
    plt.close()

    del post_trades
    gc.collect()

t_centers    = np.array(t_centers)
Y_values     = np.array(Y_values)
Y_err_values = np.array(Y_err_values)

print(f'\nTotal points collected: {len(t_centers)}')

# ── restrict to t > 1.0 for the decay fit ────────────────────────────────────
fit_mask  = t_centers > 1.0
t_fit     = t_centers[fit_mask]
Y_fit     = Y_values[fit_mask]
Y_err_fit = Y_err_values[fit_mask]

print(f'Points used for decay fit (t > 1.0): {len(t_fit)}')

if len(t_fit) == 0:
    raise SystemExit(
        'No data points available for the decay fit (t > 1.0). '
        'This means every windowed CSV in database\\' + dir + ' ended up empty, '
        'or every time/volume bin had too few points to fit. '
        'Check the "[save_transactions ...]" diagnostic lines printed above '
        '(rows written / no matching trades) to see where the data is being lost, '
        'and verify that BeginTransactionTime/EndTransactionTime/TransactionTime '
        'are on a consistent scale across the meta and trades files.'
    )

# ── fit post-impact decay to Y(t) ────────────────────────────────────────────
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
            label=r'Fitted curve')

# ── plot ──────────────────────────────────────────────────────────────────────
plt.errorbar(t_centers, Y_values, yerr=Y_err_values,
             linestyle='', marker='o', color='C0', label=r'$Y(t/T)$')

plt.xlabel(r'$t / T$ (transactions)')
plt.ylabel(rf'$I / Q^{{{power_exp}}}$')
plt.legend()
plt.grid(True, which='both', ls='-')
plt.tight_layout()

os.makedirs(f'images\\{images_subdir}', exist_ok=True)
plt.savefig(f'images\\{images_subdir}\\{dir}_impact_time_curve.png')
#plt.show()