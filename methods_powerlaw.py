import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random, datetime
from scipy.stats import powerlaw
from os import listdir

def generate(path, nb_traders, kind, exponent, start_id=0, data_dir = 'database\\data'):
    '''
    Generate metaorders from trades, and compute the partial impact 
    (intra-metaorder impact) and partial volume (intra-metaorder volume)
    for each child trade and then normalizes these partial features by the metaorder total.
    '''
    trades = pd.read_csv(f"{data_dir}\\{path}", header=None)
    t = np.array(trades[0])

    year = path[5:9]
    month = path[10:12]
    day = path[13:15]

    date_string = f"{year}-{month}-{day}"

    start_of_day_dt = pd.to_datetime(date_string)

    time_delta = pd.to_timedelta(t, unit='s')

    timestamp = start_of_day_dt + time_delta

    prices = np.array(trades[1])
    volumes = np.array(trades[2])
    signs = np.array(trades[3])

    # 2. Create Pandas DataFrame
    trades = pd.DataFrame({'timestamp':timestamp,'sign':signs,'quantity':volumes,'price':prices})

    trades['day'] = trades['timestamp'].dt.date

    trades.sort_values('timestamp', inplace = True)
    
    # Calculate daily volatility and volume
    daily_stats = trades.groupby('day').agg(
        DailySigma=('price', lambda x: (x.max() - x.min()) / x.iloc[0]),  # (max - min) / open
        DailyVolume=('quantity', 'sum')  # Sum of quantity for each day
    ).reset_index()

    trades = trades.merge(daily_stats, on='day', how='left')
    trades['BeginMid'] = np.log(trades['price']) / trades['DailySigma']
    trades['EndTime'] = trades['timestamp']
    trades['EndMid'] = trades['BeginMid'].shift(-1)

    trades.drop(trades.tail(1).index, inplace = True)

    trades['Volume'] = trades['quantity'] / trades['DailyVolume']

    trades['TradedVolume'] = trades['Volume'].cumsum()

    trades.drop(columns=['price', 'quantity', 'DailySigma'], inplace=True)

    trades = trades.rename(columns={
        'timestamp': 'BeginTime'
    })

    # 1. Attribute to each trade a trader
    trades['trader'] = mapping_function(trades, nb_traders, kind, exponent) 

    # 2. Identify Metaorders
    sorted_trades  = trades.sort_values(['trader','BeginTime']).reset_index(drop=True)
    
    # Calculate metaid
    sorted_trades['metaid'] = np.where(
        (sorted_trades['trader']!=sorted_trades['trader'].shift()) | 
        (sorted_trades.sign.shift()!= sorted_trades.sign) | 
        (sorted_trades.day!= sorted_trades.day.shift()), 
        1, 0
    ).cumsum() + start_id

    start_cum_volume = sorted_trades.groupby('metaid')['TradedVolume'].transform('first')
    start_volume = sorted_trades.groupby('metaid')['Volume'].transform('first')

    # Volume traded by everyone until a certain point
    sorted_trades['TradedVolume'] = sorted_trades['TradedVolume'] - start_cum_volume + start_volume

    sorted_trades['TradedVolume'] = sorted_trades['TradedVolume'] / sorted_trades.groupby('metaid')['TradedVolume'].transform('last')

    # Identifica il prezzo iniziale (BeginMid) del PRIMO trade per ogni metaordine
    start_prices = sorted_trades.groupby('metaid')['BeginMid'].transform('first')
    
    # Calcola l'impatto parziale come differenza tra il prezzo di fine trade corrente e il prezzo iniziale del metaordine
    sorted_trades['PartialImpact'] = (sorted_trades['EndMid'] - start_prices) * sorted_trades['sign']

    sorted_trades['PartialImpact_pre'] = (sorted_trades['BeginMid'] - start_prices) * sorted_trades['sign']

    # Volume traded by the trader until a certain point
    sorted_trades['PartialVolume'] = sorted_trades.groupby('metaid')['Volume'].cumsum()

    # CHANGE 1: 'trader':'first' instead of 'trader':'count' so the trader label is
    # preserved in metaorders_agg. NbChild is now computed separately below.
    metaorders_agg = sorted_trades.groupby('metaid').agg({
        'sign':'first', 
        'BeginMid':'first', 
        'EndMid':'last',
        'trader':'first',       # <-- was 'count': now saves the trader label
        'BeginTime':'first', 
        'EndTime':'last',
        'Volume':'sum',
        'TradedVolume':'last',
        'DailyVolume': 'first'
    }).reset_index()

    # CHANGE 2: NbChild for metaorders_agg is now computed via .size() and merged in,
    # since 'trader':'count' was removed from the agg above.
    nb_child = sorted_trades.groupby('metaid').size().reset_index(name='NbChild')
    metaorders_agg = metaorders_agg.merge(nb_child, on='metaid')

    # Number of children (unchanged — still computed on sorted_trades)
    sorted_trades['NbChild'] = sorted_trades.groupby('metaid')['trader'].transform('count')

    # Volume of the entire metaorder
    sorted_trades['MetaVolume'] = sorted_trades.groupby('metaid')['PartialVolume'].transform('last')

    begintime = sorted_trades.groupby('metaid')['BeginTime'].transform('first')
    endtime = sorted_trades.groupby('metaid')['EndTime'].transform('last')

    # Metaorder duration using metaorder start and end time
    sorted_trades['MetaDuration'] = (endtime - begintime).dt.total_seconds()

    # Partial time using trade start time and metaorder start time
    sorted_trades['ElapsedTime'] = (sorted_trades['BeginTime'] - begintime).dt.total_seconds()

    sorted_trades['Ratio'] = sorted_trades['PartialImpact'] / np.sqrt(sorted_trades['MetaVolume'])

    sorted_trades['Ratio_pre'] = sorted_trades['PartialImpact_pre'] / np.sqrt(sorted_trades['MetaVolume'])

    # CHANGE 3: 'trader' removed from the drop list so it is kept in sorted_trades.
    sorted_trades.drop(columns=['EndTime', 'day'], inplace=True)

    metaorders_agg['MetaImpact'] = (metaorders_agg['EndMid'] - metaorders_agg['BeginMid']) * metaorders_agg['sign']

    # CHANGE 4: 'trader':'NbChild' removed from rename because 'trader' now holds the
    # label and NbChild already exists as its own column.
    metaorders_agg = metaorders_agg.rename(columns={
        'Volume':'MetaVolume', 'Impact': 'MetaImpact'
    })

    metaorders_agg['Ratio'] = metaorders_agg['MetaImpact'] / np.sqrt(metaorders_agg['MetaVolume'])

    return metaorders_agg, sorted_trades

def generate_slim(path, nb_traders, kind, exponent, data_dir='database\\data'):
    """
    Lightweight version of generate() that returns only the three columns
    needed for the impact/volume study: MetaVolume, MetaImpact, NbChild.
 
    Skips all partial-impact, partial-volume, and time-series bookkeeping,
    so memory and CPU usage scale much better when collecting large samples.
    """
    trades = pd.read_csv(f"{data_dir}\\{path}", header=None)
    t = np.array(trades[0])
 
    year, month, day = path[5:9], path[10:12], path[13:15]
    start_of_day_dt = pd.to_datetime(f"{year}-{month}-{day}")
    timestamp = start_of_day_dt + pd.to_timedelta(t, unit='s')
 
    prices  = np.array(trades[1])
    volumes = np.array(trades[2])
    signs   = np.array(trades[3])
 
    trades = pd.DataFrame({
        'timestamp': timestamp,
        'sign':      signs,
        'quantity':  volumes,
        'price':     prices,
    })
    trades['day'] = trades['timestamp'].dt.date
    trades.sort_values('timestamp', inplace=True)
 
    # Daily normalisation factors
    daily_stats = trades.groupby('day').agg(
        DailySigma=('price',    lambda x: (x.max() - x.min()) / x.iloc[0]),
        DailyVolume=('quantity', 'sum'),
    ).reset_index()
    trades = trades.merge(daily_stats, on='day', how='left')
 
    # Normalised price (only first/last per metaorder needed)
    trades['Mid']    = np.log(trades['price']) / trades['DailySigma']
    trades['Volume'] = trades['quantity'] / trades['DailyVolume']
 
    # Assign traders and build metaorder ids
    trades['trader'] = mapping_function(trades, nb_traders, kind, exponent)
    trades = trades.sort_values(['trader', 'timestamp']).reset_index(drop=True)
 
    trades['metaid'] = np.where(
        (trades['trader'] != trades['trader'].shift()) |
        (trades['sign']   != trades['sign'].shift())   |
        (trades['day']    != trades['day'].shift()),
        1, 0
    ).cumsum()
 
    # Aggregate: only what is needed
    agg = trades.groupby('metaid').agg(
        sign       =('sign',      'first'),
        BeginMid   =('Mid',       'first'),
        EndMid     =('Mid',       'last'),
        MetaVolume =('Volume',    'sum'),
        NbChild    =('trader',    'count'),
    ).reset_index(drop=True)
 
    agg['MetaImpact'] = (agg['EndMid'] - agg['BeginMid']) * agg['sign']
 
    return agg[['MetaVolume', 'MetaImpact', 'NbChild']]

def mapping_function(trades, nb_traders, kind, alpha = 0.0):
    if kind == 'power':
        shape = float(alpha) if float(alpha) > 0 else 1.5
        # scipy.stats.powerlaw is bounded on [0,1] with PDF f(x) = alpha * x^(alpha-1)
        # alpha < 1 → skewed toward 0 (few traders dominate)
        # alpha > 1 → skewed toward 1 (more uniform)
        samples = powerlaw.rvs(shape, size=int(nb_traders))
        samples = np.sort(samples)[::-1]  # descending: higher weight → lower trader index
    elif kind == 'uniform':
        samples = np.ones(int(nb_traders))
    else:
        samples = np.ones(int(nb_traders))

    probabilities = samples / samples.sum()
    
    # Usa gli indici per il campionamento per mantenere l'ordine nel plot
    trader_indices = np.arange(int(nb_traders))
    chosen_indices = np.random.choice(trader_indices, size=len(trades), p=probabilities)
    
    return [f"Trader {i+1}" for i in chosen_indices]

def verify_distributions(nb_traders=20, n_samples=500_000, power_exponent=2.0):
    """
    Verify that mapping_function produces the expected distributions:
      - 'uniform': all traders get roughly equal share  → chi-squared test
      - 'power':   trader weights follow scipy.stats.powerlaw(alpha) on [0,1]
                   → KS test against powerlaw CDF + log-log linearity of rank-share

    Parameters
    ----------
    nb_traders     : number of traders
    n_samples      : number of trade assignments to draw
    power_exponent : shape parameter passed to scipy.stats.powerlaw.rvs
                     PDF: f(x) = alpha * x^(alpha-1) on [0,1]
                     alpha < 1 → concentrated near 0 (heavy skew, few traders dominate)
                     alpha > 1 → concentrated near 1 (mild skew, more uniform)
    """
    from scipy.stats import chisquare, kstest, powerlaw as sp_powerlaw

    dummy_trades = pd.DataFrame({'timestamp': range(n_samples),
                                 'sign': np.ones(n_samples)})
    labels = [f"Trader {i+1}" for i in range(nb_traders)]

    # ------------------------------------------------------------------ #
    #  1. UNIFORM                                                          #
    # ------------------------------------------------------------------ #
    result_uniform = mapping_function(dummy_trades, nb_traders, 'uniform', 0.0)
    counts_uniform = np.array([result_uniform.count(l) for l in labels], dtype=float)
    shares_uniform = counts_uniform / counts_uniform.sum()

    expected_uniform = np.full(nb_traders, 1.0 / nb_traders)
    chi2_stat, chi2_p = chisquare(counts_uniform, f_exp=counts_uniform.sum() * expected_uniform)

    print("=" * 60)
    print("UNIFORM DISTRIBUTION VERIFICATION")
    print("=" * 60)
    print(f"  Chi-squared statistic : {chi2_stat:.4f}")
    print(f"  p-value               : {chi2_p:.4f}")
    if chi2_p > 0.05:
        print("  ✓ Cannot reject uniformity (p > 0.05)")
    else:
        print("  ✗ Significant deviation from uniformity (p ≤ 0.05)")
    print(f"  Min share : {shares_uniform.min():.4f}  "
          f"Max share : {shares_uniform.max():.4f}  "
          f"Expected  : {1/nb_traders:.4f}")

    # ------------------------------------------------------------------ #
    #  2. POWER LAW  (scipy.stats.powerlaw)                               #
    # ------------------------------------------------------------------ #
    result_power = mapping_function(dummy_trades, nb_traders, 'power', power_exponent)
    counts_power = np.array([result_power.count(l) for l in labels], dtype=float)
    shares_power = counts_power / counts_power.sum()

    # Sort descending: highest-weight trader first (mirrors mapping_function sort)
    observed_shares_sorted = np.sort(shares_power)[::-1]

    # KS test: fit scipy.stats.powerlaw to the raw weight samples.
    # Because weights are normalised, we recover the underlying samples by
    # rescaling observed shares back to [0,1] range for the fit.
    weights_rescaled = observed_shares_sorted / observed_shares_sorted.max()
    fit_alpha, fit_loc, fit_scale = sp_powerlaw.fit(weights_rescaled, floc=0, fscale=1)
    ks_stat, ks_p = kstest(
        weights_rescaled,
        lambda x: sp_powerlaw.cdf(x, fit_alpha, loc=fit_loc, scale=fit_scale)
    )

    # Log-log linearity (Zipf / rank-size): rank vs share linear on log-log axes
    ranks = np.arange(1, nb_traders + 1)
    log_ranks  = np.log(ranks)
    log_shares = np.log(observed_shares_sorted)
    slope, intercept = np.polyfit(log_ranks, log_shares, 1)
    ss_res = np.sum((log_shares - (slope * log_ranks + intercept)) ** 2)
    ss_tot = np.sum((log_shares - log_shares.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Expected mean and variance of powerlaw(alpha) on [0,1]
    expected_mean = power_exponent / (power_exponent + 1)
    expected_var  = power_exponent / ((power_exponent + 2) * (power_exponent + 1) ** 2)

    print()
    print("=" * 60)
    print(f"POWER-LAW DISTRIBUTION VERIFICATION  (scipy.stats.powerlaw, α={power_exponent})")
    print("=" * 60)
    print(f"  scipy.stats.powerlaw PDF: f(x) = α·x^(α-1) on [0,1]")
    print(f"  Expected mean of weights : {expected_mean:.4f}  "
          f"Observed mean : {weights_rescaled.mean():.4f}")
    print(f"  Expected var  of weights : {expected_var:.6f}  "
          f"Observed var  : {weights_rescaled.var():.6f}")
    print(f"  Fitted α (KS fit)        : {fit_alpha:.4f}  (input: {power_exponent:.4f})")
    print(f"  KS statistic             : {ks_stat:.4f}")
    print(f"  KS p-value               : {ks_p:.4f}")
    if ks_p > 0.05:
        print("  ✓ Cannot reject powerlaw fit (p > 0.05)")
    else:
        print("  ✗ Significant deviation from powerlaw(α) (p ≤ 0.05)")
    print(f"  Log-log slope (Zipf)     : {slope:.4f}  (R² = {r2:.4f})")
    if r2 > 0.90:
        print("  ✓ Strong log-log linearity (R² > 0.90)")
    else:
        print("  ✗ Weak log-log linearity")

    # ------------------------------------------------------------------ #
    #  3. PLOTS                                                            #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    trader_idx = np.arange(1, nb_traders + 1)

    # Uniform: bar chart of shares
    ax = axes[0, 0]
    ax.bar(trader_idx, shares_uniform, color='steelblue', alpha=0.8, label='Observed')
    ax.axhline(1 / nb_traders, color='crimson', lw=2, linestyle='--', label='Expected')
    ax.set_title(f"Uniform – trader shares\n(χ² p={chi2_p:.3f})")
    ax.set_xlabel("Trader index")
    ax.set_ylabel("Share")
    ax.legend()

    # Uniform: deviation from expected
    ax = axes[0, 1]
    deviations = shares_uniform - 1 / nb_traders
    colors = ['tomato' if d < 0 else 'seagreen' for d in deviations]
    ax.bar(trader_idx, deviations * 100, color=colors, alpha=0.8)
    ax.axhline(0, color='black', lw=1)
    ax.set_title("Uniform – deviation from expected (%)")
    ax.set_xlabel("Trader index")
    ax.set_ylabel("Δ share (pp)")

    # Power-law: rank–share log-log plot with fit line
    ax = axes[1, 0]
    ax.scatter(ranks, observed_shares_sorted, color='darkorange', zorder=3, label='Observed shares')
    fit_line = np.exp(intercept) * ranks ** slope
    ax.plot(ranks, fit_line, 'k--', lw=2,
            label=f'Log-log fit  slope={slope:.2f}  R²={r2:.3f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f"Power-law – rank vs share (log-log)\n(KS p={ks_p:.3f})")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Share")
    ax.legend(fontsize=8)

    # Power-law: sorted bar + theoretical powerlaw PDF overlay on rescaled weights
    ax = axes[1, 1]
    ax.bar(trader_idx, observed_shares_sorted, color='darkorange', alpha=0.7, label='Observed')
    ax2 = ax.twinx()
    x_grid = np.linspace(1e-3, 1, 200)
    ax2.plot(
        np.linspace(1, nb_traders, 200),
        sp_powerlaw.pdf(x_grid[::-1], fit_alpha),   # reversed: rank 1 = highest weight
        color='navy', lw=2, label=f'powerlaw PDF (α={fit_alpha:.2f})'
    )
    ax.set_title(f"Power-law – sorted shares + fitted PDF\n"
                 f"(input α={power_exponent}, fitted α={fit_alpha:.2f})")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Observed share", color='darkorange')
    ax2.set_ylabel("PDF value", color='navy')
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8)

    plt.tight_layout()
    plt.savefig("images\\distribution_verification_powerlaw.png", dpi=150)
    plt.show()
    print("\nPlot saved to distribution_verification.png")


if __name__ == "__main__":
    verify_distributions(nb_traders=20, n_samples=100_000_000, power_exponent=2.0)