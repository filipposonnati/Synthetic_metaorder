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

    trades['TransactionTime'] = trades['Volume'].cumsum()

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

    start_cum_volume = sorted_trades.groupby('metaid')['TransactionTime'].transform('first')
    start_volume = sorted_trades.groupby('metaid')['Volume'].transform('first')

    # Volume traded by everyone until a certain point
    sorted_trades['TradedVolume'] = sorted_trades['TransactionTime'] - start_cum_volume + start_volume

    sorted_trades['TradedVolume'] = sorted_trades['TradedVolume'] / sorted_trades.groupby('metaid')['TradedVolume'].transform('last')

    # Identifica il prezzo iniziale (BeginMid) del PRIMO trade per ogni metaordine
    start_prices = sorted_trades.groupby('metaid')['BeginMid'].transform('first')
    
    # Calcola l'impatto parziale come differenza tra il prezzo di fine trade corrente e il prezzo iniziale del metaordine
    sorted_trades['PartialImpact'] = (sorted_trades['EndMid'] - start_prices) * sorted_trades['sign']

    sorted_trades['PartialImpact_pre'] = (sorted_trades['BeginMid'] - start_prices) * sorted_trades['sign']

    # Volume traded by the trader until a certain point
    sorted_trades['PartialVolume'] = sorted_trades.groupby('metaid')['Volume'].cumsum()

    sorted_trades['EndTransactionTime'] = sorted_trades['TransactionTime']

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
        'DailyVolume': 'first',
        'TransactionTime': 'first',
        'EndTransactionTime': 'last'
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
    sorted_trades.drop(columns=['EndTime', 'day', 'EndTransactionTime'], inplace=True)

    metaorders_agg['MetaImpact'] = (metaorders_agg['EndMid'] - metaorders_agg['BeginMid']) * metaorders_agg['sign']

    # CHANGE 4: 'trader':'NbChild' removed from rename because 'trader' now holds the
    # label and NbChild already exists as its own column.
    metaorders_agg = metaorders_agg.rename(columns={
        'Volume':'MetaVolume', 'Impact': 'MetaImpact', 'TransactionTime': 'BeginTransactionTime'
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
        # Zipf / rank-size law: w(k) ∝ k^{-shape}
        # Deterministic weights → rank-share log-log slope is exactly -shape
        ranks = np.arange(1, int(nb_traders) + 1)
        samples = ranks.astype(float) ** (-shape)   # already descending (rank 1 is largest)
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
      - 'power':   trader weights are exact Zipf w(k) = k^{-alpha}
                   → log-log slope check, R², chi-squared vs theoretical shares

    Parameters
    ----------
    nb_traders     : number of traders
    n_samples      : number of trade assignments to draw
    power_exponent : Zipf exponent α — slope of the rank-share log-log line
    """
    from scipy.stats import chisquare

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
    #  2. POWER LAW  (Zipf: deterministic weights w(k) ∝ k^{-alpha})     #
    # ------------------------------------------------------------------ #
    result_power = mapping_function(dummy_trades, nb_traders, 'power', power_exponent)
    counts_power = np.array([result_power.count(l) for l in labels], dtype=float)
    shares_power = counts_power / counts_power.sum()

    # Trader shares are already in order (Trader 1 = rank 1 = highest weight)
    observed_shares = shares_power  # keep in trader-index order

    # Theoretical Zipf weights (normalised), exact by construction
    ranks = np.arange(1, nb_traders + 1)
    zipf_weights = ranks.astype(float) ** (-power_exponent)
    theoretical_shares = zipf_weights / zipf_weights.sum()

    # Log-log slope: should be exactly -power_exponent
    log_ranks  = np.log(ranks)
    log_shares = np.log(observed_shares)
    slope, intercept = np.polyfit(log_ranks, log_shares, 1)
    ss_res = np.sum((log_shares - (slope * log_ranks + intercept)) ** 2)
    ss_tot = np.sum((log_shares - log_shares.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Max absolute deviation between observed and theoretical shares
    max_dev = np.abs(observed_shares - theoretical_shares).max()
    mean_dev = np.abs(observed_shares - theoretical_shares).mean()

    # Chi-squared: observed counts vs theoretical frequencies
    chi2_stat_pw, chi2_p_pw = chisquare(
        counts_power, f_exp=theoretical_shares * counts_power.sum()
    )

    print()
    print("=" * 60)
    print(f"POWER-LAW VERIFICATION  (Zipf, α={power_exponent})")
    print("=" * 60)
    print(f"  Weights: w(k) = k^(-α), exact by construction")
    print(f"  Log-log slope          : {slope:.6f}  (expected: {-power_exponent:.6f})")
    print(f"  Log-log R²             : {r2:.6f}")
    if abs(slope - (-power_exponent)) < 0.05 and r2 > 0.999:
        print("  ✓ Slope matches -α exactly and R² ≈ 1 (perfect power law)")
    else:
        print("  ✗ Slope or R² deviated — check implementation")
    print(f"  Max |observed - theoretical| share : {max_dev:.6f}")
    print(f"  Mean |observed - theoretical| share: {mean_dev:.6f}")
    print(f"  Chi-squared p-value (vs Zipf)      : {chi2_p_pw:.4f}")
    if chi2_p_pw > 0.05:
        print("  ✓ Observed frequencies consistent with Zipf(α) weights (p > 0.05)")
    else:
        print("  ✗ Observed frequencies deviate from Zipf(α) — increase n_samples")

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
    colors_u = ['tomato' if d < 0 else 'seagreen' for d in deviations]
    ax.bar(trader_idx, deviations * 100, color=colors_u, alpha=0.8)
    ax.axhline(0, color='black', lw=1)
    ax.set_title("Uniform – deviation from expected (%)")
    ax.set_xlabel("Trader index")
    ax.set_ylabel("Δ share (pp)")

    # Power-law: rank–share log-log with exact theoretical line
    ax = axes[1, 0]
    ax.scatter(ranks, observed_shares, color='darkorange', zorder=3, label='Observed shares')
    ax.plot(ranks, theoretical_shares, 'k--', lw=2,
            label=f'Theoretical Zipf(α={power_exponent})')
    fit_line = np.exp(intercept) * ranks ** slope
    ax.plot(ranks, fit_line, 'b:', lw=1.5,
            label=f'Log-log fit  slope={slope:.3f}  R²={r2:.4f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f"Power-law – rank vs share (log-log)\n"
                 f"slope={slope:.4f}  expected={-power_exponent:.4f}  R²={r2:.4f}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Share")
    ax.legend(fontsize=8)

    # Power-law: observed vs theoretical shares bar comparison
    ax = axes[1, 1]
    width = 0.4
    ax.bar(ranks - width/2, observed_shares, width, color='darkorange', alpha=0.8, label='Observed')
    ax.bar(ranks + width/2, theoretical_shares, width, color='navy', alpha=0.6, label='Theoretical Zipf')
    ax.set_yscale('log')
    ax.set_title(f"Power-law – observed vs theoretical shares\n(χ² p={chi2_p_pw:.3f})")
    ax.set_xlabel("Rank (trader index)")
    ax.set_ylabel("Share (log scale)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("images\\distribution_verification.png", dpi=150)
    plt.show()
    print("\nPlot saved to distribution_verification.png")


if __name__ == "__main__":
    verify_distributions(nb_traders=20, n_samples=10_000_000, power_exponent=2.0)