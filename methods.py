import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random, datetime
from scipy.stats import powerlaw as sp_powerlaw, pareto as sp_pareto, chisquare, kstest
from os import listdir

"""
UNIFIED METHODS MODULE
======================

Three power-law distribution types:
  1. kind='power'         → Zipf (deterministic): w(k) = k^(-alpha) [EXACT]
  2. kind='powerpareto'   → Pareto (stochastic): samples from Pareto(alpha) [RANDOM]
  3. kind='powerscipy'    → PowerLaw (stochastic): samples from scipy.stats.powerlaw(alpha) on [0,1] [RANDOM]
  4. kind='uniform'       → Uniform distribution (all traders equal weight)

Usage:
  metaorders_agg, sorted_trades = generate(path, nb_traders, kind='power', exponent=2.0)
  agg_slim = generate_slim(path, nb_traders, kind='powerpareto', exponent=2.0)
  verify_distributions(nb_traders=20, n_samples=10_000_000, kind='powerscipy', power_exponent=2.0)
"""

# ============================================================================
#  CORE FUNCTIONS: generate() and generate_slim()
# ============================================================================

def generate(path, nb_traders, kind, exponent, start_id=0, data_dir='database\\data'):
    '''
    Generate metaorders from trades, and compute the partial impact 
    (intra-metaorder impact) and partial volume (intra-metaorder volume)
    for each child trade and then normalizes these partial features by the metaorder total.
    
    Parameters
    ----------
    path : str
        Path to the CSV file (relative to data_dir)
    nb_traders : int
        Number of traders
    kind : str
        'power', 'powerpareto', 'powerscipy', or 'uniform'
    exponent : float
        Alpha parameter for the power law / Pareto distribution
    start_id : int
        Starting metaorder ID (default 0)
    data_dir : str
        Base directory for data files (default 'database\\data')
    
    Returns
    -------
    metaorders_agg : pd.DataFrame
        Aggregated metaorder statistics
    sorted_trades : pd.DataFrame
        Individual trades with computed features
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

    # Create DataFrame
    trades = pd.DataFrame({
        'timestamp': timestamp,
        'sign': signs,
        'quantity': volumes,
        'price': prices
    })

    trades['day'] = trades['timestamp'].dt.date
    trades.sort_values('timestamp', inplace=True)
    
    # Calculate daily volatility and volume
    daily_stats = trades.groupby('day').agg(
        DailySigma=('price', lambda x: (x.max() - x.min()) / x.iloc[0]),
        DailyVolume=('quantity', 'sum')
    ).reset_index()

    trades = trades.merge(daily_stats, on='day', how='left')
    trades['BeginMid'] = np.log(trades['price']) / trades['DailySigma']
    trades['EndTime'] = trades['timestamp']
    trades['EndMid'] = trades['BeginMid'].shift(-1)

    trades.drop(trades.tail(1).index, inplace=True)

    trades['Volume'] = trades['quantity'] / trades['DailyVolume']
    trades['TransactionTime'] = trades['Volume'].cumsum()

    trades.drop(columns=['price', 'quantity', 'DailySigma'], inplace=True)

    trades = trades.rename(columns={'timestamp': 'BeginTime'})

    # Attribute traders to trades
    trades['trader'] = mapping_function(trades, nb_traders, kind, exponent)

    # Identify Metaorders
    sorted_trades = trades.sort_values(['trader', 'BeginTime']).reset_index(drop=True)
    
    # Calculate metaid
    sorted_trades['metaid'] = np.where(
        (sorted_trades['trader'] != sorted_trades['trader'].shift()) | 
        (sorted_trades.sign.shift() != sorted_trades.sign) | 
        (sorted_trades.day != sorted_trades.day.shift()), 
        1, 0
    ).cumsum() + start_id

    start_cum_volume = sorted_trades.groupby('metaid')['TransactionTime'].transform('first')
    start_volume = sorted_trades.groupby('metaid')['Volume'].transform('first')

    # Volume traded by everyone until a certain point
    sorted_trades['TradedVolume'] = sorted_trades['TransactionTime'] - start_cum_volume + start_volume
    sorted_trades['TradedVolume'] = sorted_trades['TradedVolume'] / sorted_trades.groupby('metaid')['TradedVolume'].transform('last')

    # Partial impact
    start_prices = sorted_trades.groupby('metaid')['BeginMid'].transform('first')
    sorted_trades['PartialImpact'] = (sorted_trades['EndMid'] - start_prices) * sorted_trades['sign']
    sorted_trades['PartialImpact_pre'] = (sorted_trades['BeginMid'] - start_prices) * sorted_trades['sign']

    # Partial volume
    sorted_trades['PartialVolume'] = sorted_trades.groupby('metaid')['Volume'].cumsum()
    sorted_trades['EndTransactionTime'] = sorted_trades['TransactionTime']

    # Aggregate metaorders
    metaorders_agg = sorted_trades.groupby('metaid').agg({
        'sign': 'first', 
        'BeginMid': 'first', 
        'EndMid': 'last',
        'trader': 'first',
        'BeginTime': 'first', 
        'EndTime': 'last',
        'Volume': 'sum',
        'TradedVolume': 'last',
        'DailyVolume': 'first',
        'TransactionTime': 'first',
        'EndTransactionTime': 'last'
    }).reset_index()

    nb_child = sorted_trades.groupby('metaid').size().reset_index(name='NbChild')
    metaorders_agg = metaorders_agg.merge(nb_child, on='metaid')

    sorted_trades['NbChild'] = sorted_trades.groupby('metaid')['trader'].transform('count')
    sorted_trades['MetaVolume'] = sorted_trades.groupby('metaid')['PartialVolume'].transform('last')

    begintime = sorted_trades.groupby('metaid')['BeginTime'].transform('first')
    endtime = sorted_trades.groupby('metaid')['EndTime'].transform('last')

    sorted_trades['MetaDuration'] = (endtime - begintime).dt.total_seconds()
    sorted_trades['ElapsedTime'] = (sorted_trades['BeginTime'] - begintime).dt.total_seconds()

    sorted_trades['Ratio'] = sorted_trades['PartialImpact'] / np.sqrt(sorted_trades['MetaVolume'])
    sorted_trades['Ratio_pre'] = sorted_trades['PartialImpact_pre'] / np.sqrt(sorted_trades['MetaVolume'])

    sorted_trades.drop(columns=['EndTime', 'day', 'EndTransactionTime'], inplace=True)

    metaorders_agg['MetaImpact'] = (metaorders_agg['EndMid'] - metaorders_agg['BeginMid']) * metaorders_agg['sign']

    metaorders_agg = metaorders_agg.rename(columns={
        'Volume': 'MetaVolume', 'Impact': 'MetaImpact', 'TransactionTime': 'BeginTransactionTime'
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
 
    prices = np.array(trades[1])
    volumes = np.array(trades[2])
    signs = np.array(trades[3])
 
    trades = pd.DataFrame({
        'timestamp': timestamp,
        'sign': signs,
        'quantity': volumes,
        'price': prices,
    })
    trades['day'] = trades['timestamp'].dt.date
    trades.sort_values('timestamp', inplace=True)
 
    # Daily normalisation factors
    daily_stats = trades.groupby('day').agg(
        DailySigma=('price', lambda x: (x.max() - x.min()) / x.iloc[0]),
        DailyVolume=('quantity', 'sum'),
    ).reset_index()
    trades = trades.merge(daily_stats, on='day', how='left')
 
    # Normalised price (only first/last per metaorder needed)
    trades['Mid'] = np.log(trades['price']) / trades['DailySigma']
    trades['Volume'] = trades['quantity'] / trades['DailyVolume']
 
    # Assign traders and build metaorder ids
    trades['trader'] = mapping_function(trades, nb_traders, kind, exponent)
    trades = trades.sort_values(['trader', 'timestamp']).reset_index(drop=True)
 
    trades['metaid'] = np.where(
        (trades['trader'] != trades['trader'].shift()) |
        (trades['sign'] != trades['sign'].shift()) |
        (trades['day'] != trades['day'].shift()),
        1, 0
    ).cumsum()
 
    # Aggregate: only what is needed
    agg = trades.groupby('metaid').agg(
        sign=('sign', 'first'),
        BeginMid=('Mid', 'first'),
        EndMid=('Mid', 'last'),
        MetaVolume=('Volume', 'sum'),
        NbChild=('trader', 'count'),
    ).reset_index(drop=True)
 
    agg['MetaImpact'] = (agg['EndMid'] - agg['BeginMid']) * agg['sign']
 
    return agg[['MetaVolume', 'MetaImpact', 'NbChild']]


# ============================================================================
#  MAPPING FUNCTION: Unified for all distribution types
# ============================================================================

def mapping_function(trades, nb_traders, kind='power', alpha=0.0):
    """
    Assign traders to trades based on the specified distribution.
    
    Parameters
    ----------
    trades : pd.DataFrame
        DataFrame containing trades (must have 'timestamp' and 'sign' columns)
    nb_traders : int
        Number of traders
    kind : str
        Type of distribution:
        - 'power': Zipf (deterministic) w(k) = k^(-alpha)
        - 'powerpareto': Pareto (stochastic) samples from Pareto(alpha)
        - 'powerscipy': PowerLaw scipy (stochastic) samples from scipy.stats.powerlaw(alpha)
        - 'uniform': Uniform (all traders equal weight)
    alpha : float
        Exponent for power-law distributions (default 0.0 → 1.5)
    
    Returns
    -------
    list
        List of trader names assigned to each trade
    """
    if kind == 'power':
        # ====== ZIPF (Deterministic) ======
        shape = float(alpha) if float(alpha) > 0 else 1.5
        ranks = np.arange(1, int(nb_traders) + 1)
        samples = ranks.astype(float) ** (-shape)   # w(k) = k^(-alpha)
        
    elif kind == 'powerpareto':
        # ====== PARETO (Stochastic) ======
        shape = float(alpha) if float(alpha) > 0 else 1.5
        samples = (np.random.pareto(shape, int(nb_traders)) + 1)
        samples = np.sort(samples)[::-1]  # Sort descending
        
    elif kind == 'powerscipy':
        # ====== SCIPY POWERLAW (Stochastic) ======
        shape = float(alpha) if float(alpha) > 0 else 1.5
        samples = sp_powerlaw.rvs(shape, size=int(nb_traders))
        samples = np.sort(samples)[::-1]  # Sort descending
        
    elif kind == 'uniform':
        # ====== UNIFORM ======
        samples = np.ones(int(nb_traders))
    else:
        # Default to uniform
        samples = np.ones(int(nb_traders))

    probabilities = samples / samples.sum()
    
    # Sample traders based on probabilities
    trader_indices = np.arange(int(nb_traders))
    chosen_indices = np.random.choice(trader_indices, size=len(trades), p=probabilities)
    
    return [f"Trader {i+1}" for i in chosen_indices]


# ============================================================================
#  VERIFICATION FUNCTION: Unified for all distribution types
# ============================================================================

def verify_distributions(nb_traders=20, n_samples=500_000, kind='power', power_exponent=2.0):
    """
    Verify that mapping_function produces the expected distributions.
    
    Parameters
    ----------
    nb_traders : int
        Number of traders
    n_samples : int
        Number of trade assignments to draw
    kind : str
        Type of distribution: 'power', 'powerpareto', 'powerscipy', 'uniform'
    power_exponent : float
        Exponent α for power-law distributions
    """
    
    dummy_trades = pd.DataFrame({
        'timestamp': range(n_samples),
        'sign': np.ones(n_samples)
    })
    labels = [f"Trader {i+1}" for i in range(nb_traders)]

    print(f"\n{'='*70}")
    print(f"DISTRIBUTION VERIFICATION: kind='{kind}'  |  α={power_exponent}  |  n={n_samples:,}")
    print(f"{'='*70}\n")

    # ================================================================ #
    #  1. UNIFORM (same for all distributions)                         #
    # ================================================================ #
    result_uniform = mapping_function(dummy_trades, nb_traders, 'uniform', 0.0)
    counts_uniform = np.array([result_uniform.count(l) for l in labels], dtype=float)
    shares_uniform = counts_uniform / counts_uniform.sum()

    expected_uniform = np.full(nb_traders, 1.0 / nb_traders)
    chi2_stat_u, chi2_p_u = chisquare(counts_uniform, f_exp=counts_uniform.sum() * expected_uniform)

    print("─" * 70)
    print("1. UNIFORM DISTRIBUTION (reference)")
    print("─" * 70)
    print(f"  χ² statistic    : {chi2_stat_u:.4f}")
    print(f"  p-value         : {chi2_p_u:.4f}")
    if chi2_p_u > 0.05:
        print("  ✓ Cannot reject uniformity (p > 0.05)")
    else:
        print("  ✗ Significant deviation from uniformity (p ≤ 0.05)")
    print(f"  Min share : {shares_uniform.min():.4f}  |  "
          f"Max share : {shares_uniform.max():.4f}  |  "
          f"Expected : {1/nb_traders:.4f}")

    # ================================================================ #
    #  2. POWER-LAW (varies by type)                                   #
    # ================================================================ #
    result_power = mapping_function(dummy_trades, nb_traders, kind, power_exponent)
    counts_power = np.array([result_power.count(l) for l in labels], dtype=float)
    shares_power = counts_power / counts_power.sum()

    observed_shares_sorted = np.sort(shares_power)[::-1]
    ranks = np.arange(1, nb_traders + 1)
    log_ranks = np.log(ranks)
    log_shares = np.log(observed_shares_sorted)
    slope, intercept = np.polyfit(log_ranks, log_shares, 1)
    ss_res = np.sum((log_shares - (slope * log_ranks + intercept)) ** 2)
    ss_tot = np.sum((log_shares - log_shares.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print("\n" + "─" * 70)
    print(f"2. POWER-LAW DISTRIBUTION  (kind='{kind}')")
    print("─" * 70)

    if kind == 'power':
        # ============================================================ #
        # ZIPF (Deterministic)
        # ============================================================ #
        # Theoretical Zipf weights (normalised)
        zipf_weights = ranks.astype(float) ** (-power_exponent)
        theoretical_shares = zipf_weights / zipf_weights.sum()

        max_dev = np.abs(observed_shares_sorted - theoretical_shares).max()
        mean_dev = np.abs(observed_shares_sorted - theoretical_shares).mean()

        chi2_stat_pw, chi2_p_pw = chisquare(
            counts_power, f_exp=theoretical_shares * counts_power.sum()
        )

        print(f"  Distribution   : Zipf (DETERMINISTIC)")
        print(f"  Weights        : w(k) = k^(-α), exact by construction")
        print(f"  Log-log slope  : {slope:.6f}  (expected: {-power_exponent:.6f})")
        print(f"  Log-log R²     : {r2:.6f}")
        if abs(slope - (-power_exponent)) < 0.05 and r2 > 0.999:
            print("  ✓ Slope matches -α exactly and R² ≈ 1 (perfect power law)")
        else:
            print("  ✗ Slope or R² deviated — check implementation")
        print(f"  Max dev        : {max_dev:.6f}")
        print(f"  Mean dev       : {mean_dev:.6f}")
        print(f"  χ² p-value     : {chi2_p_pw:.4f}")
        if chi2_p_pw > 0.05:
            print("  ✓ Observed frequencies consistent with Zipf(α) (p > 0.05)")
        else:
            print("  ✗ Observed frequencies deviate from Zipf(α) — increase n_samples")

    elif kind == 'powerpareto':
        # ============================================================ #
        # PARETO (Stochastic)
        # ============================================================ #
        fit_alpha, fit_loc, fit_scale = sp_pareto.fit(observed_shares_sorted, floc=0)
        ks_stat, ks_p = kstest(
            observed_shares_sorted,
            lambda x: sp_pareto.cdf(x, fit_alpha, loc=fit_loc, scale=fit_scale)
        )

        print(f"  Distribution   : Pareto (STOCHASTIC)")
        print(f"  Sampler        : np.random.pareto(α)")
        print(f"  Fitted α (KS)  : {fit_alpha:.4f}  (input: {power_exponent:.4f})")
        print(f"  KS statistic   : {ks_stat:.4f}")
        print(f"  KS p-value     : {ks_p:.4f}")
        if ks_p > 0.05:
            print("  ✓ Cannot reject Pareto fit (p > 0.05)")
        else:
            print("  ✗ Significant deviation from Pareto (p ≤ 0.05)")
        print(f"  Log-log slope  : {slope:.4f}  (R² = {r2:.4f})")
        if r2 > 0.95:
            print("  ✓ Strong log-log linearity (R² > 0.95)")
        else:
            print("  ✗ Weak log-log linearity — distribution may not be power-law")

    elif kind == 'powerscipy':
        # ============================================================ #
        # SCIPY POWERLAW (Stochastic)
        # ============================================================ #
        weights_rescaled = observed_shares_sorted / observed_shares_sorted.max()
        fit_alpha, fit_loc, fit_scale = sp_powerlaw.fit(weights_rescaled, floc=0, fscale=1)
        ks_stat, ks_p = kstest(
            weights_rescaled,
            lambda x: sp_powerlaw.cdf(x, fit_alpha, loc=fit_loc, scale=fit_scale)
        )

        # Expected mean and variance of powerlaw(alpha) on [0,1]
        expected_mean = power_exponent / (power_exponent + 1)
        expected_var = power_exponent / ((power_exponent + 2) * (power_exponent + 1) ** 2)

        print(f"  Distribution   : scipy.stats.powerlaw (STOCHASTIC)")
        print(f"  Sampler        : scipy.stats.powerlaw.rvs(α) on [0,1]")
        print(f"  PDF            : f(x) = α·x^(α-1)")
        print(f"  Expected mean  : {expected_mean:.4f}  |  Observed: {weights_rescaled.mean():.4f}")
        print(f"  Expected var   : {expected_var:.6f}  |  Observed: {weights_rescaled.var():.6f}")
        print(f"  Fitted α (KS)  : {fit_alpha:.4f}  (input: {power_exponent:.4f})")
        print(f"  KS statistic   : {ks_stat:.4f}")
        print(f"  KS p-value     : {ks_p:.4f}")
        if ks_p > 0.05:
            print("  ✓ Cannot reject powerlaw fit (p > 0.05)")
        else:
            print("  ✗ Significant deviation from powerlaw(α) (p ≤ 0.05)")
        print(f"  Log-log slope  : {slope:.4f}  (R² = {r2:.4f})")
        if r2 > 0.90:
            print("  ✓ Strong log-log linearity (R² > 0.90)")
        else:
            print("  ✗ Weak log-log linearity")

    # ================================================================ #
    #  3. PLOTS - TYPE-SPECIFIC VISUALIZATIONS                        #
    # ================================================================ #
    
    if kind == 'power':
        # ============================================================ #
        # ZIPF (Deterministic) - 4 plots
        # ============================================================ #
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Distribution Verification: ZIPF (Deterministic)  α={power_exponent}", 
                     fontsize=14)

        trader_idx = np.arange(1, nb_traders + 1)

        # Plot 1: Uniform bar chart
        ax = axes[0, 0]
        ax.bar(trader_idx, shares_uniform, color='steelblue', alpha=0.8, label='Observed')
        ax.axhline(1 / nb_traders, color='crimson', lw=2, linestyle='--', label='Expected')
        ax.set_title(f"Uniform distribution (reference) (χ² p={chi2_p_u:.3f})")
        ax.set_xlabel("Trader index")
        ax.set_ylabel("Share")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Uniform deviation
        ax = axes[0, 1]
        deviations = shares_uniform - 1 / nb_traders
        colors = ['tomato' if d < 0 else 'seagreen' for d in deviations]
        ax.bar(trader_idx, deviations * 100, color=colors, alpha=0.8)
        ax.axhline(0, color='black', lw=1)
        ax.set_title("Uniform – deviation from expected (%)")
        ax.set_xlabel("Trader index")
        ax.set_ylabel("Δ share (pp)")
        ax.grid(True, alpha=0.3)

        # Plot 3: Zipf log-log (observed vs theoretical)
        ax = axes[1, 0]
        ax.scatter(ranks, observed_shares_sorted, color='darkorange', zorder=3, s=60, 
                   label='Observed shares')
        fit_line = np.exp(intercept) * ranks ** slope
        ax.plot(ranks, fit_line, 'b--', lw=2,
                label=f'Log-log fit: slope={slope:.4f}  R²={r2:.4f}')
        ax.plot(ranks, theoretical_shares, 'k-', lw=2.5, 
                label=f'Theoretical Zipf(α={power_exponent})')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"Log-log: Observed vs Theoretical Zipf (slope={slope:.4f}  expected={-power_exponent:.4f})")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Share")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        # Plot 4: Zipf bar chart (observed vs theoretical)
        ax = axes[1, 1]
        width = 0.35
        ax.bar(ranks - width/2, observed_shares_sorted, width, color='darkorange', 
               alpha=0.8, label='Observed')
        ax.bar(ranks + width/2, theoretical_shares, width, color='navy', alpha=0.6, 
               label='Theoretical Zipf')
        ax.set_title(f"Observed vs Theoretical Shares (χ² p={chi2_p_pw:.3f})")
        ax.set_yscale('log')
        ax.set_xlabel("Rank")
        ax.set_ylabel("Share (log scale)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        filename = f"images\\distribution_verification_{kind}.png"
        plt.savefig(filename, dpi=150)
        print(f"\n✓ Plot saved to: {filename}")
        plt.show()

    elif kind == 'powerpareto':
        # ============================================================ #
        # PARETO (Stochastic) - 4 plots
        # ============================================================ #
        # Generate raw samples to visualize the sampling distribution
        raw_samples = (np.random.pareto(power_exponent, 100000) + 1)
        raw_samples_sorted_asc = np.sort(raw_samples)  # ASCENDING for CDF
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Distribution Verification: PARETO (Stochastic)  α={power_exponent}", 
                     fontsize=14)

        trader_idx = np.arange(1, nb_traders + 1)

        # Plot 1: Raw samples histogram vs Pareto PDF
        ax = axes[0, 0]
        ax.hist(raw_samples, bins=80, density=True, alpha=0.6, color='steelblue', label='Raw samples')
        # Theoretical Pareto PDF: f(x) = α / (1+x)^(α+1)
        x_vals = np.linspace(0, raw_samples.max(), 500)
        pdf_pareto = sp_pareto.pdf(x_vals, power_exponent, loc=0, scale=1)
        ax.plot(x_vals, pdf_pareto, 'crimson', lw=2.5, label=f'Pareto PDF (α={power_exponent})')
        ax.set_title("Raw Samples vs Pareto PDF (before rank-sorting)")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: KS test on raw samples
        ax = axes[0, 1]
        fit_alpha_raw, fit_loc_raw, fit_scale_raw = sp_pareto.fit(raw_samples, floc=0)
        empirical_cdf = np.arange(1, len(raw_samples_sorted_asc) + 1) / len(raw_samples_sorted_asc)
        theoretical_cdf = sp_pareto.cdf(raw_samples_sorted_asc, fit_alpha_raw, 
                                       loc=fit_loc_raw, scale=fit_scale_raw)
        ax.plot(raw_samples_sorted_asc, empirical_cdf, 'o-', color='steelblue', alpha=0.6, 
                markersize=3, label='Empirical CDF')
        ax.plot(raw_samples_sorted_asc, theoretical_cdf, 'crimson', lw=2, 
                label=f'Theoretical Pareto(α={fit_alpha_raw:.3f})')
        ax.set_title(f"CDF Comparison: KS test (KS stat={ks_stat:.4f}, p={ks_p:.4f})")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Cumulative Probability")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Log-log rank-size (sorted empirical distribution)
        ax = axes[1, 0]
        ax.scatter(ranks, observed_shares_sorted, color='darkorange', zorder=3, s=60, 
                   label='Observed shares (rank-sorted)')
        fit_line = np.exp(intercept) * ranks ** slope
        ax.plot(ranks, fit_line, 'b--', lw=2,
                label=f'Log-log fit: slope={slope:.4f}  R²={r2:.4f}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("Log-log Rank-Size Plot (tests power-law behavior after ranking)")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Share")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        # Plot 4: Fitted vs observed shares (sorted)
        ax = axes[1, 1]
        ax.bar(trader_idx, observed_shares_sorted, color='darkorange', alpha=0.8, 
               label='Observed (this run)')
        ax.set_title("Empirical Trader Shares (Ranked) (this instance varies; seed for reproducibility)")
        ax.set_yscale('log')
        ax.set_xlabel("Rank")
        ax.set_ylabel("Share (log scale)")
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        filename = f"images\\distribution_verification_{kind}.png"
        plt.savefig(filename, dpi=150)
        print(f"\n✓ Plot saved to: {filename}")
        plt.show()

    elif kind == 'powerscipy':
        # ============================================================ #
        # SCIPY POWERLAW (Stochastic) - 4 plots
        # ============================================================ #
        # Generate raw samples to visualize the sampling distribution
        raw_samples = sp_powerlaw.rvs(power_exponent, size=100000)
        raw_samples_sorted_asc = np.sort(raw_samples)  # ASCENDING for CDF
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Distribution Verification: SCIPY POWERLAW (Stochastic)  α={power_exponent}", 
                     fontsize=14)

        trader_idx = np.arange(1, nb_traders + 1)

        # Plot 1: Raw samples histogram vs PowerLaw PDF
        ax = axes[0, 0]
        ax.hist(raw_samples, bins=80, density=True, alpha=0.6, color='steelblue', label='Raw samples')
        # Theoretical PowerLaw PDF: f(x) = α * x^(α-1)  on [0,1]
        x_vals = np.linspace(0, 1, 500)
        pdf_powerlaw = sp_powerlaw.pdf(x_vals, power_exponent)
        ax.plot(x_vals, pdf_powerlaw, 'crimson', lw=2.5, label=f'PowerLaw PDF (α={power_exponent})')
        ax.set_title("Raw Samples vs PowerLaw PDF (before rank-sorting)")
        ax.set_xlabel("Weight (on [0,1])")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: KS test on raw samples
        ax = axes[0, 1]
        weights_rescaled_test = raw_samples_sorted_asc / raw_samples_sorted_asc.max()
        empirical_cdf = np.arange(1, len(weights_rescaled_test) + 1) / len(weights_rescaled_test)
        theoretical_cdf = sp_powerlaw.cdf(weights_rescaled_test, power_exponent)
        ax.plot(weights_rescaled_test, empirical_cdf, 'o-', color='steelblue', alpha=0.6, 
                markersize=3, label='Empirical CDF')
        ax.plot(weights_rescaled_test, theoretical_cdf, 'crimson', lw=2, 
                label=f'Theoretical PowerLaw(α={power_exponent})')
        ax.set_title(f"CDF Comparison: KS test (KS stat={ks_stat:.4f}, p={ks_p:.4f})")
        ax.set_xlabel("Weight (rescaled)")
        ax.set_ylabel("Cumulative Probability")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Log-log rank-size (sorted empirical distribution)
        ax = axes[1, 0]
        ax.scatter(ranks, observed_shares_sorted, color='darkorange', zorder=3, s=60, 
                   label='Observed shares (rank-sorted)')
        fit_line = np.exp(intercept) * ranks ** slope
        ax.plot(ranks, fit_line, 'b--', lw=2,
                label=f'Log-log fit: slope={slope:.4f}  R²={r2:.4f}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("Log-log Rank-Size Plot (tests power-law behavior after ranking)")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Share")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        # Plot 4: Expected vs observed statistics
        ax = axes[1, 1]
        expected_mean = power_exponent / (power_exponent + 1)
        expected_var = power_exponent / ((power_exponent + 2) * (power_exponent + 1) ** 2)
        observed_mean = raw_samples.mean()
        observed_var = raw_samples.var()
        
        metrics = ['Mean', 'Variance']
        expected_vals = [expected_mean, expected_var]
        observed_vals = [observed_mean, observed_var]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        ax.bar(x_pos - width/2, expected_vals, width, color='navy', alpha=0.6, label='Theoretical')
        ax.bar(x_pos + width/2, observed_vals, width, color='darkorange', alpha=0.8, label='Observed')
        ax.set_title("Expected vs Observed Statistics (from raw samples)")
        ax.set_ylabel("Value")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add text annotations
        for i, (exp, obs) in enumerate(zip(expected_vals, observed_vals)):
            pct_diff = 100 * abs(obs - exp) / exp if exp != 0 else 0
            ax.text(i, max(exp, obs) * 1.1, f'{pct_diff:.1f}%', ha='center', fontsize=9)

        plt.tight_layout()
        filename = f"images\\distribution_verification_{kind}.png"
        plt.savefig(filename, dpi=150)
        print(f"\n✓ Plot saved to: {filename}")
        plt.show()


# ============================================================================
#  MAIN / TESTING
# ============================================================================

if __name__ == "__main__":
    # Test all three distribution types
    
    print("\n" + "="*70)
    print("TESTING ALL THREE DISTRIBUTIONS")
    print("="*70)
    
    # Test 1: Zipf (Deterministic)
    print("\n\n### TEST 1: ZIPF DETERMINISTIC ###")
    verify_distributions(
        nb_traders=20, 
        n_samples=10_000_000, 
        kind='power', 
        power_exponent=2.0
    )
    
    # Test 2: Pareto (Stochastic)
    print("\n\n### TEST 2: PARETO STOCHASTIC ###")
    verify_distributions(
        nb_traders=20, 
        n_samples=10_000_000, 
        kind='powerpareto', 
        power_exponent=2.0
    )
    
    # Test 3: PowerLaw scipy (Stochastic)
    print("\n\n### TEST 3: SCIPY POWERLAW STOCHASTIC ###")
    verify_distributions(
        nb_traders=20, 
        n_samples=10_000_000, 
        kind='powerscipy', 
        power_exponent=2.0
    )
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)