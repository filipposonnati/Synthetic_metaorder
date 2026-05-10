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
        # Genera i pesi e ORDINALi in modo decrescente
        samples = (np.random.pareto(shape, int(nb_traders)) + 1)
        samples = np.sort(samples)[::-1] # I trader iniziali avranno pesi maggiori
    elif kind == 'uniform':
        samples = np.ones(int(nb_traders))
    else:
        samples = np.ones(int(nb_traders))

    probabilities = samples / samples.sum()
    
    # Usa gli indici per il campionamento per mantenere l'ordine nel plot
    trader_indices = np.arange(int(nb_traders))
    chosen_indices = np.random.choice(trader_indices, size=len(trades), p=probabilities)
    
    return [f"Trader {i+1}" for i in chosen_indices]

if __name__ == "__main__":
    nb_traders = 20
    result = mapping_function(np.full(10000000, 0.0), nb_traders, 'uniform', 2.0)
    
    # Ordiniamo le etichette per l'asse X (Trader 1, Trader 2, ...)
    labels = [f"Trader {i+1}" for i in range(nb_traders)]
    counts = [result.count(l) for l in labels]
    
    plt.bar(labels, counts)
    plt.xticks(rotation=45)
    plt.show()