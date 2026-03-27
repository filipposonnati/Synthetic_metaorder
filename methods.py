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

    trades.sort_values('timestamp')

    trades['TradedVolume'] = trades['quantity'].cumsum()
    
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
 
    sorted_trades = sorted_trades.dropna()
    
    # Calcola l'impatto parziale come differenza tra il prezzo di fine trade corrente e il prezzo iniziale del metaordine
    sorted_trades['PartialImpact'] = (sorted_trades['EndMid'] - start_prices) * sorted_trades['sign']

    sorted_trades['PartialImpact_pre'] = (sorted_trades['BeginMid'] - start_prices) * sorted_trades['sign']

    # Volume traded by the trader until a certain point
    sorted_trades['PartialVolume'] = sorted_trades.groupby('metaid')['Volume'].cumsum()

    metaorders_agg = sorted_trades.groupby('metaid').agg({
        'sign':'first', 
        'BeginMid':'first', 
        'EndMid':'last',
        'trader':'count', 
        'BeginTime':'first', 
        'EndTime':'last',
        'Volume':'sum',
        'TradedVolume':'last',
        'DailyVolume': 'first'
    }).reset_index()

    # Number of children
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

    sorted_trades.drop(columns=['EndTime', 'day', 'trader'], inplace=True)

    metaorders_agg['MetaImpact'] = (metaorders_agg['EndMid'] - metaorders_agg['BeginMid']) * metaorders_agg['sign']

    metaorders_agg = metaorders_agg.rename(columns={
        'trader': 'NbChild', 'Volume':'MetaVolume', 'Impact': 'MetaImpact'
    })

    metaorders_agg['Ratio'] = metaorders_agg['MetaImpact'] / np.sqrt(metaorders_agg['MetaVolume'])

    #sorted_trades = sorted_trades.groupby('metaid').apply(lambda x: x.iloc[2:])

    return metaorders_agg, sorted_trades

def mapping_function(trades, nb_traders, kind, alpha = 0.0) :
    '''
    Given a trading frequency distribution, generate a list of traders for each trade
    Inputs :
    trades : DataFrame with trades
    nb_traders : number of traders  
    kind : type of distribution for the trader's frequency
    alpha : exponent of the distribution (if kind = 'power')

    Outputs :
    traders : list of traders for each trade
    '''
    ### Choose a trading frequency distribution
    if kind == 'power':
        samples = powerlaw.rvs(int(alpha), size=int(nb_traders))
    if kind =='uniform':
        samples = np.ones(nb_traders)
    frequencies = samples / samples.sum()
    cum_freq = np.cumsum(frequencies)
    traders = []

    ### Assign traders to trades
    for _ in range (len(trades)):
        u = random.random()
        trader_index = np.searchsorted(cum_freq,u)
        traders.append(f"Trader {trader_index+1}")
    return traders