"""
meta_creation_lmf.py
====================
Variante di meta_creation.py per i dati sintetici generati da lmf_synthetic_market.py.

Invece di assegnare i trade a trader fittizi (methods.mapping_function), i metaordini
vengono ricostruiti dall'id LMF salvato nella quinta colonna dei CSV:
    tempo [s], prezzo, volume, segno, id metaordine LMF

Le grandezze sono calcolate esattamente come in methods.generate (stessa normalizzazione
giornaliera, stesso EndMid = BeginMid del trade successivo in ordine temporale, stesso
trade finale scartato) e il file di output ha le stesse colonne di meta_creation.py:

  metaid, sign, BeginMid, EndMid, trader, BeginTime, EndTime, MetaVolume, TradedVolume,
  DailyVolume, BeginTransactionTime, EndTransactionTime, NbChild, MetaImpact, Ratio

Differenze rispetto a meta_creation.py:
  - 'trader' vale sempre "LMF" (non esistono trader nel modello LMF);
  - i trade di rumore (id = -1) non appartengono a nessun metaordine e sono esclusi
    (include_noise=True li tratta come metaordini da un solo trade);
  - il campionamento e' deterministico, quindi non servono le iterazioni;
  - metaid e' univoco su tutti i giorni.
"""
import os
import re
import gc

import numpy as np
import pandas as pd

OUT_COLUMNS = ['metaid', 'sign', 'BeginMid', 'EndMid', 'trader', 'BeginTime', 'EndTime',
               'MetaVolume', 'TradedVolume', 'DailyVolume', 'BeginTransactionTime',
               'EndTransactionTime', 'NbChild', 'MetaImpact', 'Ratio']


def generate_lmf(path, data_dir, start_id=0, include_noise=False):
    """
    Ricostruisce i metaordini LMF di un giorno a partire dall'id salvato nei dati.

    Parameters
    ----------
    path : str
        Nome del file CSV (relativo a data_dir), con 5 colonne
        (tempo, prezzo, volume, segno, id metaordine).
    data_dir : str
        Cartella dei dati sintetici.
    start_id : int
        Offset per il metaid (i metaid del giorno vanno da start_id + 1).
    include_noise : bool
        Se True, ogni trade di rumore (id = -1) diventa un metaordine con un solo trade.

    Returns
    -------
    metaorders_agg : pd.DataFrame
        Statistiche aggregate dei metaordini, stesse colonne di methods.generate.
    """
    trades = pd.read_csv(os.path.join(data_dir, path), header=None)
    if trades.shape[1] < 5:
        raise ValueError(f"{path}: manca la colonna con l'id del metaordine LMF "
                         f"(rigenerare i dati con save_meta_ids=True)")

    date_match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', path)
    date_string = "-".join(date_match.groups()) if date_match else "2020-01-01"
    timestamp = pd.to_datetime(date_string) + pd.to_timedelta(np.array(trades[0]), unit='s')

    df = pd.DataFrame({
        'timestamp': timestamp,
        'sign': np.array(trades[3]),
        'quantity': np.array(trades[2]),
        'price': np.array(trades[1]),
        'lmf_id': np.array(trades[4]).astype(np.int64),
    })
    df['day'] = df['timestamp'].dt.date
    # ordinamento stabile: a parita' di timestamp resta l'ordine di generazione
    df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)

    # Normalizzazioni giornaliere (come in methods.generate)
    gp = df.groupby('day')['price']
    daily_sigma = (gp.transform('max') - gp.transform('min')) / gp.transform('first')
    df['DailyVolume'] = df.groupby('day')['quantity'].transform('sum')
    df['BeginMid'] = np.log(df['price']) / daily_sigma
    df['EndMid'] = df['BeginMid'].shift(-1)

    # l'ultimo trade non ha EndMid: scartato come in methods.generate
    df = df.iloc[:-1].copy()

    df['Volume'] = df['quantity'] / df['DailyVolume']
    df['TransactionTime'] = df['Volume'].cumsum()      # su tutti i trade, rumore incluso
    df['BeginTime'] = df['timestamp']
    df['EndTime'] = df['timestamp']
    df['trader'] = 'LMF'

    # Chiave del metaordine (le feature sopra sono gia' calcolate sull'intero flusso)
    noise = (df['lmf_id'] < 0).to_numpy()
    key = df['lmf_id'].to_numpy().copy()
    if include_noise:
        key[noise] = key.max() + 1 + np.arange(noise.sum())
        df['key'] = key
    else:
        df['key'] = key
        df = df[~noise]

    if len(df) == 0:
        return pd.DataFrame(columns=OUT_COLUMNS)

    df = df.sort_values('key', kind='stable')           # dentro ogni metaordine: ordine temporale
    df['metaid'] = df.groupby('key').ngroup() + 1 + start_id

    # Frazione di volume scambiato da tutti, relativa all'inizio del metaordine
    g = df.groupby('metaid')
    start_cum_volume = g['TransactionTime'].transform('first')
    start_volume = g['Volume'].transform('first')
    df['TradedVolume'] = df['TransactionTime'] - start_cum_volume + start_volume
    df['TradedVolume'] = df['TradedVolume'] / df.groupby('metaid')['TradedVolume'].transform('last')
    df['EndTransactionTime'] = df['TransactionTime']

    metaorders_agg = df.groupby('metaid').agg({
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
        'EndTransactionTime': 'last',
    }).reset_index()

    nb_child = df.groupby('metaid').size().reset_index(name='NbChild')
    metaorders_agg = metaorders_agg.merge(nb_child, on='metaid')

    metaorders_agg['MetaImpact'] = (metaorders_agg['EndMid'] - metaorders_agg['BeginMid']) \
        * metaorders_agg['sign']
    metaorders_agg = metaorders_agg.rename(columns={
        'Volume': 'MetaVolume', 'TransactionTime': 'BeginTransactionTime'
    })
    metaorders_agg['Ratio'] = metaorders_agg['MetaImpact'] / np.sqrt(metaorders_agg['MetaVolume'])

    return metaorders_agg[OUT_COLUMNS]


if __name__ == '__main__':
    model = 'lmf_1.5_0.3_empirical_tim_0.0_1000'
    n_filter = 2              # tiene i metaordini con NbChild >= n_filter
    include_noise = False     # True: i trade di rumore (id -1) diventano metaordini da 1 trade

    data_dir = 'database\\data'
    if model != "":
        data_dir = data_dir + "_" + model

    meta_dir = 'database\\meta'
    if model != "":
        meta_dir = meta_dir + "_" + model

    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)

    paths = np.array(sorted(os.listdir(data_dir)))

    filename = 'meta_lmf'
    if n_filter > 2:
        filename = filename + f'_{n_filter}'
    filename = filename + '.csv'

    file_path = os.path.join(meta_dir, filename)

    # Skip se il file di output esiste gia'
    if os.path.exists(file_path):
        print(f"Skipping — {filename} already exists.")
        exit()

    print('LMF metaorders from ids')

    l = 0
    first = True

    for path in paths:
        print(path)
        meta = generate_lmf(path, data_dir, start_id=l, include_noise=include_noise)
        l += len(meta)                          # metaid univoci tra giorni (prima del filtro)
        meta = meta[meta['NbChild'] >= n_filter]

        meta.to_csv(file_path, mode='a', index=False, header=first)
        first = False

        del meta
        gc.collect()