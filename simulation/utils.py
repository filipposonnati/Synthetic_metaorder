import numpy as np
import pandas as pd
from pathlib import Path

def save_simulated_data(path, prices, volumes):
    """
    Salva i dati simulati in un file CSV.
    Formato: indice, prezzo, volume_in_modulo, segno_volume
    """

    file_path = Path(path)
    
    if file_path.exists():
        file_path.unlink()

    indices = np.arange(len(prices))
    abs_volumes = np.abs(volumes)
    signs = np.sign(volumes)
    
    df_sim = pd.DataFrame({
        'indice': indices,
        'prezzo': prices,
        'volume_in_modulo': abs_volumes,
        'segno_volume': signs
    })

    df_sim.to_csv(path, index=False, header=False)

def clear_data(model):
    dir = Path(f"..\\database\\data_{model}")

    for file in dir.iterdir():
        if file.is_file():
            file.unlink()

def open_data(path):
    trades = pd.read_csv(f"..\\database\\data\\{path}", header=None)
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

    return prices, volumes, signs
