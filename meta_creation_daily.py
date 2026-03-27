import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random, datetime
from scipy.stats import powerlaw
import os
import shutil
from os import listdir
import methods
from pathlib import Path

paths = np.array(listdir('database\\data'))

nb_traders = 4
kind = 'power'
exponent = 2.0

print(nb_traders, kind, exponent)

if kind == 'power':
    filename = f'{nb_traders}_{kind}_{exponent}'
else:
    filename = f'{nb_traders}_{kind}'

# Directory creation if not present and cancellation of what is inside
dir_path = Path("database\\meta_" + filename)
dir_path.mkdir(parents=True, exist_ok=True)

for item in dir_path.iterdir():
    if item.is_file() or item.is_symlink():
        item.unlink()
    elif item.is_dir():
        shutil.rmtree(item)

dir_path = Path("database\\trades_" + filename)
dir_path.mkdir(parents=True, exist_ok=True)

for item in dir_path.iterdir():
    if item.is_file() or item.is_symlink():
        item.unlink()
    elif item.is_dir():
        shutil.rmtree(item)

l = 0

for path in paths:
    year = path[5:9]
    month = path[10:12]
    day = path[13:15]

    date_string = f"{year}-{month}-{day}"

    print(date_string)

    meta, trades = methods.generate(path, nb_traders, kind, exponent, l)

    l += len(meta)

    meta.to_csv(f'database\\meta_' + filename + '\\meta_' + date_string + '.csv', index=False)
    trades.to_csv(f'database\\trades_' + filename + '\\trades_' + date_string + '.csv', index=False)