import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random, datetime
from scipy.stats import powerlaw
import os
from os import listdir
import methods

model = "var_1000"

data_dir = 'database\\data'

if model != "":
    data_dir = data_dir + "_" + model

meta_dir = 'database\\meta'
if model != "":
    meta_dir = meta_dir + "_" + model

if not os.path.exists(meta_dir):
    os.makedirs(meta_dir)

paths = np.array(listdir(data_dir))

configurations = pd.read_csv(f"configurations.csv", header=0)

for index, configuration in configurations.iterrows():
    nb_traders = configuration['nb_traders']
    kind = configuration['kind']
    exponent = configuration['exponent']

    print(nb_traders, kind, exponent)

    if nb_traders == 1:
        filename = f'meta_{nb_traders}.csv'
    elif kind == 'power':
        filename = f'meta_{nb_traders}_{kind}_{exponent}.csv'
    else:
        filename = f'meta_{nb_traders}_{kind}.csv'

    if os.path.exists(f'{meta_dir}\\' + filename):
        os.remove(f'{meta_dir}\\' + filename)

    l = 0
    first = True

    for path in paths:
        print(path)
        meta, _ = methods.generate(path, nb_traders, kind, exponent, l, data_dir)
        l += len(meta)

        meta.to_csv(f'{meta_dir}\\' + filename, mode='a', index=False, header=first)
        first = False