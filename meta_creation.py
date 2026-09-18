import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random, datetime
from scipy.stats import powerlaw
import os
from os import listdir
import methods

model = "lmf_1.8_0.3_log_ar_tim_sqrt"
#model = ''
iterations = 50  # Added iterations variable with default value of 10
n_filter = 10

data_dir = 'database\\data'

if model != "":
    data_dir = data_dir + "_" + model

meta_dir = 'database\\meta'
if model != "":
    meta_dir = meta_dir + "_" + model

if not os.path.exists(meta_dir):
    os.makedirs(meta_dir)

paths = np.array(listdir(data_dir))

#configurations = pd.read_csv(f"configurations.csv", header=0)

#for index, configuration in configurations.iterrows():
#nb_traders = configuration['nb_traders']
#kind = configuration['kind']
#exponent = configuration['exponent']

nb_traders = 20
kind = 'power'
exponent = 2.0

if nb_traders == 1:
    filename = f'meta_{nb_traders}'
elif kind == 'uniform':
    filename = f'meta_{nb_traders}_{kind}'
else:
    filename = f'meta_{nb_traders}_{kind}_{exponent}'

if n_filter > 2:
    filename = filename + f'_{n_filter}'

filename = filename + '.csv'

file_path = os.path.join(meta_dir, filename)

# Skip configuration if the target file already exists
if os.path.exists(file_path):
    print(f"Skipping configuration ({nb_traders}, {kind}, {exponent}) — {filename} already exists.")
    exit()

print(nb_traders, kind, exponent)

l = 0
first = True

for path in paths:
    print(path)
    # Loop for the specified number of iterations per path/day
    for it in range(iterations):
        meta, _ = methods.generate(path, nb_traders, kind, exponent, l, data_dir)
        meta = meta[meta['NbChild'] >= n_filter]
        l += len(meta)

        meta.to_csv(file_path, mode='a', index=False, header=first)
        first = False