import numpy as np
import pandas as pd
import os
from os import listdir
import methods

model = ""
N_REALIZATIONS = 25
MIN_CHILD = 2
MAX_PER_CHILD = 1_000_000  # cap per NbChild value; None = no cap

data_dir = 'database\\data'
if model != "":
    data_dir = data_dir + "_" + model

meta_dir = 'database\\meta_slim'
if model != "":
    meta_dir = meta_dir + "_" + model

if not os.path.exists(meta_dir):
    os.makedirs(meta_dir)

paths = np.array(listdir(data_dir))

configurations = pd.read_csv("configurations.csv", header=0)

for index, configuration in configurations.iterrows():
    nb_traders = configuration['nb_traders']
    kind       = configuration['kind']
    exponent   = configuration['exponent']

    print(f"\n=== nb_traders={nb_traders}, kind={kind}, exponent={exponent} ===")

    if nb_traders == 1:
        filename = f'meta_{nb_traders}.csv'
    elif kind == 'power':
        filename = f'meta_{nb_traders}_{kind}_{exponent}.csv'
    else:
        filename = f'meta_{nb_traders}_{kind}.csv'

    full_path = f'{meta_dir}\\{filename}'

    # If file exists, resume from current counts; otherwise start fresh
    if os.path.exists(full_path):
        existing = pd.read_csv(full_path, usecols=['NbChild'])
        counts = existing['NbChild'].value_counts().to_dict()
        first = False
        print(f"  Resuming — existing counts: { {k: v for k, v in sorted(counts.items())} }")
    else:
        counts = {}
        first = True

    for r in range(N_REALIZATIONS):
        print(f"  Realization {r + 1}/{N_REALIZATIONS}")
        for path in paths:
            slim = methods.generate_slim(path, nb_traders, kind, exponent, data_dir)

            if slim.empty:
                continue

            # Filter 1: drop metaorders below minimum length
            slim = slim[slim['NbChild'] >= MIN_CHILD]

            # Filter 2: for each row, save it only if its bucket is not full
            if MAX_PER_CHILD is not None:
                mask = slim['NbChild'].map(lambda nb: counts.get(nb, 0) < MAX_PER_CHILD)
                slim = slim[mask]
                for nb, n in slim['NbChild'].value_counts().items():
                    counts[nb] = counts.get(nb, 0) + n

            if slim.empty:
                continue

            slim.to_csv(full_path, mode='a', index=False, header=first)
            first = False

    print(f"  Final counts: { {k: v for k, v in sorted(counts.items())} }")