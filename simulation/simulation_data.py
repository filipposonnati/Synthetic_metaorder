import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import random, datetime
from scipy.stats import powerlaw
from os import listdir
from scipy.optimize import minimize
from pathlib import Path
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import clear_data, open_data, save_simulated_data
from ar import ar_fit, simulate_ar
from var import var_fit, simulate_var
from delta import (
    power_transform,
    delta_fit,
    delta_fit_fixed,
    frac_diff,
    arfima_fit,
    simulate_delta,
    simulate_delta_fixed,
)

paths = np.array(listdir('..\\database\\data'))

p = 1000

name = f'ar_1000'

dir = 'database\\data_' + name

if not os.path.exists('..\\' + dir):
    os.makedirs('..\\' + dir)

clear_data(name)

for path in paths:
    print(path)

    prices, volumes, signs = open_data(path)

    r = (prices[1:] - prices[:-1]) / prices[:-1]
    v = volumes[:-1] * signs[:-1]

    # Prendi i primi 'p' valori come seme per la simulazione
    initial_v = v[:p]
    initial_r = r[:p]
    initial_price = prices[p]  # Prezzo reale al punto p

    results = ar_fit(r, v, p)

    prices_sim, volumes_sim, r_sim = simulate_ar(
        results,
        n_steps=len(r),
        initial_v=initial_v,
        initial_r=initial_r,
        initial_price=initial_price
    )

    save_simulated_data(f"..\\{dir}\\" + path, prices_sim, volumes_sim)