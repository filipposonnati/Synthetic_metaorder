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

from utils import clear_data, open_data, save_simulated_data
from ar import ar_fit, simulate_ar
from var import var_fit, simulate_var
from var_reduced import var_reduced_fit, simulate_var_reduced
from delta import (
    power_transform,
    delta_fit_fixed,
    simulate_delta_fixed,
)
from sign_impact import sign_impact_fit, simulate_sign_impact

paths = np.array(listdir('..\\database\\data'))

p = 1000

name = f'ver_reduced_1000'

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

    results = var_reduced_fit(r, v, p = p)

    prices_sim, volumes_sim, r_sim = simulate_var_reduced(
        results,
        n_steps=len(r),
        initial_v=initial_v,
        initial_r=initial_r,
        initial_price=initial_price
    )

    save_simulated_data(f"..\\{dir}\\" + path, prices_sim, volumes_sim)