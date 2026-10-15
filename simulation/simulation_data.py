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
from var import var_fit, simulate_var
from var_reduced import var_reduced_fit, simulate_var_reduced
from simulation.tim import impact_simulate, impact_fit, save_impact_model, load_impact_model, plot_impact_fit

paths = np.array(listdir('..\\database\\data'))

p = 100
delta = 0.0
bootstrap_residuals = True

name = f'reg_delta_{delta}_{p}'
#name = f'var_reduced_{p}'

dir = 'database\\data_' + name

if not os.path.exists('..\\' + dir):
    os.makedirs('..\\' + dir)

clear_data(name)

model_dir = Path('..') / 'models' / name

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

for path in paths:
    print(path)
    model_path = model_dir / (Path(path).stem + '.npz')

    prices, volumes, signs = open_data(path)

    r = (prices[1:] - prices[:-1]) / prices[:-1]
    v = volumes[:-1] * signs[:-1]

    # Prendi i primi 'p' valori come seme per la simulazione
    initial_v = v[:p]
    initial_r = r[:p]
    initial_price = prices[p]  # Prezzo reale al punto p

    if model_path.exists():
        print('Load Model')
        results = load_impact_model(model_path)       # salta il fit
    else:
        print('Compute Model')
        results = impact_fit(r, v, delta=delta, p=p)
        save_impact_model(results, model_path)

    save_impact_model(results, model_path)

    plot_impact_fit(results)

    exit()

    prices_sim, volumes_sim, r_sim = impact_simulate(
        results,
        n_steps=len(r),
        initial_v=initial_v,
        initial_price=initial_price,
        bootstrap=True
    )

    """
    results = var_reduced_fit(r, v, p = p, plot=False)

    prices_sim, volumes_sim, r_sim = simulate_var_reduced(
        results,
        n_steps=len(r),
        initial_v=initial_v,
        initial_r=initial_r,
        initial_price=initial_price,
        bootstrap=True
    )
    """

    #save_simulated_data(f"..\\{dir}\\" + path, prices_sim, volumes_sim)