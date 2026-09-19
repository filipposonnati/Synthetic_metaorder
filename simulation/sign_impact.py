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

def sign_impact_fit(returns, volumes, p=10):
    n = len(returns)
    # Calcolo del segno dei volumi (1 per >0, -1 per <0, 0 per 0)
    sign_volumes = np.sign(volumes)
    
    # Volume: [sign(V_{t-1}), ..., sign(V_{t-p})]
    X_vol_final = np.column_stack([sign_volumes[p-i:n-i] for i in range(1, p+1)])
    
    # Return: [sign(V_t), sign(V_{t-1}), ..., sign(V_{t-p})]
    X_ret_final = np.column_stack([sign_volumes[p:]] + [sign_volumes[p-i:n-i] for i in range(1, p+1)])
    
    y_vol_final = volumes[p:]
    y_ret_final = returns[p:]

    res_vol = sm.OLS(y_vol_final, X_vol_final).fit()
    res_ret = sm.OLS(y_ret_final, X_ret_final).fit()

    return {
        "p": p,
        "volume_model": {
            "params": res_vol.params, # [sign(v_{t-1}), ..., sign(v_{t-p})]
            "sigma2": np.var(res_vol.resid)
        },
        "return_model": {
            "params": res_ret.params, # [sign(v_t), sign(v_{t-1}), ..., sign(v_{t-p})]
            "sigma2": np.var(res_ret.resid)
        }
    }

def simulate_sign_impact(results, n_steps, initial_v, initial_r, initial_price=100.0):
    p = results['p']
    sim_v = np.zeros(n_steps + p) 
    sim_r = np.zeros(n_steps + p)
    prices = np.zeros(n_steps + p)
    
    # Inizializzazione con dati reali
    sim_v[:p] = initial_v
    sim_r[:p] = initial_r
    prices[p-1] = initial_price

    v_params = results['volume_model']['params']
    v_sigma = np.sqrt(results['volume_model']['sigma2'])
    r_params = results['return_model']['params']
    r_sigma = np.sqrt(results['return_model']['sigma2'])

    for t in range(p, n_steps + p):
        # Calcolo dei segni dei volumi passati
        sign_v_lagged = np.sign(sim_v[t-p:t][::-1])
        
        sim_v[t] = np.dot(sign_v_lagged, v_params) + np.random.normal(0, v_sigma)
        
        # Calcolo del segno del volume corrente e combinazione con i ritardi
        sign_v_curr = np.sign(sim_v[t])
        exog_sign_r = np.concatenate([[sign_v_curr], sign_v_lagged])
        
        sim_r[t] = np.dot(exog_sign_r, r_params) + np.random.normal(0, r_sigma)
        prices[t] = prices[t-1] * (1 + sim_r[t])

    return prices[p:], sim_v[p:], sim_r[p:]