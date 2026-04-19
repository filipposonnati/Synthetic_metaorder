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
from numpy.lib.stride_tricks import sliding_window_view

def ar_fit(returns, volumes, p=1000):
    n = len(returns)
    
    # Creates a 2D matrix of shape (n-p, p) instantly
    # We slice [::-1] on the last axis so the lags are ordered [t-1, t-2, ..., t-p]
    X_vol_final = sliding_window_view(volumes[:-1], window_shape=p)[::-1]
    
    # For returns, we need to include V_t as the first column
    y_vol_final = volumes[p:]
    y_ret_final = returns[p:]
    
    # X_ret_final: V_t combined with the lagged matrix
    v_curr = volumes[p:].reshape(-1, 1)
    X_ret_final = np.hstack([v_curr, X_vol_final])

    res_vol = sm.OLS(y_vol_final, X_vol_final).fit()
    res_ret = sm.OLS(y_ret_final, X_ret_final).fit()

    return {
        "p": p,
        "volume_model": {
            "params": res_vol.params, 
            "sigma2": np.var(res_vol.resid)
        },
        "return_model": {
            "params": res_ret.params, 
            "sigma2": np.var(res_ret.resid)
        }
    }

from numba import njit

@njit
def _run_ar_simulation(n_steps, p, sim_v, sim_r, prices, v_params_rev, v_sigma, r_params_0, r_params_lags_rev, r_sigma):
    for t in range(p, n_steps + p):
        # v_lagged is naturally contiguous because it's a forward slice
        v_lagged = sim_v[t-p:t]
        
        # Now this dot product will run at maximum speed
        sim_v[t] = np.dot(v_lagged, v_params_rev) + np.random.normal(0.0, v_sigma)
        
        v_curr = sim_v[t]
        sim_r[t] = (r_params_0 * v_curr) + np.dot(v_lagged, r_params_lags_rev) + np.random.normal(0.0, r_sigma)
        
        prices[t] = prices[t-1] * (1.0 + sim_r[t])
        
    return prices, sim_v, sim_r

# 2. The Python Wrapper
def simulate_ar(results, n_steps, initial_v, initial_r, initial_price=100.0):
    p = results['p']
    sim_v = np.zeros(n_steps + p) 
    sim_r = np.zeros(n_steps + p)
    prices = np.zeros(n_steps + p)
    
    sim_v[:p] = initial_v
    sim_r[:p] = initial_r
    prices[p-1] = initial_price

    # Extract parameters
    v_params = results['volume_model']['params']
    v_sigma = np.sqrt(results['volume_model']['sigma2'])
    r_params = results['return_model']['params']
    r_sigma = np.sqrt(results['return_model']['sigma2'])

    # THE FIX: Create fresh, contiguous arrays in memory for the reversed parameters
    v_params_rev = np.ascontiguousarray(v_params[::-1])
    r_params_0 = r_params[0]
    r_params_lags_rev = np.ascontiguousarray(r_params[1:][::-1])

    # Execute compiled loop
    prices, sim_v, sim_r = _run_ar_simulation(
        n_steps, p, sim_v, sim_r, prices, 
        v_params_rev, v_sigma, r_params_0, r_params_lags_rev, r_sigma
    )

    return prices[p:], sim_v[p:], sim_r[p:]