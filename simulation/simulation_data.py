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
#from ar import ar_fit, simulate_ar

def ar_fit(returns, volumes, p=10):
    n = len(returns)
    # Volume: [V_{t-1}, ..., V_{t-p}]
    X_vol_final = np.column_stack([volumes[p-i:n-i] for i in range(1, p+1)])
    
    # Return: [V_t, V_{t-1}, ..., V_{t-p}]
    X_ret_final = np.column_stack([volumes[p:]] + [volumes[p-i:n-i] for i in range(1, p+1)])
    
    y_vol_final = volumes[p:]
    y_ret_final = returns[p:]

    res_vol = sm.OLS(y_vol_final, X_vol_final).fit()
    res_ret = sm.OLS(y_ret_final, X_ret_final).fit()

    return {
        "p": p,
        "volume_model": {
            "params": res_vol.params, # [v_{t-1}, ..., v_{t-p}]
            "sigma2": np.var(res_vol.resid)
        },
        "return_model": {
            "params": res_ret.params, # [v_t, v_{t-1}, ..., v_{t-p}]
            "sigma2": np.var(res_ret.resid)
        }
    }

def power_transform(v, delta):
    """Applica la trasformazione sign(v) * |v|^delta."""
    return np.sign(v) * (np.abs(v) ** delta)

def delta_fit(returns, volumes, p=10):
    n = len(returns)
    y_ret = returns[p:]
    v_curr = volumes[p:]
    v_lags = np.column_stack([volumes[p-i:n-i] for i in range(1, p+1)])

    def loss_function(params):
        # params: [gamma, beta_1...beta_p, delta] (No const)
        gamma = params[0]
        betas = params[1:-1]
        delta = params[-1]
        
        v_curr_transformed = np.sign(v_curr) * (np.abs(v_curr) ** delta)
        v_lags_transformed = np.sign(v_lags) * (np.abs(v_lags) ** delta)
        
        # Modello senza intercetta
        pred = gamma * v_curr_transformed + np.dot(v_lags_transformed, betas)
        return np.sum((y_ret - pred) ** 2)

    # Bounds e Guess iniziale ridotti di 1 unità
    bounds = [(None, None)] * (p + 1) + [(0.001, 0.999)]
    initial_guess = np.zeros(p + 2)
    initial_guess[-1] = 0.5
    
    res = minimize(loss_function, initial_guess, method='L-BFGS-B', bounds=bounds)
    
    delta_est = res.x[-1]
    params_ret = res.x[:-1] # [gamma, beta_1...beta_p]
    
    # Fit volumi lineare senza costante
    X_vol_final = np.column_stack([volumes[p-i:n-i] for i in range(1, p+1)])
    res_vol = sm.OLS(volumes[p:], X_vol_final).fit()

    return {
        "p": p, "delta": delta_est,
        "volume_model": {"params": res_vol.params, "sigma2": np.var(res_vol.resid)},
        "return_model": {"params": params_ret, "sigma2": np.var(y_ret - (params_ret[0] * (np.sign(v_curr)*(np.abs(v_curr)**delta_est)) + np.dot(np.sign(v_lags)*(np.abs(v_lags)**delta_est), params_ret[1:]))), "success": res.success}
    }

def delta_fit_fixed(returns, volumes, delta=0.5, p=10):
    n = len(returns)
    y_ret = returns[p:]
    
    # Trasformazione dei volumi con il delta scelto
    v_curr = volumes[p:]
    v_curr_transf = np.sign(v_curr) * (np.abs(v_curr) ** delta)
    
    v_lags = np.column_stack([volumes[p-i:n-i] for i in range(1, p+1)])
    v_lags_transf = np.sign(v_lags) * (np.abs(v_lags) ** delta)
    
    # Creazione della matrice X: [V_curr^delta, V_{t-1}^delta, ..., V_{t-p}^delta]
    X_ret = np.column_stack([v_curr_transf, v_lags_transf])
    
    # Fit lineare
    res_ret = sm.OLS(y_ret, X_ret).fit()
    
    # Fit volumi (rimane lineare sui volumi originali)
    X_vol_final = np.column_stack([volumes[p-i:n-i] for i in range(1, p+1)])
    res_vol = sm.OLS(volumes[p:], X_vol_final).fit()

    return {
        "p": p, 
        "delta": delta,
        "volume_model": {
            "params": res_vol.params, 
            "sigma2": np.var(res_vol.resid)
        },
        "return_model": {
            "params": res_ret.params, # [gamma, beta_1...beta_p]
            "sigma2": np.var(res_ret.resid)
        }
    }

def var_fit(returns, volumes, p=10):
    n = len(returns)
    v_lags_f = np.column_stack([volumes[p-i:n-i] for i in range(1, p+1)])
    r_lags_f = np.column_stack([returns[p-i:n-i] for i in range(1, p+1)])
    y_vol_f = volumes[p:]; y_ret_f = returns[p:]; v_curr_f = volumes[p:]

    # No add_constant
    X_vol = np.hstack([v_lags_f, r_lags_f])
    res_vol = sm.OLS(y_vol_f, X_vol).fit()

    X_ret = np.column_stack([v_curr_f, v_lags_f, r_lags_f])
    res_ret = sm.OLS(y_ret_f, X_ret).fit()

    return {
        "p": p,
        "volume_model": {
            "params": res_vol.params,
            "sigma2": np.var(res_vol.resid),
            "labels": [f"V_l{i}" for i in range(1, p+1)] + [f"R_l{i}" for i in range(1, p+1)]
        },
        "return_model": {
            "params": res_ret.params,
            "sigma2": np.var(res_ret.resid),
            "labels": ["V_curr"] + [f"V_l{i}" for i in range(1, p+1)] + [f"R_l{i}" for i in range(1, p+1)]
        }
    }

def frac_diff(x, d, threshold=1e-5):
    """Manually applies fractional differencing to a series."""
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold or k > len(x): 
            break
        weights.append(w)
        k += 1
    
    weights = np.array(weights[::-1])
    res = np.convolve(x, weights, mode='valid')
    return res

def arfima_fit(returns, volumes, p=10, d=0.3):
    # 1. Fractionally difference the volumes
    v_diff = frac_diff(volumes, d)
    
    # Adjust returns to match the length of differenced volumes
    # Diffing reduces length by (len(weights) - 1)
    diff_len_loss = len(volumes) - len(v_diff)
    r_adj = returns[diff_len_loss:]
    
    n = len(r_adj)
    
    # 2. Fit AR(p) on the differenced volumes
    v_lags = np.column_stack([v_diff[p-i:n-i] for i in range(1, p+1)])
    y_v = v_diff[p:]
    res_vol = sm.OLS(y_v, v_lags).fit()
    
    # 3. Fit Return Model: R_t depends on V_t, past V, and past R
    # Using original volumes here to maintain the 'impact' relationship
    v_orig_adj = volumes[diff_len_loss+p:]
    v_lags_orig = np.column_stack([volumes[diff_len_loss+p-i:len(volumes)-i] for i in range(1, p+1)])
    r_lags = np.column_stack([r_adj[p-i:n-i] for i in range(1, p+1)])
    y_r = r_adj[p:]
    
    X_ret = np.column_stack([v_orig_adj, v_lags_orig, r_lags])
    res_ret = sm.OLS(y_r, X_ret).fit()
    
    return {
        "p": p, "d": d,
        "volume_model": {"params": res_vol.params, "sigma2": np.var(res_vol.resid)},
        "return_model": {"params": res_ret.params, "sigma2": np.var(res_ret.resid)}
    }

def simulate_ar(results, n_steps, initial_v, initial_r, initial_price=100.0):
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
        v_lagged = sim_v[t-p:t][::-1]
        sim_v[t] = np.dot(v_lagged, v_params) + np.random.normal(0, v_sigma)
        
        v_curr = sim_v[t]
        exog_r = np.concatenate([[v_curr], v_lagged])
        sim_r[t] = np.dot(exog_r, r_params) + np.random.normal(0, r_sigma)
        prices[t] = prices[t-1] * (1 + sim_r[t])

    return prices[p:], sim_v[p:], sim_r[p:]

def simulate_delta(results, n_steps, initial_v, initial_r, initial_price=100.0):
    p = results['p']
    delta = results['delta']
    sim_v = np.zeros(n_steps + p) 
    sim_r = np.zeros(n_steps + p)
    prices = np.zeros(n_steps + p)
    
    sim_v[:p] = initial_v
    sim_r[:p] = initial_r
    prices[p-1] = initial_price

    v_params = results['volume_model']['params']
    v_sigma = np.sqrt(results['volume_model']['sigma2'])
    r_params = results['return_model']['params']
    r_sigma = np.sqrt(results['return_model']['sigma2'])

    for t in range(p, n_steps + p):
        v_lagged = sim_v[t-p:t][::-1]
        sim_v[t] = np.dot(v_lagged, v_params) + np.random.normal(0, v_sigma)

        v_curr = sim_v[t]
        v_lags_transf = np.sign(v_lagged) * (np.abs(v_lagged) ** delta)
        v_curr_transf = np.sign(v_curr) * (np.abs(v_curr) ** delta)
        
        sim_r[t] = r_params[0] * v_curr_transf + np.dot(v_lags_transf, r_params[1:]) + np.random.normal(0, r_sigma)
        prices[t] = prices[t-1] * (1 + sim_r[t])

    return prices[p:], sim_v[p:], sim_r[p:]

def simulate_var(results, n_steps, initial_v, initial_r, initial_price=100.0):
    p = results['p']
    sim_v = np.zeros(n_steps + p) 
    sim_r = np.zeros(n_steps + p)
    prices = np.zeros(n_steps + p)
    
    sim_v[:p] = initial_v
    sim_r[:p] = initial_r
    prices[p-1] = initial_price

    v_params = results['volume_model']['params']
    v_sigma = np.sqrt(results['volume_model']['sigma2'])
    r_params = results['return_model']['params']
    r_sigma = np.sqrt(results['return_model']['sigma2'])

    for t in range(p, n_steps + p):
        v_lagged = sim_v[t-p:t][::-1]
        r_lagged = sim_r[t-p:t][::-1]
        
        exog_v = np.concatenate([v_lagged, r_lagged])
        sim_v[t] = np.dot(exog_v, v_params) + np.random.normal(0, v_sigma)

        v_curr = sim_v[t]
        exog_r = np.concatenate([[v_curr], v_lagged, r_lagged])
        sim_r[t] = np.dot(exog_r, r_params) + np.random.normal(0, r_sigma)
        prices[t] = prices[t-1] * (1 + sim_r[t])

    return prices[p:], sim_v[p:], sim_r[p:]

def simulate_delta_fixed(results, n_steps, initial_v, initial_r, initial_price=100.0):
    p = results['p']
    delta = results['delta']
    
    # Inizializziamo con i dati reali invece di zeri
    sim_v = np.zeros(n_steps + p)
    sim_r = np.zeros(n_steps + p)
    prices = np.zeros(n_steps + p)
    
    # Inseriamo i dati reali nei primi 'p' slot
    sim_v[:p] = initial_v
    sim_r[:p] = initial_r
    prices[p-1] = initial_price

    v_params = results['volume_model']['params']
    v_sigma = np.sqrt(results['volume_model']['sigma2'])
    r_params = results['return_model']['params']
    r_sigma = np.sqrt(results['return_model']['sigma2'])

    for t in range(p, n_steps + p):
        # 1. Simulazione Volume (AR lineare)
        v_lagged = sim_v[t-p:t][::-1]
        sim_v[t] = np.dot(v_lagged, v_params) + np.random.normal(0, v_sigma)

        # 2. Simulazione Return (Power Law Impact)
        v_curr = sim_v[t]
        v_curr_transf = np.sign(v_curr) * (np.abs(v_curr) ** delta)
        v_lags_transf = np.sign(v_lagged) * (np.abs(v_lagged) ** delta)
        
        exog_r = np.concatenate([[v_curr_transf], v_lags_transf])
        sim_r[t] = np.dot(exog_r, r_params) + np.random.normal(0, r_sigma)

        # 3. Aggiornamento Prezzo
        prices[t] = prices[t-1] * (1 + sim_r[t])

    # Restituiamo solo la parte simulata (tagliando i primi p valori reali)
    return prices[p:], sim_v[p:], sim_r[p:]

def simulate_arfima(results, n_steps, initial_v, initial_r, initial_price=100.0):
    p = results['p']
    d = results['d']
    
    # Volume AR parameters
    v_params = results['volume_model']['params']
    v_sigma = np.sqrt(results['volume_model']['sigma2'])
    
    # Return parameters
    r_params = results['return_model']['params']
    r_sigma = np.sqrt(results['return_model']['sigma2'])
    
    # Prepare arrays
    sim_v = np.zeros(n_steps + p)
    sim_r = np.zeros(n_steps + p)
    prices = np.zeros(n_steps + p)
    
    sim_v[:p] = initial_v
    sim_r[:p] = initial_r
    prices[p-1] = initial_price

    # For ARFIMA, we track the 'differenced' state to generate new volume values
    # We'll use a simplified integration approach for the simulation
    for t in range(p, n_steps + p):
        # 1. Generate the 'innovation' for volume via AR(p)
        # In a true ARFIMA, the volume is the integration of this AR process
        v_lagged = sim_v[t-p:t][::-1]
        
        # Approximate the long-memory volume update
        # (This combines the AR part and the fractional memory)
        innovation_v = np.dot(v_lagged, v_params) + np.random.normal(0, v_sigma)
        
        # For simplicity in simulation, we treat the AR process as the driver
        # and the return as the responder
        sim_v[t] = innovation_v 
        
        # 2. Simulate Returns
        v_curr = sim_v[t]
        r_lagged = sim_r[t-p:t][::-1]
        
        exog_r = np.concatenate([[v_curr], v_lagged, r_lagged])
        sim_r[t] = np.dot(exog_r, r_params) + np.random.normal(0, r_sigma)
        
        prices[t] = prices[t-1] * (1 + sim_r[t])
        
    return prices[p:], sim_v[p:], sim_r[p:]

def arfima_power_fit(returns, volumes, p=10, d=0.3):
    # 1. Volume Model: ARFIMA(p, d, 0)
    v_diff = frac_diff(volumes, d)
    diff_len_loss = len(volumes) - len(v_diff)
    
    # Fit AR(p) on differenced volumes
    n_v = len(v_diff)
    v_lags_diff = np.column_stack([v_diff[p-i:n_v-i] for i in range(1, p+1)])
    y_v = v_diff[p:]
    res_vol = sm.OLS(y_v, v_lags_diff).fit()

    # 2. Return Model: R_t = gamma * sign(V_t)|V_t|^alpha + ...
    r_adj = returns[diff_len_loss:]
    y_ret = r_adj[p:]
    v_curr = volumes[diff_len_loss+p:]
    v_lags = np.column_stack([volumes[diff_len_loss+p-i:len(volumes)-i] for i in range(1, p+1)])
    r_lags = np.column_stack([r_adj[p-i:len(r_adj)-i] for i in range(1, p+1)])

    def loss_function(params):
        alpha = params[0]
        gamma = params[1]
        betas = params[2:2+p]  # Coefficients for past V^alpha
        phi_r = params[2+p:]   # Coefficients for past R
        
        v_curr_pow = np.sign(v_curr) * (np.abs(v_curr) ** alpha)
        v_lags_pow = np.sign(v_lags) * (np.abs(v_lags) ** alpha)
        
        pred = (gamma * v_curr_pow + 
                np.dot(v_lags_pow, betas) + 
                np.dot(r_lags, phi_r))
        return np.sum((y_ret - pred) ** 2)

    # Optimization bounds: alpha in (0, 1)
    initial_guess = [0.5, 0.0] + [0.0]*p + [0.0]*p
    bounds = [(0.001, 0.999), (None, None)] + [(None, None)]*(2*p)
    
    res = minimize(loss_function, initial_guess, method='L-BFGS-B', bounds=bounds)
    
    return {
        "p": p, "d": d, "alpha": res.x[0],
        "volume_model": {"params": res_vol.params, "sigma2": np.var(res_vol.resid)},
        "return_model": {"params": res.x[1:], "sigma2": res.fun / len(y_ret)}
    }

def simulate_arfima_power(results, n_steps, initial_v, initial_r, initial_price=100.0):
    p, d, alpha = results['p'], results['d'], results['alpha']
    v_params = results['volume_model']['params']
    v_sigma = np.sqrt(results['volume_model']['sigma2'])
    r_params = results['return_model']['params'] # [gamma, betas..., phi_r...]
    r_sigma = np.sqrt(results['return_model']['sigma2'])

    sim_v = np.zeros(n_steps + p)
    sim_r = np.zeros(n_steps + p)
    prices = np.zeros(n_steps + p)
    
    sim_v[:p], sim_r[:p], prices[p-1] = initial_v, initial_r, initial_price

    gamma = r_params[0]
    betas = r_params[1:1+p]
    phi_r = r_params[1+p:]

    for t in range(p, n_steps + p):
        # 1. Volume (ARFIMA approximation)
        v_lagged = sim_v[t-p:t][::-1]
        sim_v[t] = np.dot(v_lagged, v_params) + np.random.normal(0, v_sigma)

        # 2. Returns with Power Law Impact: sign(V) * |V|^alpha
        v_curr = sim_v[t]
        v_curr_pow = np.sign(v_curr) * (np.abs(v_curr) ** alpha)
        v_lags_pow = np.sign(v_lagged) * (np.abs(v_lagged) ** alpha)
        r_lagged = sim_r[t-p:t][::-1]

        sim_r[t] = (gamma * v_curr_pow + 
                    np.dot(v_lags_pow, betas) + 
                    np.dot(r_lagged, phi_r) + 
                    np.random.normal(0, r_sigma))
        
        prices[t] = prices[t-1] * (1 + sim_r[t])

    return prices[p:], sim_v[p:], sim_r[p:]

paths = np.array(listdir('..\\database\\data'))

p = 1000

name = f'ar_{p}'

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
    initial_price = prices[p] # Prezzo reale al punto p

    #results = arfima_power_fit(r, v, p=p, d=0.3) # Example d value
    results = ar_fit(r, v, p)

    """
    prices_sim, volumes_sim, r_sim = simulate_arfima_power(
        results,
        n_steps=len(r),
        initial_v=initial_v,
        initial_r=initial_r,
        initial_price=initial_price
    )
    """
    prices_sim, volumes_sim, r_sim = simulate_ar(
        results,
        n_steps=len(r),
        initial_v=initial_v,
        initial_r=initial_r,
        initial_price=initial_price
    )

    save_simulated_data(f"..\\{dir}\\" + path, prices_sim, volumes_sim)