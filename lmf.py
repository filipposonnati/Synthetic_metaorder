import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit

def power_law(x, constant, alpha):
    return constant * x**alpha

def simulate_lillo_farmer_multi_trader(alpha, n_traders, total_steps):
    """
    Advanced simulation: pool of traders with overlapping metaorders.
    """
    # Initialization: each trader has a metaorder (length and sign)
    # trader_state: [sign (+1/-1), remaining_pieces]
    trader_state = np.zeros((n_traders, 2), dtype=int)
    
    def get_new_metaorder():
        length = int(np.random.pareto(alpha) + 1)
        side = np.random.choice([1, -1])
        return side, length

    # Populate the initial pool
    for i in range(n_traders):
        trader_state[i] = get_new_metaorder()

    order_flow = np.zeros(total_steps)

    for t in range(total_steps):
        # 1. Select a random trader (scheduling mechanism)
        idx = np.random.randint(0, n_traders)
        
        # 2. Execute one unit of their metaorder
        side, remaining = trader_state[idx]
        order_flow[t] = side
        
        # 3. Update the trader's state
        remaining -= 1
        if remaining <= 0:
            # If the metaorder is finished, the trader receives a new one
            trader_state[idx] = get_new_metaorder()
        else:
            trader_state[idx, 1] = remaining
            
    return order_flow

# --- Parameters ---
alpha = 1.5         # Pareto exponent
n_traders = 20       # Number of traders operating in parallel
total_steps = 5_000_000
max_lag = 1000

# Simulation
flow = simulate_lillo_farmer_multi_trader(alpha, n_traders, total_steps)

# ACF Calculation
auto_corr = acf(flow, nlags=max_lag, fft=True)

# Theory: C(tau) ~ tau^-(gamma) where gamma = alpha - 1
# Note: The exact relationship depends on the sampling type, 
# but the log-log slope is dictated by alpha.
theoretical_gamma = alpha - 1
lags = np.arange(1, max_lag + 1)
theoretical_decay = (lags**(-theoretical_gamma)) * auto_corr[1]

popt, pcov = curve_fit(power_law, lags[25 - 1:], auto_corr[25:])
print(f"Fit parameters: {popt}, Errors: {np.sqrt(np.diag(pcov))}")

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.loglog(lags, auto_corr[1:], label='Simulated ACF (Multi-trader)')
plt.loglog(lags, theoretical_decay, 'r--', label=r'Theory: $tau^{-( \alpha - 1)}$')
plt.loglog(lags, lags**(popt[1]) * popt[0], 'b--', label='Power Law Fit')

plt.xlabel("Lag $\\tau$")
plt.ylabel("ACF $C(\\tau)$")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig('images\\lmf_acf.png')
plt.show()