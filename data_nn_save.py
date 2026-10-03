import numpy as np
data = np.load('database\\NN_simulation.npz')

returns = data['returns']
volumes = np.abs(data['signed_volume'])
signs = np.sign(data['signed_volume']).astype(int)

initial_price = 1.0

cumulative_returns = np.cumsum(returns)

prices = initial_price * np.exp(cumulative_returns)

time = np.linspace(1, len(returns), len(returns)).astype(int)

"""
limit = 1_000_000
time = time[:limit]
prices = prices[:limit]
volumes = volumes[:limit]
signs = signs[:limit]
"""

a = np.asarray([time, prices, volumes, signs]).T
np.savetxt("database\\data_nn\\data_2023-01-03_type4_aggregato_auction.csv", a, delimiter=",")
