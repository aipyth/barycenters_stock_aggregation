import numpy as np
import matplotlib.pyplot as plt
from ot import barycenter
from ot.utils import sample_histogram

# Number of assets
num_assets = 3

# Simulate returns for each asset as different Gaussian distributions
asset_returns = []
for i in range(num_assets):
    mean = np.random.uniform(-0.05, 0.05)  # Mean return between -5% and 5%
    std = np.random.uniform(0.1, 0.3)      # Standard deviation between 10% and 30%
    returns = np.random.normal(loc=mean, scale=std, size=1000)
    hist, bins = np.histogram(returns, bins=100, density=True)
    asset_returns.append(hist)

asset_returns = np.array(asset_returns)

# Define equal weights for simplicity
asset_weights = np.ones(num_assets) / num_assets

# Compute the barycenter
bc_fin = barycenter(asset_returns, asset_weights, numItermax=1000, verbose=True)

# Define bin centers
bin_centers_fin = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(10, 6))

# Plot asset return distributions
for i in range(num_assets):
    plt.plot(bin_centers_fin, asset_returns[i], label=f'Asset {i+1}')

# Plot barycenter
plt.plot(bin_centers_fin, bc_fin, label='Barycenter', linewidth=3, color='red')

plt.title('Barycenter of Asset Return Distributions')
plt.xlabel('Return')
plt.ylabel('Density')
plt.legend()
plt.show()
