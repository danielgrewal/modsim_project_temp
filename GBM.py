import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load NASDAQ-100 data
csv_filename = "nasdaq100_data.csv"  # Ensure this file exists in your directory
nasdaq_data = pd.read_csv(csv_filename, index_col=0, parse_dates=True)

# Convert "Close" column to numeric values (force errors to NaN, then drop any bad rows)
nasdaq_data["Close"] = pd.to_numeric(nasdaq_data["Close"], errors="coerce")
nasdaq_data = nasdaq_data.dropna()

# Extract required values for GBM model
S0 = nasdaq_data["Close"].iloc[-1]  # Latest closing price
returns = np.log(nasdaq_data["Close"] / nasdaq_data["Close"].shift(1)).dropna()
mu = returns.mean() * 252  # Annualized drift (mean return)
sigma = 0.28  # Set volatility explicitly to 28% (better for NASDAQ-100): NASDAQ-100 has higher historical volatility (~25-30%), so 28% is more accurate

# Simulation parameters
T = 1  # Time horizon (in years)
dt = 1/252  # Time step (daily data)
N = int(T/dt)  # Number of time steps
num_simulations = 10000  # Number of Monte Carlo simulations

# Generate GBM paths
np.random.seed(42)
time = np.linspace(0, T, N)
W = np.random.standard_normal(size=(num_simulations, N))  # Brownian motion
W = np.cumsum(np.sqrt(dt) * W, axis=1)  # Cumulative sum for Wiener process
S_paths = S0 * np.exp((mu - 0.5 * sigma**2) * time + sigma * W)

# Convert to DataFrame
gbm_simulated_prices = pd.DataFrame(S_paths.T, index=pd.date_range(start=nasdaq_data.index[-1], periods=N, freq='D'))

# Save simulated data
gbm_simulated_csv = "gbm_simulated_prices.csv"
gbm_simulated_prices.to_csv(gbm_simulated_csv)

# Plot simulation results
plt.figure(figsize=(12, 6))
plt.plot(gbm_simulated_prices, alpha=0.1, color="blue")
plt.xlabel("Date")
plt.ylabel("Simulated Price")
plt.title("GBM Simulated Stock Price Paths for NASDAQ-100")
plt.grid()

# Save plot
gbm_plot_filename = "gbm_simulation_plot.png"
plt.savefig(gbm_plot_filename, dpi=300)
plt.close()

print(f"GBM Simulation completed! Data saved to: {gbm_simulated_csv}")
print(f"Plot saved to: {gbm_plot_filename}")
