import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load NASDAQ-100 data
csv_filename = "nasdaq100_data.csv"  # Ensure this file exists in your directory
nasdaq_data = pd.read_csv(csv_filename, index_col=0, parse_dates=True)

# Convert "Close" column to numeric values
nasdaq_data["Close"] = pd.to_numeric(nasdaq_data["Close"], errors="coerce")
nasdaq_data = nasdaq_data.dropna()

# Extract required values
S0 = nasdaq_data["Close"].iloc[-1]  # Latest closing price
returns = np.log(nasdaq_data["Close"] / nasdaq_data["Close"].shift(1)).dropna()
mu = returns.mean() * 252  # Annualized drift
sigma = 0.28  # Set NASDAQ-100 volatility explicitly to 28% based on historical data and to be consistent across the models

# Simulation parameters
T = 1  # Time horizon (1 year)
dt = 1/252  # Time step (daily data)
N = int(T/dt)  # Number of time steps
num_simulations = 10000  # Number of Monte Carlo simulations

# Merton Jump Diffusion parameters
lambda_jump = 0.3  # 0.3 jumps per day (~75 jumps per year), More reasonable for high-volatility indices like NASDAQ-100
mu_jump = 0.04  # Jump magnitude, matches historical NASDAQ-100 jump sizes
sigma_jump = 0.2  # Jump volatility (impact), balances extreme jumps without over-exaggeration

# Generate Jump Diffusion paths
np.random.seed(42)
W_merton = np.random.standard_normal(size=(num_simulations, N))  # Brownian motion
W_merton = np.cumsum(np.sqrt(dt) * W_merton, axis=1)

# Generate Poisson-distributed jump occurrences
Jumps = np.random.poisson(lambda_jump * dt, size=(num_simulations, N))
Jump_Sizes = np.random.normal(mu_jump, sigma_jump, size=(num_simulations, N)) * Jumps

# Compute Merton model price paths
S_merton_paths = S0 * np.exp((mu - 0.5 * sigma**2) * np.linspace(0, T, N) + sigma * W_merton + Jump_Sizes.cumsum(axis=1))

# Convert to DataFrame
merton_simulated_prices = pd.DataFrame(S_merton_paths.T, index=pd.date_range(start=nasdaq_data.index[-1], periods=N, freq='D'))

# Save simulated data
merton_simulated_csv = "merton_simulated_prices.csv"
merton_simulated_prices.to_csv(merton_simulated_csv)

# Plot simulation results
plt.figure(figsize=(12, 6))
plt.plot(merton_simulated_prices, alpha=0.1, color="red")
plt.xlabel("Date")
plt.ylabel("Simulated Price")
plt.title("Merton Jump Diffusion Model Simulated Stock Price Paths for NASDAQ-100")
plt.grid()

# Save plot instead of displaying it
merton_plot_filename = "merton_simulation_plot.png"
plt.savefig(merton_plot_filename, dpi=300)
plt.close()

print(f"Merton Simulation completed! Data saved to: {merton_simulated_csv}")
print(f"Plot saved to: {merton_plot_filename}")
