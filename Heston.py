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

# Simulation parameters
T = 1  # Time horizon (1 year)
dt = 1/252  # Time step (daily data)
N = int(T/dt)  # Number of time steps
num_simulations = 10000  # Number of Monte Carlo simulations

# Heston model parameters for NASDAQ-100
v0 = 0.09  # Start at 30% volatility (0.30^2)
theta = 0.09  # Long-run variance (30% volatility)
kappa = 2.0  # Slower mean reversion, NASDAQ-100 volatility is more persistent
xi = 0.2  # Increased volatility of volatility for NASDAQ-100
rho = -0.5  # Correlation between asset price and volatility

# Generate correlated Brownian motions
np.random.seed(42)
W_asset = np.random.standard_normal(size=(num_simulations, N))  # Brownian motion for asset price
W_vol = np.random.standard_normal(size=(num_simulations, N))  # Brownian motion for volatility

# Apply correlation to the Brownian motions
W_vol = rho * W_asset + np.sqrt(1 - rho**2) * W_vol

# Initialize volatility and price arrays
v_t = np.full((num_simulations, N), v0)
S_heston_paths = np.full((num_simulations, N), S0)

# Simulate paths using Euler-Maruyama method
for t in range(1, N):
    v_t[:, t] = np.abs(v_t[:, t - 1] + kappa * (theta - v_t[:, t - 1]) * dt + xi * np.sqrt(v_t[:, t - 1] * dt) * W_vol[:, t])
    S_heston_paths[:, t] = S_heston_paths[:, t - 1] * np.exp((mu - 0.5 * v_t[:, t]) * dt + np.sqrt(v_t[:, t] * dt) * W_asset[:, t])

# Convert to DataFrame
heston_simulated_prices = pd.DataFrame(S_heston_paths.T, index=pd.date_range(start=nasdaq_data.index[-1], periods=N, freq='D'))

# Save simulated data
heston_simulated_csv = "heston_simulated_prices.csv"
heston_simulated_prices.to_csv(heston_simulated_csv)

# Plot simulation results
plt.figure(figsize=(12, 6))
plt.plot(heston_simulated_prices, alpha=0.1, color="green")
plt.xlabel("Date")
plt.ylabel("Simulated Price")
plt.title("Heston Stochastic Volatility Model Simulated Stock Price Paths for NASDAQ-100")
plt.grid()

# Save plot instead of displaying it
heston_plot_filename = "heston_simulation_plot.png"
plt.savefig(heston_plot_filename, dpi=300)
plt.close()

print(f"Heston Simulation completed! Data saved to: {heston_simulated_csv}")
print(f"Plot saved to: {heston_plot_filename}")
