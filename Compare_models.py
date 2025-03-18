import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load simulated model data
gbm_simulated_prices = pd.read_csv("gbm_simulated_prices.csv", index_col=0, parse_dates=True)
merton_simulated_prices = pd.read_csv("merton_simulated_prices.csv", index_col=0, parse_dates=True)
heston_simulated_prices = pd.read_csv("heston_simulated_prices.csv", index_col=0, parse_dates=True)

# Load NASDAQ-100 real market data
nasdaq_data = pd.read_csv("nasdaq100_data.csv", index_col=0, parse_dates=True)
nasdaq_data["Close"] = pd.to_numeric(nasdaq_data["Close"], errors="coerce")
nasdaq_data.dropna(inplace=True)

# Compute log returns for NASDAQ-100 and each model
nasdaq_returns = np.log(nasdaq_data["Close"] / nasdaq_data["Close"].shift(1)).dropna()
gbm_returns = np.log(gbm_simulated_prices.iloc[-1] / gbm_simulated_prices.iloc[-2]).dropna()
merton_returns = np.log(merton_simulated_prices.iloc[-1] / merton_simulated_prices.iloc[-2]).dropna()
heston_returns = np.log(heston_simulated_prices.iloc[-1] / heston_simulated_prices.iloc[-2]).dropna()

# Compute statistical summaries for return distributions
model_comparison_summary = pd.DataFrame({
    "Model": ["GBM", "Merton", "Heston"],
    "Mean Return": [gbm_returns.mean(), merton_returns.mean(), heston_returns.mean()],
    "Median Return": [gbm_returns.median(), merton_returns.median(), heston_returns.median()],
    "Min Return": [gbm_returns.min(), merton_returns.min(), heston_returns.min()],
    "Max Return": [gbm_returns.max(), merton_returns.max(), heston_returns.max()],
    "Std Dev of Returns": [gbm_returns.std(), merton_returns.std(), heston_returns.std()]
})

# Print the statistical comparison summary
print("\nModel Comparison Summary (Based on Returns):")
print(model_comparison_summary)

# Generate histogram for return distributions
plt.figure(figsize=(12, 6))
plt.hist(nasdaq_returns, bins=50, density=True, alpha=0.8, color="black", label="NASDAQ-100 Real Data")
plt.hist(gbm_returns, bins=50, density=True, alpha=0.5, color="blue", label="GBM")
plt.hist(merton_returns, bins=50, density=True, alpha=0.5, color="red", label="Merton")
plt.hist(heston_returns, bins=50, density=True, alpha=0.5, color="green", label="Heston")
plt.legend()
plt.title("Comparison of Model Simulated Returns vs. NASDAQ-100 Real Returns")
plt.xlabel("Log Returns")
plt.ylabel("Density")
plt.savefig("model_comparison_plot.png", dpi=300)
plt.close()

print("\nModel comparison based on returns completed! Check 'model_comparison_plot.png' for insights.")
