import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Load NASDAQ-100 real data
nasdaq_data = pd.read_csv("nasdaq100_data.csv", index_col=0, parse_dates=True, dayfirst=True)
nasdaq_data["Close"] = pd.to_numeric(nasdaq_data["Close"], errors="coerce")
nasdaq_data.dropna(inplace=True)

# Compute real returns (log returns for better distribution comparison)
nasdaq_returns = np.log(nasdaq_data["Close"] / nasdaq_data["Close"].shift(1)).dropna()

# Load simulated data
gbm_simulated = pd.read_csv("gbm_simulated_prices.csv", index_col=0, parse_dates=True)
merton_simulated = pd.read_csv("merton_simulated_prices.csv", index_col=0, parse_dates=True)
heston_simulated = pd.read_csv("heston_simulated_prices.csv", index_col=0, parse_dates=True)

# Compute returns for simulated models
gbm_returns = np.log(gbm_simulated.iloc[-1] / gbm_simulated.iloc[-2]).dropna()
merton_returns = np.log(merton_simulated.iloc[-1] / merton_simulated.iloc[-2]).dropna()
heston_returns = np.log(heston_simulated.iloc[-1] / heston_simulated.iloc[-2]).dropna()

# Kolmogorov-Smirnov Test (KS Test) - Compare distributions of returns
ks_gbm = ks_2samp(gbm_returns, nasdaq_returns)
ks_merton = ks_2samp(merton_returns, nasdaq_returns)
ks_heston = ks_2samp(heston_returns, nasdaq_returns)

# Print KS Test Results
print("Kolmogorov-Smirnov Test Results:")
print(f"GBM vs NASDAQ-100: D-stat = {ks_gbm.statistic:.4f}, p-value = {ks_gbm.pvalue:.4f}")
print(f"Merton vs NASDAQ-100: D-stat = {ks_merton.statistic:.4f}, p-value = {ks_merton.pvalue:.4f}")
print(f"Heston vs NASDAQ-100: D-stat = {ks_heston.statistic:.4f}, p-value = {ks_heston.pvalue:.4f}")

# Plot distributions of returns for visual validation
plt.figure(figsize=(12, 6))
plt.hist(gbm_returns, bins=50, alpha=0.5, label="GBM", density=True, color='blue')
plt.hist(merton_returns, bins=50, alpha=0.5, label="Merton", density=True, color='red')
plt.hist(heston_returns, bins=50, alpha=0.5, label="Heston", density=True, color='green')
plt.hist(nasdaq_returns, bins=50, alpha=0.5, label="NASDAQ-100 Real Data", density=True, color='black')
plt.legend()
plt.xlabel("Log Returns")
plt.ylabel("Density")
plt.title("Comparison of Model Simulated Returns vs. NASDAQ-100 Real Returns")
plt.grid()

# Save plot instead of showing it
plt.savefig("validation_comparison_plot.png", dpi=300)
plt.close()

print("Validation completed! Check validation_comparison_plot.png for visual analysis.")
