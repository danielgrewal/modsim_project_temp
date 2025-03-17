import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load simulated price data from all three models
gbm_simulated_prices = pd.read_csv("gbm_simulated_prices.csv", index_col=0, parse_dates=True)
merton_simulated_prices = pd.read_csv("merton_simulated_prices.csv", index_col=0, parse_dates=True)
heston_simulated_prices = pd.read_csv("heston_simulated_prices.csv", index_col=0, parse_dates=True)

# Extract final simulated prices for each model (last row of each dataset)
final_prices_gbm = gbm_simulated_prices.iloc[-1]
final_prices_merton = merton_simulated_prices.iloc[-1]
final_prices_heston = heston_simulated_prices.iloc[-1]

# Create a summary DataFrame for comparison
comparison_summary = pd.DataFrame({
    "Mean Final Price": [final_prices_gbm.mean(), final_prices_merton.mean(), final_prices_heston.mean()],
    "Median Final Price": [final_prices_gbm.median(), final_prices_merton.median(), final_prices_heston.median()],
    "Min Final Price": [final_prices_gbm.min(), final_prices_merton.min(), final_prices_heston.min()],
    "Max Final Price": [final_prices_gbm.max(), final_prices_merton.max(), final_prices_heston.max()],
    "Std Dev of Final Prices": [final_prices_gbm.std(), final_prices_merton.std(), final_prices_heston.std()],
}, index=["GBM", "Merton", "Heston"])

# Save summary to CSV for reference
comparison_summary.to_csv("model_comparison_summary.csv")

# Display summary stats
print("\nModel Comparison Summary:\n", comparison_summary)

# Plot final price distributions for each model
plt.figure(figsize=(12, 6))
plt.hist(final_prices_gbm, bins=50, alpha=0.5, label="GBM", color="blue", density=True)
plt.hist(final_prices_merton, bins=50, alpha=0.5, label="Merton", color="red", density=True)
plt.hist(final_prices_heston, bins=50, alpha=0.5, label="Heston", color="green", density=True)
plt.xlabel("Final Simulated Price")
plt.ylabel("Density")
plt.title("Comparison of Final Price Distributions (GBM, Merton, Heston)")
plt.legend()
plt.grid()

# Save the plot
plt.savefig("model_comparison_plot.png", dpi=300)
plt.close()

print("\nModel comparison completed!")
print("Results saved as:")
print("  - model_comparison_summary.csv")
print("  - model_comparison_plot.png")
