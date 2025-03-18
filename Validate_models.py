import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import ks_2samp, anderson_ksamp, jarque_bera, cramervonmises_2samp

# 1. Load NASDAQ-100 real-world data (Close Prices)
nasdaq_data = pd.read_csv("nasdaq100_data.csv", index_col=0, parse_dates=True)
nasdaq_data["Close"] = pd.to_numeric(nasdaq_data["Close"], errors="coerce")
nasdaq_data.dropna(inplace=True)

# 2. Compute NASDAQ-100 log returns for better distribution comparison
nasdaq_returns = np.log(nasdaq_data["Close"] / nasdaq_data["Close"].shift(1)).dropna()

# 3. Load simulated stock price paths from models
gbm_simulated = pd.read_csv("gbm_simulated_prices.csv", index_col=0, parse_dates=True)
merton_simulated = pd.read_csv("merton_simulated_prices.csv", index_col=0, parse_dates=True)
heston_simulated = pd.read_csv("heston_simulated_prices.csv", index_col=0, parse_dates=True)

# 4. Compute simulated log returns from the last two time steps of each model
gbm_returns = np.log(gbm_simulated.iloc[-1] / gbm_simulated.iloc[-2]).dropna()
merton_returns = np.log(merton_simulated.iloc[-1] / merton_simulated.iloc[-2]).dropna()
heston_returns = np.log(heston_simulated.iloc[-1] / heston_simulated.iloc[-2]).dropna()

# ---- Statistical Tests ----

# 5. Kolmogorov-Smirnov (KS) Test
# This test checks if two distributions come from the same population.
# A high D-statistic and low p-value indicate a significant difference between distributions.
ks_gbm = ks_2samp(gbm_returns, nasdaq_returns)
ks_merton = ks_2samp(merton_returns, nasdaq_returns)
ks_heston = ks_2samp(heston_returns, nasdaq_returns)

# 6. Anderson-Darling (AD) Test
# This test gives more weight to the tails of the distribution.
# A high AD-statistic and low p-value indicate stronger differences, especially in extreme values.
ad_gbm = anderson_ksamp([gbm_returns, nasdaq_returns])
ad_merton = anderson_ksamp([merton_returns, nasdaq_returns])
ad_heston = anderson_ksamp([heston_returns, nasdaq_returns])

# 7. Jarque-Bera (JB) Test
# This test checks if the data is normally distributed by measuring skewness & kurtosis.
# A low p-value means the distribution deviates significantly from normality.
jb_gbm = jarque_bera(gbm_returns)
jb_merton = jarque_bera(merton_returns)
jb_heston = jarque_bera(heston_returns)

# 8. Cramér–von Mises (CVM) Test
# This test is similar to KS but considers differences across the entire distribution.
# A high W-statistic and low p-value indicate a poor match between the two distributions.
cvm_gbm = cramervonmises_2samp(gbm_returns, nasdaq_returns)
cvm_merton = cramervonmises_2samp(merton_returns, nasdaq_returns)
cvm_heston = cramervonmises_2samp(heston_returns, nasdaq_returns)

# ---- Visualization ----

# 9. Validation Histogram Plot (Re-added)
# This plot compares the final distribution of returns from all models with real NASDAQ-100 returns.
plt.figure(figsize=(12, 6))
plt.hist(nasdaq_returns, bins=50, density=True, alpha=0.8, color="black", label="NASDAQ-100 Real Data")
plt.hist(gbm_returns, bins=50, density=True, alpha=0.5, color="blue", label="GBM")
plt.hist(merton_returns, bins=50, density=True, alpha=0.5, color="red", label="Merton")
plt.hist(heston_returns, bins=50, density=True, alpha=0.5, color="green", label="Heston")
plt.legend()
plt.title("Comparison of Simulated Model Returns vs. NASDAQ-100 Real Returns")
plt.xlabel("Log Returns")
plt.ylabel("Density")
plt.savefig("validation_comparison_plot.png", dpi=300)
plt.close()

# 10. QQ Plots
# Quantile-Quantile (QQ) plots show how well each model matches the real-world NASDAQ-100 returns.
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
stats.probplot(nasdaq_returns, dist="norm", plot=ax[0])
ax[0].set_title("QQ Plot - NASDAQ-100")
stats.probplot(gbm_returns, dist="norm", plot=ax[1])
ax[1].set_title("QQ Plot - GBM")
stats.probplot(merton_returns, dist="norm", plot=ax[2])
ax[2].set_title("QQ Plot - Merton")
plt.tight_layout()
plt.savefig("qq_plots.png", dpi=300)
plt.close()

# 11. Skewness & Kurtosis Bar Chart
# This shows whether models capture fat tails and asymmetry correctly.
metrics = {
    "NASDAQ-100": [nasdaq_returns.skew(), nasdaq_returns.kurtosis()],
    "GBM": [gbm_returns.skew(), gbm_returns.kurtosis()],
    "Merton": [merton_returns.skew(), merton_returns.kurtosis()],
    "Heston": [heston_returns.skew(), heston_returns.kurtosis()]
}
df_metrics = pd.DataFrame(metrics, index=["Skewness", "Kurtosis"])
df_metrics.T.plot(kind="bar", figsize=(10, 6), title="Skewness & Kurtosis Comparison")
plt.savefig("skewness_kurtosis.png", dpi=300)
plt.close()

# ---- Print Results ----

print("\n1. Kolmogorov-Smirnov Test (KS Test) Results:")
print(f"GBM vs NASDAQ-100: D-stat = {ks_gbm.statistic:.4f}, p-value = {ks_gbm.pvalue:.4f}")
print(f"Merton vs NASDAQ-100: D-stat = {ks_merton.statistic:.4f}, p-value = {ks_merton.pvalue:.4f}")
print(f"Heston vs NASDAQ-100: D-stat = {ks_heston.statistic:.4f}, p-value = {ks_heston.pvalue:.4f}")

print("\n2. Anderson-Darling Test (AD Test) Results:")
print(f"GBM vs NASDAQ-100: AD-stat = {ad_gbm.statistic:.4f}, p-value ~ {ad_gbm.significance_level}")
print(f"Merton vs NASDAQ-100: AD-stat = {ad_merton.statistic:.4f}, p-value ~ {ad_merton.significance_level}")
print(f"Heston vs NASDAQ-100: AD-stat = {ad_heston.statistic:.4f}, p-value ~ {ad_heston.significance_level}")

print("\n3. Jarque-Bera Test (JB Test) Results - Skewness & Kurtosis:")
print(f"GBM: JB-stat = {jb_gbm.statistic:.4f}, p-value = {jb_gbm.pvalue:.4f}")
print(f"Merton: JB-stat = {jb_merton.statistic:.4f}, p-value = {jb_merton.pvalue:.4f}")
print(f"Heston: JB-stat = {jb_heston.statistic:.4f}, p-value = {jb_heston.pvalue:.4f}")

print("\n4. Cramér–von Mises Test (CVM Test) Results:")
print(f"GBM vs NASDAQ-100: W-stat = {cvm_gbm.statistic:.4f}, p-value = {cvm_gbm.pvalue:.4f}")
print(f"Merton vs NASDAQ-100: W-stat = {cvm_merton.statistic:.4f}, p-value = {cvm_merton.pvalue:.4f}")
print(f"Heston vs NASDAQ-100: W-stat = {cvm_heston.statistic:.4f}, p-value = {cvm_heston.pvalue:.4f}")

print("\nValidation completed! Check the new plots for visual insights.")
