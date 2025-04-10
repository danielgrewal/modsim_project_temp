import numpy as np
import pandas as pd
import statsmodels.api as sm

# -------------------------------
# Load NASDAQ-100 Real Data
# -------------------------------
csv_filename = "nasdaq100_data.csv"
data = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
data.dropna(inplace=True)

# -------------------------------
# Compute Daily Log Returns
# -------------------------------
data['log_return'] = np.log(data["Close"] / data["Close"].shift(1))
data.dropna(inplace=True)

# -------------------------------
# GBM Parameter Estimation
# -------------------------------
# Daily drift and volatility estimates:
mu_daily = data['log_return'].mean()
sigma_daily = data['log_return'].std()

# Annualized values (approx. 252 trading days/year)
mu_gbm = mu_daily * 252
sigma_gbm = sigma_daily * np.sqrt(252)

print()
print("GBM Parameters:")
print("----------------")
print(f"Annual drift (μ): {mu_gbm:.4f}")
print(f"Annual volatility (σ): {sigma_gbm:.4f}\n")

# -------------------------------
# Merton Jump-Diffusion Parameters
# -------------------------------
# Here we define jumps as days when the absolute log return exceeds 3 times the daily standard deviation.
threshold = 3 * sigma_daily
jump_mask = data['log_return'].abs() > threshold
num_jump_days = jump_mask.sum()

# Annual jump intensity (λ): jumps per day scaled to annual frequency.
lambda_jump = (num_jump_days / len(data)) * 252

# For jump sizes, we compute the mean and standard deviation of the returns considered as jumps.
jump_returns = data.loc[jump_mask, 'log_return']
if len(jump_returns) > 0:
    mu_jump = jump_returns.mean()
    sigma_jump = jump_returns.std()
else:
    mu_jump = 0.0
    sigma_jump = 0.0

print("Merton Jump-Diffusion Parameters:")
print("----------------------------------")
print(f"Annual jump intensity (λ): {lambda_jump:.4f}")
print(f"Average jump size (μ_jump): {mu_jump:.4f}")
print(f"Jump volatility (σ_jump): {sigma_jump:.4f}\n")

# -------------------------------
# Heston Stochastic Volatility Parameters
# -------------------------------
# Since we only have daily returns, we estimate the daily variance as a proxy.
# To reduce noise, we use a rolling variance with a window (e.g., 21 trading days).
window = 21
data['variance'] = data['log_return'].rolling(window=window).var()
data.dropna(subset=['variance'], inplace=True)

# v0: use the most recent estimated variance as the initial variance.
v0 = data['variance'].iloc[-1]

# θ (theta): long-run variance estimated as the mean of the rolling variances.
theta = data['variance'].mean()

# Estimate κ (kappa) and ξ (xi) via a simplified regression.
# The Heston variance dynamics (in continuous time) are approximated by:
#   dv = κ (θ - v) dt + ξ √v dW.
# In discrete time, we compute the daily change in variance (dv) and regress
#   dv ~ (θ - v) dt.
dt = 1 / 252  # daily time step

data['dvariance'] = data['variance'].diff()  # change in variance day-to-day
data.dropna(subset=['dvariance'], inplace=True)

# Prepare the regression variable: X = (θ - v)*dt
X = (theta - data['variance']) * dt
X = sm.add_constant(X)
y = data['dvariance']
model = sm.OLS(y, X).fit()

# The slope gives an estimate of κ * dt; hence, κ = slope / dt.
# Use .iloc to access the slope value by position.
kappa = model.params.iloc[1] / dt

# Estimate ξ using the standard deviation of the residuals adjusted by √(mean variance).
xi = model.resid.std() / np.sqrt(data['variance'].mean())

# Estimate ρ: the correlation between daily log returns and the change in variance.
rho = data['log_return'].corr(data['dvariance'])

print("Heston Model Parameters (Estimated):")
print("-------------------------------------")
print(f"Initial variance (v0): {v0:.4f}")
print(f"Long-run variance (θ): {theta:.4f}")
print(f"Mean reversion speed (κ): {kappa:.4f}")
print(f"Volatility of volatility (ξ): {xi:.4f}")
print(f"Correlation (ρ): {rho:.4f}")
