import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define parameters
ticker = "^NDX"  # NASDAQ-100 index
start_date = "2015-01-01"  # Start date for historical data
end_date = "2025-03-15"  # End date (recent)

# Fetch data from Yahoo Finance
nasdaq_data = yf.download(ticker, start=start_date, end=end_date)

# Select relevant columns (excluding Adj Close)
nasdaq_data = nasdaq_data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Drop any missing values
nasdaq_data = nasdaq_data.dropna()

# Save to CSV
csv_filename = "nasdaq100_data.csv"
nasdaq_data.to_csv(csv_filename)

# Generate a summary
summary_stats = nasdaq_data.describe()
print(summary_stats)

# Plot the Close price over time and save it
plt.figure(figsize=(12, 6))
plt.plot(nasdaq_data.index, nasdaq_data["Close"], label="NASDAQ-100 Close Price", color='blue')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("NASDAQ-100 Historical Price")
plt.legend()
plt.grid()

# Save the plot instead of displaying it
plot_filename = "nasdaq100_price_trend.png"
plt.savefig(plot_filename, dpi=300)  # High-quality save
plt.close()

print(f"Data saved to: {csv_filename}")
print(f"Plot saved to: {plot_filename}")
