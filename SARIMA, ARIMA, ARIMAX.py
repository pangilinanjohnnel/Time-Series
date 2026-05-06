import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load datasets
exo = pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\advance data science\D 1.1.4\archive\MJ_Days_Played_Exog_Data.csv")
nike = pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\advance data science\D 1.1.4\archive\Nike_Historical_Stock_Price_Refined.csv")

# Merge on Date
merge = pd.merge(nike, exo, on="Date")

# Set index and sort
merge['Date'] = pd.to_datetime(merge['Date'])
merge.set_index('Date', inplace=True)
merge.sort_index(inplace=True)

# Fill missing values
clean = merge.asfreq('D').ffill()

# Check
print(clean.head())
print(clean.tail())

# Plot
clean.plot()
plt.title("Nike Stock Prices and Exogenous Variable")
plt.xlabel("Date")
plt.ylabel("Values")
plt.show()

# ACF & PACF
plt.figure(figsize=(12,4))
plot_acf(clean["Adj Close"], lags=40)
plt.show()

plt.figure(figsize=(12,4))
plot_pacf(clean["Adj Close"], lags=40)
plt.show()

# ARIMA
model_arima = ARIMA(clean["Adj Close"], order=(1,1,1))
fit_arima = model_arima.fit()
print(fit_arima.summary())

# SARIMA
model_sarima = SARIMAX(clean["Adj Close"],
                       order=(1,1,1),
                       seasonal_order=(1,1,1,12))
fit_sarima = model_sarima.fit()
print(fit_sarima.summary())

# SARIMAX with exogenous variable
model_sarimax = SARIMAX(clean["Adj Close"],
                        order=(1,1,1),
                        seasonal_order=(1,1,1,12),
                        exog=clean[['Exogeneous_Variable']])
fit_sarimax = model_sarimax.fit()
print(fit_sarimax.summary())

# Predictions
pred_arima = fit_arima.predict(start=0, end=len(clean)-1)
pred_sarima = fit_sarima.predict(start=0, end=len(clean)-1)
pred_sarimax = fit_sarimax.predict(start=0, end=len(clean)-1, exog=clean[["Exogeneous_Variable"]])

# Use the actual values
true = clean["Adj Close"]

# ARIMA
mse = mean_squared_error(true, pred_arima)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true, pred_arima)
print("ARIMA:", mse, rmse, mae)

# SARIMA
mse = mean_squared_error(true, pred_sarima)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true, pred_sarima)
print("SARIMA:", mse, rmse, mae)

# SARIMAX
mse = mean_squared_error(true, pred_sarimax)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true, pred_sarimax)
print("SARIMAX:", mse, rmse, mae)

