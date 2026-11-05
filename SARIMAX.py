import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score


# Load datasets
exo = pd.read_csv(r"archive\MJ_Days_Played_Exog_Data.csv")
nike = pd.read_csv(r"archive\Nike_Historical_Stock_Price_Refined.csv")

# Merge on Date
merge = pd.merge(nike, exo, on="Date")

# Set index and sort
merge['Date'] = pd.to_datetime(merge['Date'])
merge.set_index('Date', inplace=True)
merge.sort_index(inplace=True)

# Fill missing values
clean = merge.asfreq('B').ffill()

X = clean["Exogeneous_Variable"]
y = clean["Adj Close"]

#splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#d(I)
d = y.diff().dropna()
d.plot()
plt.show()

#p(AR)
plot_pacf(d, lags=20)
plt.show()

#q(MA)
plot_acf(d, lags=20)
plt.show()

model = SARIMAX(y_train,
                exog=X_train,
                order=(1, 1, 1),
                seasonal_order=(1,1,1,7)) # The seasonal version (7-day cycle)
fit = model.fit()
forecast = fit.forecast(steps=len(y_test), exog=X_test)
print(fit.summary())

#plot
plt.plot(y_test, label="Actual", c='blue')
plt.plot(forecast, label="Predicted", c='red')
plt.title("Nike Stock Price Prediction (SARIMAX Model)")
plt.xlabel("Days in Test Set")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

#metrics
mse = mean_squared_error(y_test, forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, forecast)
mape = mean_absolute_percentage_error(y_test, forecast)
r2 = r2_score(y_test, forecast)
print(f"SARIMAX:MSE:{mse:.4f},RMSE:{rmse:.4f},MAE:{mae:.4f}, MAPE: {mape*100:.4f}%,R2:{r2:.4f}")

