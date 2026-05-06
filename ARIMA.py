import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score

# Load datasets
nike = pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\advance data science\D1.1.4\archive\Nike_Historical_Stock_Price_Refined.csv")

# Set index and sort
nike['Date'] = pd.to_datetime(nike['Date'])
nike.set_index('Date', inplace=True)
nike.sort_index(inplace=True)

# Fill missing values
clean = nike.asfreq('B').ffill()

y = clean["Adj Close"]

#splitting
y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)

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

model = ARIMA(y_train, order=(1, 1, 1))
fit = model.fit()
forecast = fit.forecast(steps=len(y_test))
print(fit.summary())

#plot
plt.plot(y_test, label="Actual", c='blue')
plt.plot(forecast, label="Predicted", c='red')
plt.title("Nike Stock Price Prediction (ARIMA Model)")
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
print(f"ARIMA:MSE:{mse:.4f},RMSE:{rmse:.4f},MAE:{mae:.4f}, MAPE: {mape*100:.4f}%,R2:{r2:.4f}")


