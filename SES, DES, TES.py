import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

# Load datasets
exo = pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\advance data science\D1.1.4\archive\MJ_Days_Played_Exog_Data.csv")
nike = pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\advance data science\D1.1.4\archive\Nike_Historical_Stock_Price_Refined.csv")

# Merge on Date
merge = pd.merge(nike, exo, on="Date")

# Set index and sort
merge['Date'] = pd.to_datetime(merge['Date'])
merge.set_index("Date", inplace=True)
merge.sort_index(inplace=True)

# Fill missing values
clean = merge.asfreq('D').ffill()


ses = SimpleExpSmoothing(clean["Adj Close"]).fit(
smoothing_level=0.3,
optimized=False)
ses_fit=ses.fittedvalues
ses_for=ses.forecast(30)

des= Holt(clean["Adj Close"]).fit(
smoothing_level=0.3,
smoothing_trend=0.2)
des_fit=des.fittedvalues
des_for=des.forecast(30)

tes = ExponentialSmoothing(clean["Adj Close"],
trend="add",
seasonal="add",
seasonal_periods=365).fit()
tes_fit=tes.fittedvalues
tes_for=tes.forecast(30)

plt.plot(clean['Adj Close'], label='Actual', c="black")
plt.plot(ses_fit, label='SES fit', c="b")
plt.plot(ses_for, label='SES forecast', c="r")
plt.legend()
plt.show()

plt.plot(clean['Adj Close'], label='Actual', c="black")
plt.plot(des_fit, label='DES fit', c="b")
plt.plot(des_for, label='DES forecast', c="r")
plt.legend()
plt.show()

plt.plot(clean['Adj Close'], label='Actual', c="black")
plt.plot(tes_fit, label='TES fit', c="b")
plt.plot(tes_for, label='TES forecast', c="r")
plt.legend()
plt.show()
