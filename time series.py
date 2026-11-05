import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load data
df = pd.read_csv(r"archive\MLTempDataset1.csv")

# Datetime conversion
df["Datetime"] = pd.to_datetime(df["Datetime"])

# Cleaning
df.drop(columns=["Unnamed: 0"], inplace=True)
df.drop_duplicates(inplace=True)

# Set index and frequency
df.set_index("Datetime", inplace=True)
df.sort_index(inplace=True)
df = df.asfreq("h")

# Handle missing values
df["Hourly_Temp"] = df["Hourly_Temp"].interpolate()

df.info()
print(df.head())
print(df.tail())
print(df.describe())

# Plot time series
df["Hourly_Temp"].plot(figsize=(10,5),
                       marker="v",
                       ms=10,
                       mfc="#ed2602",
                       mec="#080101",
                       ls="dotted",
                       lw="1",
                       color=("#12fc19"))
plt.title("Time Series Room Temperature Data",
          fontsize=20,
          family="Arial",
          fontweight="bold",
          color="#080101")
plt.xlabel("DateTime",
          fontsize=20,
          family="Arial",
          fontweight="bold",
          color="#12fc19")
plt.ylabel("Hourly Temp",
          fontsize=20,
          family="Arial",
          fontweight="bold",
          color="#ed2602")
plt.gcf().autofmt_xdate()
plt.subplots_adjust(bottom=0.2)
plt.show()

# Additive decomposition
additive = seasonal_decompose(
    df["Hourly_Temp"],
    model="additive",
    period=24
)

additive.plot()
plt.gcf().autofmt_xdate() #formating the date so it won't overlap
plt.show()

# Multiplicative decomposition (only if valid)
if (df["Hourly_Temp"] <= 0).any():
    print("Multiplicative model not suitable due to non-positive values")
else:
    multiplicative = seasonal_decompose(
        df["Hourly_Temp"],
        model="multiplicative",
        period=24
    )
    multiplicative.plot()
    plt.gcf().autofmt_xdate()
    plt.show()
