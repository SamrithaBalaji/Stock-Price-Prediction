import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Download stock data
stock = yf.download("INFY.NS", start="2020-01-01", end="2024-01-01")
print(stock.head())

stock.to_csv("data/stock_data.csv")

print(stock.info())
print(stock.isnull().sum())

plt.figure(figsize=(10,5))
plt.plot(stock['Close'])
plt.title("Stock Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.savefig("outputs/stock_trend.png")
plt.show()

stock['MA50'] = stock['Close'].rolling(window=50).mean()

plt.figure(figsize=(10,5))
plt.plot(stock['Close'], label='Close')
plt.plot(stock['MA50'], label='MA50')
plt.legend()
plt.savefig("outputs/moving_average.png")
plt.show()

data = stock[['Close']]
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)
X = []
y = []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)
print("X shape:", X.shape)
print("y shape:", y.shape)

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

# reshape data
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

plt.figure(figsize=(10,5))

plt.plot(y_test_actual, label="Actual Price")
plt.plot(predictions, label="Predicted Price")

plt.title("Actual vs Predicted Stock Price")
plt.xlabel("Time")
plt.ylabel("Price")

plt.legend()
plt.savefig("outputs/actual_vs_predicted.png")
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

mse = mean_squared_error(y_test_actual, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, predictions)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)