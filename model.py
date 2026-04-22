import yfinance as yf
import pandas as pd

stock = yf.download("AAPL", start="2024-01-01", end="2025-01-01")

print(stock.head())
print(stock.tail())
data = stock[['Close']]
print(data.head())
data['Prediction'] = data[['Close']].shift(-1)

print(data.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array(data.drop(['Prediction'], axis=1))[:-1]
y = np.array(data['Prediction'])[:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

print(prediction[:5])
future_price = model.predict([[data['Close'].iloc[-1]]])
print("Tomorrow Price Prediction:", future_price[0])
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(data['Close'])
plt.title("Stock Closing Price")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()