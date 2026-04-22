import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("Stock Price Prediction System")

stock_name = st.text_input("Enter Stock Symbol", "AAPL")

if st.button("Predict"):

    # Download stock data
    stock_data = yf.download(stock_name, start="2024-01-01", end="2025-01-01")

    # Check if data exists
    if stock_data.empty:
        st.error("No data found. Please enter a valid stock symbol.")
    else:

        # Fix MultiIndex columns if present
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)

        st.subheader("Stock Data")
        st.write(stock_data.tail())

        # Create working dataframe
        data = stock_data[['Close']].copy()

        # Moving averages
        data['50_MA'] = data['Close'].rolling(50).mean()
        data['200_MA'] = data['Close'].rolling(200).mean()

        # Prediction column
        data['Prediction'] = data['Close'].shift(-1)

        # Prepare ML data
        X = np.array(data[['Close']])[:-1]
        y = np.array(data['Prediction'])[:-1]

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Model accuracy
        accuracy = model.score(X_test, y_test)

        # Current and future price
        current_price = float(data['Close'].iloc[-1])
        future_price = model.predict([[current_price]])
        future_price = float(future_price[0])

        st.subheader("Prediction Result")
        st.write("Current Price:", round(current_price, 2))
        st.write("Predicted Next Day Price:", round(future_price, 2))
        st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

        # Trend prediction
        if future_price > current_price:
            st.success("Stock price is predicted to increase 📈")
        else:
            st.error("Stock price is predicted to decrease 📉")

        # Closing price graph
        st.subheader("Closing Price Graph")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data['Close'], label='Close Price')
        ax.set_title(f"{stock_name} Closing Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()

        st.pyplot(fig)

        # Moving average chart
        st.subheader("Moving Average Chart")
        st.line_chart(data[['Close', '50_MA', '200_MA']])