import json
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
import base64

def lambda_handler(event, context):

    # Extract parameters
    max_time_window = event.get("max_time_window", 1)
    ticker_symbol = event.get("ticker_symbol", "AAPL")
    interval = event.get("interval", "1h")
    stock_data = event.get("stock_data", {})

    print(f"Max Time Window: {max_time_window}")
    print(f"Ticker Symbol: {ticker_symbol}")
    print(f"Interval: {interval}")

    # Load test data into DataFrame
    stock_data_df = pd.DataFrame(stock_data)

    # Prepare data for linear regression
    stock_data_df['Time'] = np.arange(len(stock_data_df))
    X = stock_data_df[['Time']]
    y = stock_data_df['Close']

    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future stock prices
    future_times = pd.DataFrame(np.arange(len(stock_data_df), len(stock_data_df) + 5), columns=['Time'])
    future_prices = model.predict(future_times)

    # Serialize the model
    model_serialized = base64.b64encode(pickle.dumps(model)).decode('utf-8')

    # Example processing
    message = f"Hello from Lambda! Received ticker {ticker_symbol} with a time window of {max_time_window} and interval {interval}."

    # Return a response
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": message,
                "max_time_window": max_time_window,
                "ticker_symbol": ticker_symbol,
                "interval": interval,
                "predicted_prices": future_prices.tolist(),
                "model": model_serialized
            }
        ),
    }
