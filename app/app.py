import json
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
import base64

def lambda_handler(event, context):
    # Extract parameters
    user = event.get("user", "default")
    model_id = event.get("model_id", "114514")
    ticker_symbol = event.get("ticker_symbol", "AAPL")
    max_time_window = event.get("max_time_window", 1)
    interval = event.get("interval", "1h")
    stock_data = event.get("data", {})

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
    sample_future_times = pd.DataFrame(np.arange(len(stock_data_df), len(stock_data_df) + 5), columns=['Time'])
    sample_future_prices = model.predict(sample_future_times)

    # Serialize the model and set path
    model_serialized = base64.b64encode(pickle.dumps(model)).decode('utf-8')
    model_path = f'models/{user}/{ticker_symbol}/{model_id}.pkl'

    # Return a response
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "user": user,
                "ticker_symbol": ticker_symbol,
                "max_time_window": max_time_window,
                "interval": interval,
                "predicted_prices": sample_future_prices.tolist(),
                "model": model_serialized,
                "model_path": model_path
            }
        ),
    }
