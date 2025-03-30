import json
import pandas as pd
import numpy as np
import pickle
import base64
import boto3
import optuna
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error  # For model evaluation
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Define the Optuna objective function
def objective(trial, train_y, test_y):
    try:
        # Suggest hyperparameters for SARIMAX
        p = trial.suggest_int('p', 0, 2)  # AR order (p)
        d = trial.suggest_int('d', 0, 1)  # Differencing order (d)
        q = trial.suggest_int('q', 0, 2)  # MA order (q)
        
        seasonal_p = trial.suggest_int('seasonal_p', 0, 1)  # Seasonal AR order (p)
        seasonal_d = trial.suggest_int('seasonal_d', 0, 1)  # Seasonal differencing order (d)
        seasonal_q = trial.suggest_int('seasonal_q', 0, 2)  # Seasonal MA order (q)
        seasonal_s = trial.suggest_int('seasonal_s', 24, 24)  # Seasonal period (s), fixed at 24 for daily seasonality
        
        order = (p, d, q)
        seasonal_order = (seasonal_p, seasonal_d, seasonal_q, seasonal_s)
        
        # Fit the SARIMAX model without exogenous variables
        sarimax_model = SARIMAX(
            train_y,                    # Endogenous variable (target)
            order=order,                # AR, I, MA order (p, d, q)
            seasonal_order=seasonal_order,  # Seasonal components (p, d, q, s)
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        # Fit the model
        sarimax_model_fitted = sarimax_model.fit(disp=False)

        # Predict for the test set
        n_periods = len(test_y)
        sample_future_prices = sarimax_model_fitted.predict(start=len(train_y), end=len(train_y) + n_periods - 1)

        # Calculate Mean Absolute Error (MAE) for this model
        mae = mean_absolute_error(test_y, sample_future_prices)
        
        return mae  # Return the error metric to be minimized

    except Exception as e:
        print(f"Error in objective function: {e}")
        return float('inf')  # In case of an error, return a very high error value to ignore this trial

# Define the Optuna study and optimization function
def lambda_handler(event, context):
    try: 
        # Extract parameters
        user = event.get("user", "default")
        model_id = event.get("model_id", "114514")
        ticker_symbol = event.get("ticker_symbol", "AAPL")
        max_time_window = event.get("max_time_window", 1)
        interval = event.get("interval", "1h")
        stock_data = event.get("data", {})

        # Load test data into DataFrame
        stock_data_df = pd.DataFrame(stock_data)

        # Ensure 'Close' column exists
        if 'Close' not in stock_data_df:
            raise ValueError("Missing 'Close' column in stock data")

        # Prepare the target variable (y)
        y = stock_data_df['Close']

        # Train-test split (80% for training, 20% for testing)
        train_size = int(len(y) * 0.8)
        train_y, test_y = y[:train_size], y[train_size:]

        # Create the Optuna study to optimize the objective function
        # Use TPESampler to improve convergence speed
        sampler = TPESampler()

        # Use a pruning strategy to speed up convergence
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=20, interval_steps=5)

        study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)  # Minimize MAE
        
        # Run the optimization with the fast-converging strategy
        study.optimize(lambda trial: objective(trial, train_y, test_y), n_trials=10)  # n trials (can be adjusted)

        # Print best parameters found by Optuna
        print(f"Best parameters: {study.best_params}")
        print(f"Best MAE: {study.best_value}")

        # Best model based on the optimal parameters
        best_params = study.best_params
        p, d, q = best_params['p'], best_params['d'], best_params['q']
        seasonal_p, seasonal_d, seasonal_q, seasonal_s = best_params['seasonal_p'], best_params['seasonal_d'], best_params['seasonal_q'], best_params['seasonal_s']

        # Fit the SARIMAX model with the best parameters
        order = (p, d, q)
        seasonal_order = (seasonal_p, seasonal_d, seasonal_q, seasonal_s)
        
        sarimax_model = SARIMAX(
            train_y,                    
            order=order,                
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Fit the best model
        sarimax_model_fitted = sarimax_model.fit(disp=False)

        # Serialize the best model and save in S3
        model_serialized = base64.b64encode(pickle.dumps(sarimax_model_fitted)).decode('utf-8')
        
        # Define S3 path
        s3_bucket = "goldrush-main-12705"
        model_path = f'models/{user}/{ticker_symbol}/{model_id}.pkl'

        # Upload model to S3
        s3_client = boto3.client('s3')
        s3_client.put_object(Bucket=s3_bucket, Key=model_path, Body=model_serialized.encode())

        # Sample prediction: Forecast the next 'n' steps
        forecast_steps = 10  # Predict the next 10 time steps (can be adjusted)
        forecast = sarimax_model_fitted.predict(start=len(train_y), end=len(train_y) + forecast_steps - 1)

        # Return the response with the best model and evaluation metrics
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "user": user,
                    "ticker_symbol": ticker_symbol,
                    "max_time_window": max_time_window,
                    "interval": interval,
                    "best_mae": study.best_value,
                    "key": model_path,  # Path to the saved model
                    "model": None,  # Serialized model
                    "best_params": study.best_params,  # Return best parameters found by Optuna
                    "sample_prediction": forecast.tolist(),  # Sample prediction
                }
            ),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e), "input": event}),
        }
