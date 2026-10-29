import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from db_setup import query_joined_hourly

FORECAST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "forecast_model.joblib")

def create_forecasting_features(df):
    """
    Constructs time-series lag features and cyclical time encodings
    to support multi-horizon forward power forecasting.
    """
    df_feat = df.copy()
    
    # 15-minute lag
    df_feat['ac_power_lag1'] = df_feat['total_ac_power'].shift(1)
    # 1-hour lag (4 intervals)
    df_feat['ac_power_lag4'] = df_feat['total_ac_power'].shift(4)
    # 24-hour lag (96 intervals)
    df_feat['ac_power_lag96'] = df_feat['total_ac_power'].shift(96)
    
    # Cyclical hour features
    dt_series = pd.to_datetime(df_feat['date_time'])
    hour_val = dt_series.dt.hour + dt_series.dt.minute / 60.0
    df_feat['hour_sin'] = np.sin(2 * np.pi * hour_val / 24.0)
    df_feat['hour_cos'] = np.cos(2 * np.pi * hour_val / 24.0)
    
    # Rolling averages over past 1 hour (4 intervals)
    df_feat['rolling_ac_1h'] = df_feat['total_ac_power'].shift(1).rolling(4, min_periods=1).mean()
    
    return df_feat.dropna()

def train_forecaster():
    """
    Trains a forward-looking Random Forest Regressor and calculates residual error bounds.
    """
    df = query_joined_hourly()
    df_feat = create_forecasting_features(df)
    
    features = [
        'irradiation', 'ambient_temperature', 'module_temperature',
        'ac_power_lag1', 'ac_power_lag4', 'ac_power_lag96',
        'hour_sin', 'hour_cos', 'rolling_ac_1h'
    ]
    target = 'total_ac_power'
    
    X = df_feat[features]
    y = df_feat[target]
    
    # Chronological 80/20 train/test split to prevent temporal leakage
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    preds = np.maximum(0.0, model.predict(X_test))
    residuals = y_test.values - preds
    std_err = float(np.std(residuals))
    
    mae_val = float(mean_absolute_error(y_test, preds))
    rmse_val = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2_val = float(r2_score(y_test, preds))
    
    os.makedirs(os.path.dirname(FORECAST_MODEL_PATH), exist_ok=True)
    joblib.dump({
        'model': model,
        'features': features,
        'std_err': std_err,
        'metrics': {'mae': mae_val, 'rmse': rmse_val, 'r2': r2_val}
    }, FORECAST_MODEL_PATH)
    
    print(f"Forecast model trained. R2={r2_val:.4f}, MAE={mae_val:.2f} W, StdErr={std_err:.2f} W.")
    return model, features, std_err, {'mae': mae_val, 'rmse': rmse_val, 'r2': r2_val}

def load_forecaster():
    if os.path.exists(FORECAST_MODEL_PATH):
        return joblib.load(FORECAST_MODEL_PATH)
    return None

def get_forecast_predictions(df, horizon_intervals=96):
    """
    Generates multi-step forward predictions with 95% confidence intervals.
    """
    f_data = load_forecaster()
    if f_data is None:
        model, features, std_err, metrics = train_forecaster()
    else:
        model = f_data['model']
        features = f_data['features']
        std_err = f_data['std_err']
        metrics = f_data['metrics']
        
    df_feat = create_forecasting_features(df)
    if len(df_feat) == 0:
        return None
        
    # Take latest horizon_intervals rows
    recent = df_feat.tail(horizon_intervals).copy()
    preds = np.maximum(0.0, model.predict(recent[features]))
    
    recent['forecast_ac_power'] = preds
    # 95% Confidence Interval bounds (+/- 1.96 * std_err)
    recent['lower_bound'] = np.maximum(0.0, preds - 1.96 * std_err)
    recent['upper_bound'] = preds + 1.96 * std_err
    
    return recent, metrics, std_err

if __name__ == "__main__":
    train_forecaster()
