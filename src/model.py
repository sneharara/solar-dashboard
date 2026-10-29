import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from db_setup import query_joined_hourly

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "output_model.joblib")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "model_comparison.csv")

def mape(y_true, y_pred):
    mask = y_true > 10.0 # Exclude near-zero night power
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def train_and_compare():
    df = query_joined_hourly()
    features = ["irradiation", "ambient_temperature", "module_temperature"]
    target = "total_ac_power"
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    }
    
    results = []
    trained_objects = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae_val = float(mean_absolute_error(y_test, preds))
        rmse_val = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2_val = float(r2_score(y_test, preds))
        mape_val = mape(y_test.values, preds)
        
        results.append({
            "model": name,
            "mae": round(mae_val, 2),
            "rmse": round(rmse_val, 2),
            "mape": round(mape_val, 2),
            "r2": round(r2_val, 4)
        })
        trained_objects[name] = (model, rmse_val)
        
    results_df = pd.DataFrame(results).sort_values("rmse")
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    results_df.to_csv(CSV_PATH, index=False)
    
    # Save best model by RMSE
    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_objects[best_model_name][0]
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model comparison saved. Best model '{best_model_name}' saved to {MODEL_PATH}.")
    return results_df

def flag_anomalies(df, model, std_threshold=1.0):
    features = ["irradiation", "ambient_temperature", "module_temperature"]
    df = df.copy()
    preds = model.predict(df[features])
    df["predicted_power"] = np.maximum(0.0, preds)
    df["residual"] = df["total_ac_power"] - df["predicted_power"]
    
    # Compute residual stats over daylight hours
    daylight = df[df["irradiation"] > 0.05]
    if len(daylight) > 0:
        mean_res = daylight["residual"].mean()
        std_res = daylight["residual"].std()
    else:
        mean_res, std_res = 0.0, 1.0
        
    # Anomaly condition: daylight AND large negative residual
    cutoff = mean_res - std_threshold * std_res
    df["is_anomaly"] = np.where((df["irradiation"] > 0.05) & (df["residual"] < cutoff), 1, 0)
    return df

if __name__ == "__main__":
    train_and_compare()
