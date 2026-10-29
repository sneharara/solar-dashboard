import pandas as pd
import numpy as np

RATED_CAPACITY_KW = 30.0 # Rated plant AC capacity in kW (D-5 patch comment: peak irradiation ~1.22 kW/m2)

def compute_performance_ratio(df, rated_capacity_kw=RATED_CAPACITY_KW):
    # P_theo = rated_capacity_kw * 1000 * irradiation
    df = df.copy()
    p_theo = rated_capacity_kw * 1000.0 * df["irradiation"]
    
    # Avoid div by zero
    pr = np.where(df["irradiation"] > 0.05, df["total_ac_power"] / np.maximum(p_theo, 1.0), np.nan)
    pr = np.clip(pr, 0.0, 1.5)
    df["performance_ratio"] = pr
    return df

def compute_cuf(total_ac_energy_kwh, rated_capacity_kw=RATED_CAPACITY_KW, total_hours=816):
    if rated_capacity_kw <= 0 or total_hours <= 0:
        return 0.0
    return float(total_ac_energy_kwh / (rated_capacity_kw * total_hours))

def add_temperature_loss(df):
    df = df.copy()
    temp_diff = np.maximum(0.0, df["module_temperature"] - 25.0)
    df["temperature_loss_pct"] = temp_diff * 0.4 # 0.4% per degree C above 25C
    return df

def daily_summary(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date_time"]).dt.date
    summary = df.groupby("date").agg(
        total_ac_kwh=("total_ac_power", lambda x: (x * 0.25 / 1000.0).sum()),
        mean_pr=("performance_ratio", "mean"),
        max_irradiation=("irradiation", "max"),
        mean_module_temp=("module_temperature", "mean")
    ).reset_index()
    return summary
