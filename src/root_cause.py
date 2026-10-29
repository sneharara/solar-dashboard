import pandas as pd
import numpy as np

def assign_root_cause(row):
    if row.get("is_anomaly", 0) == 0:
        return "Normal Operation"
        
    p_actual = row["total_ac_power"]
    p_pred = row.get("predicted_power", p_actual)
    irr = row["irradiation"]
    t_mod = row["module_temperature"]
    t_amb = row["ambient_temperature"]
    res = row["residual"]
    
    # 1. Inverter Fault
    if p_actual < 0.15 * p_pred and irr > 0.3:
        return "Possible Inverter Fault"
        
    # 2. Shading / Soiling
    if irr > 0.4 and abs(res) / max(p_pred, 1.0) < 0.6:
        return "Possible Shading or Soiling"
        
    # 3. Overheating
    if (t_mod - t_amb) > 25.0:
        return "Possible Overheating / Thermal Loss"
        
    return "Unexplained - Manual Inspection"

def summarize_causes(df):
    df = df.copy()
    if "likely_cause" not in df.columns:
        df["likely_cause"] = df.apply(assign_root_cause, axis=1)
        
    anomalies = df[df["is_anomaly"] == 1]
    summary = anomalies["likely_cause"].value_counts().reset_index()
    summary.columns = ["likely_cause", "count"]
    return summary
