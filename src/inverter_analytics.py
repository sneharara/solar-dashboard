import os
import pandas as pd
import numpy as np
from db_setup import query_inverter_details, query_inverter_heatmap

def get_inverter_leaderboard(db_path=None):
    if db_path:
        df = query_inverter_details(db_path)
    else:
        df = query_inverter_details()
        
    df["rank"] = range(1, len(df) + 1)
    df["status"] = np.where(df["dc_ac_ratio"] > 11.5, "Scaling Artifact / Degraded", "Normal")
    return df

def get_inverter_heatmap_matrix(db_path=None):
    if db_path:
        df = query_inverter_heatmap(db_path)
    else:
        df = query_inverter_heatmap()
        
    pivot_df = df.pivot(index="source_key", columns="hour", values="avg_ac_power").fillna(0.0)
    return pivot_df

if __name__ == "__main__":
    lb = get_inverter_leaderboard()
    print("Inverter Leaderboard Head:")
    print(lb.head())
