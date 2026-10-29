import os
import joblib
import pandas as pd
import numpy as np

from db_setup import query_joined_hourly
from model import flag_anomalies, MODEL_PATH

EVAL_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "anomaly_evaluation.csv")

def evaluate_anomalies():
    if not os.path.exists(MODEL_PATH):
        print("Model file missing. Running train_and_compare...")
        from model import train_and_compare
        train_and_compare()
        
    model = joblib.load(MODEL_PATH)
    df = query_joined_hourly()
    
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    results = []
    
    for k in thresholds:
        flagged = flag_anomalies(df, model, std_threshold=k)
        daylight = flagged[flagged["irradiation"] > 0.05]
        
        y_true = daylight["known_fault"].values if "known_fault" in daylight.columns else np.zeros(len(daylight))
        y_pred = daylight["is_anomaly"].values
        
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        
        results.append({
            "std_threshold": k,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "false_positive_rate": round(fpr, 3),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        })
        
    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
    res_df.to_csv(EVAL_PATH, index=False)
    print(f"Anomaly evaluation completed and saved to {EVAL_PATH}.")
    return res_df

if __name__ == "__main__":
    evaluate_anomalies()
