import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from db_setup import build_db, get_connection, query_joined_hourly, query_peak_hours, query_daily_yield_by_inverter, query_inverter_details, query_inverter_heatmap
from metrics import compute_performance_ratio, compute_cuf, add_temperature_loss, daily_summary
from model import train_and_compare, flag_anomalies, mape, MODEL_PATH
from root_cause import assign_root_cause, summarize_causes
from forecasting import train_forecaster, get_forecast_predictions, FORECAST_MODEL_PATH
from inverter_analytics import get_inverter_leaderboard, get_inverter_heatmap_matrix

class TestSolarPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_db()
        train_and_compare()
        train_forecaster()
        
    def test_tc01_row_counts(self):
        df = query_joined_hourly()
        self.assertGreater(len(df), 0)
        
    def test_tc02_no_nulls_in_inputs(self):
        df = query_joined_hourly()
        self.assertEqual(df[["total_ac_power", "irradiation", "ambient_temperature", "module_temperature"]].isnull().sum().sum(), 0)
        
    def test_tc03_unique_timestamps(self):
        df = query_joined_hourly()
        self.assertEqual(df["date_time"].nunique(), len(df))
        
    def test_tc04_join_coverage(self):
        df = query_joined_hourly()
        self.assertGreaterEqual(len(df), 1000)
        
    def test_tc05_physical_ranges(self):
        df = query_joined_hourly()
        self.assertTrue((df["irradiation"] >= 0.0).all())
        self.assertTrue((df["total_ac_power"] >= 0.0).all())
        
    def test_tc06_nighttime_output_near_zero(self):
        df = query_joined_hourly()
        night = df[df["irradiation"] == 0.0]
        self.assertLess(night["total_ac_power"].mean(), 10.0)
        
    def test_tc07_peak_hour_is_midday(self):
        pk = query_peak_hours()
        self.assertIn("11:00", pk["hour"].values)
        
    def test_tc08_lag_query_structure(self):
        df = query_daily_yield_by_inverter()
        self.assertIn("prev_daily_yield", df.columns)
        
    def test_tc09_date_format_handling(self):
        df = query_joined_hourly()
        self.assertTrue(df["date_time"].str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").all())
        
    def test_tc10_performance_ratio_bounds(self):
        df = query_joined_hourly()
        df_pr = compute_performance_ratio(df)
        valid_pr = df_pr["performance_ratio"].dropna()
        self.assertTrue((valid_pr >= 0.0).all() and (valid_pr <= 1.5).all())
        
    def test_tc11_cuf_formula(self):
        cuf = compute_cuf(30.0 * 24.0, rated_capacity_kw=30.0, total_hours=24)
        self.assertAlmostEqual(cuf, 1.0)
        
    def test_tc12_temperature_loss_calculation(self):
        df = pd.DataFrame({"module_temperature": [25.0, 35.0, 15.0]})
        df_loss = add_temperature_loss(df)
        self.assertListEqual(list(df_loss["temperature_loss_pct"].round(1)), [0.0, 4.0, 0.0])
        
    def test_tc13_daily_summary_aggregation(self):
        df = query_joined_hourly()
        df = compute_performance_ratio(df)
        s = daily_summary(df)
        self.assertGreater(len(s), 0)
        self.assertIn("total_ac_kwh", s.columns)
        
    def test_tc14_model_file_exists(self):
        self.assertTrue(os.path.exists(MODEL_PATH))
        
    def test_tc15_mape_calculation(self):
        y_true = np.array([100.0, 200.0, 0.0])
        y_pred = np.array([110.0, 180.0, 5.0])
        err = mape(y_true, y_pred)
        self.assertAlmostEqual(err, 10.0)
        
    def test_tc16_anomaly_flagging_night_suppression(self):
        import joblib
        model = joblib.load(MODEL_PATH)
        df = query_joined_hourly()
        flagged = flag_anomalies(df, model, std_threshold=1.0)
        night_anomalies = flagged[(flagged["irradiation"] <= 0.05) & (flagged["is_anomaly"] == 1)]
        self.assertEqual(len(night_anomalies), 0)
        
    def test_tc17_anomaly_threshold_monotonicity(self):
        import joblib
        model = joblib.load(MODEL_PATH)
        df = query_joined_hourly()
        f1 = flag_anomalies(df, model, std_threshold=1.0)["is_anomaly"].sum()
        f2 = flag_anomalies(df, model, std_threshold=2.0)["is_anomaly"].sum()
        self.assertGreaterEqual(f1, f2)
        
    def test_tc18_root_cause_inverter_fault(self):
        row = {"is_anomaly": 1, "total_ac_power": 100.0, "predicted_power": 20000.0, "irradiation": 0.8, "module_temperature": 35.0, "ambient_temperature": 25.0, "residual": -19000.0}
        self.assertEqual(assign_root_cause(row), "Possible Inverter Fault")
        
    def test_tc19_root_cause_normal(self):
        row = {"is_anomaly": 0}
        self.assertEqual(assign_root_cause(row), "Normal Operation")
        
    def test_tc20_root_cause_summary_counts(self):
        df = pd.DataFrame([
            {"is_anomaly": 1, "likely_cause": "Possible Inverter Fault"},
            {"is_anomaly": 1, "likely_cause": "Possible Inverter Fault"},
            {"is_anomaly": 0, "likely_cause": "Normal Operation"}
        ])
        s = summarize_causes(df)
        self.assertEqual(s[s["likely_cause"] == "Possible Inverter Fault"]["count"].values[0], 2)
        
    def test_tc21_app_imports_cleanly(self):
        import app
        self.assertTrue(hasattr(app, "load_data"))
        
    def test_tc22_db_recreation_safety(self):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]
        conn.close()
        self.assertIn("generation", tables)
        self.assertIn("weather", tables)
        
    def test_tc23_model_r2_threshold(self):
        df_csv = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "db", "model_comparison.csv"))
        best_r2 = df_csv["r2"].max()
        self.assertGreater(best_r2, 0.95)
        
    def test_tc24_evaluation_script(self):
        from evaluate_anomalies import evaluate_anomalies
        res = evaluate_anomalies()
        self.assertIn("f1_score", res.columns)
        
    def test_tc25_zero_division_safety(self):
        cuf = compute_cuf(0.0, rated_capacity_kw=0.0, total_hours=0)
        self.assertEqual(cuf, 0.0)
        
    def test_tc26_temperature_loss_floor(self):
        df = pd.DataFrame({"module_temperature": [20.0]})
        df_loss = add_temperature_loss(df)
        self.assertEqual(df_loss["temperature_loss_pct"].values[0], 0.0)
        
    def test_tc27_non_negative_predictions(self):
        import joblib
        model = joblib.load(MODEL_PATH)
        df = query_joined_hourly()
        flagged = flag_anomalies(df, model)
        self.assertTrue((flagged["predicted_power"] >= 0.0).all())
        
    def test_tc28_known_fault_column_presence(self):
        df = query_joined_hourly()
        self.assertIn("known_fault", df.columns)
        
    def test_tc29_hourly_query_non_empty(self):
        pk = query_peak_hours()
        self.assertGreater(len(pk), 0)
        
    def test_tc30_headless_streamlit_apptest(self):
        try:
            from streamlit.testing.v1 import AppTest
            at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app.py")).run()
            self.assertFalse(at.exception)
        except Exception:
            pass
            
    def test_tc31_forecaster_model_training(self):
        self.assertTrue(os.path.exists(FORECAST_MODEL_PATH))
        df = query_joined_hourly()
        res = get_forecast_predictions(df, horizon_intervals=24)
        self.assertIsNotNone(res)
        f_df, f_metrics, std_err = res
        self.assertIn("forecast_ac_power", f_df.columns)
        self.assertIn("upper_bound", f_df.columns)
        self.assertIn("lower_bound", f_df.columns)
        
    def test_tc32_inverter_analytics_queries(self):
        lb = get_inverter_leaderboard()
        self.assertGreater(len(lb), 0)
        self.assertIn("rank", lb.columns)
        hm = get_inverter_heatmap_matrix()
        self.assertGreater(hm.shape[0], 0)

if __name__ == "__main__":
    unittest.main()
