### Intelligent Solar PV Performance Monitoring, Telemetry & Anomaly Detection System

An enterprise-grade, interactive solar asset management dashboard for real-time SCADA telemetry, multi-parameter variable information presentation, multi-model expected-power prediction, residual underperformance anomaly detection, 24h–72h forward generation forecasting with confidence bounds, and automated O&M financial dispatch logging.

---

#### 🌟 Advanced Key Features

*   **🎛️ Variable Information Studio (iSolarCloud++ View)**:
    * Dynamic variable selector to overlay any combination of parameters: Actual AC Power, Expected AC Power, Total DC Power, Irradiation, Module Temperature, Ambient Temperature, Performance Ratio (%), and Residual Loss.
    * Dual Y-Axis interactive Plotly graphs with multi-granularity aggregation toggles (Raw 15-Min, Hourly Average, Daily Aggregate).
    * Live anomaly markers and overlay toggles.

*   **⚡ Inverter Fleet Analytics & Ranking Matrix**:
    * Per-inverter DC vs AC generation performance leaderboard.
    * 24-Hour 22-Inverter Generation Intensity Heatmap (Inverter ID vs Hour of Day).
    * Individual inverter health and yield comparison bar charts.

*   **🔮 Predictive Forward Generation Forecasting (24h - 72h)**:
    * Time-series Random Forest forecasting combining historical lags ($P_{t-15\text{min}}$, $P_{t-24\text{hr}}$) and weather predictions.
    * 95% Confidence Error Bands ($\pm 1.96 \cdot \sigma_{\text{residual}}$) showing projected upper and lower generation limits.
    * Temporal validation accuracy metrics ($R^2 = 0.9946$, $\text{MAE} = 291.6\text{ W}$).

*   **💼 Asset Financial Impact & Automated O&M Dispatch**:
    * Real-time monetary revenue loss quantification ($ / ₹ / €) based on customizable feed-in tariffs.
    * Environmental offset metrics ($CO_2$ emissions avoided in kg).
    * Prioritized automated technical work order tickets (High/Medium/Low priority badges) with recommended technician maintenance actions.

*   **🎯 Model Comparison & Custom Cross-Plotter Studio**:
    * Multi-model comparison matrix (Linear Regression, Random Forest, XGBoost).
    * Synthetic fault-injection benchmark evaluation sweep ($k = 1.0\sigma \dots 3.0\sigma$).
    * Custom X vs Y Variable Scatter Plotter with OLS trendline fitting.

---

#### 📁 Project Directory Structure

```text
solar-dashboard/
├── data/                          # Raw generation & weather CSVs
├── db/                            # SQLite DB, trained models, comparison CSVs
│   ├── solar.db                   # SQLite DB with primary key and index retention
│   ├── output_model.joblib        # Regression model for expected power
│   ├── forecast_model.joblib      # Time-series forecasting model
│   ├── model_comparison.csv       # Model comparison metrics
│   └── anomaly_evaluation.csv     # Synthetic fault benchmark evaluation
├── src/
│   ├── db_setup.py                # Database schema, ISO date parsing, & SQL queries
│   ├── metrics.py                 # Performance Ratio (PR) & CUF calculations
│   ├── model.py                   # Multi-model regression & residual anomaly detection
│   ├── forecasting.py             # Forward forecasting engine with confidence bounds
│   ├── inverter_analytics.py      # Inverter leaderboard & 24h heatmap builder
│   ├── evaluate_anomalies.py      # Benchmark threshold sweep evaluation
│   ├── root_cause.py              # Rule-based diagnostic labeling
│   └── generate_sample_data.py    # Synthetic fault-injection dataset generator
├── tests/
│   └── test_pipeline.py           # 32-case automated unit test suite (32/32 passing)
├── app.py                         # Advanced Streamlit Dashboard (iSolarCloud++ theme)
├── README.md                      # Project documentation
├── patches.md                     # Defect log & patch documentation
└── requirements.txt               # Pinned environment dependencies
```

---

#### 🚀 Quick Start Guide

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Database & Train Models**:
   ```bash
   python src/db_setup.py
   cd src && python model.py && python forecasting.py && python evaluate_anomalies.py && cd ..
   ```

3. **Run Automated Test Suite (32/32 Passing)**:
   ```bash
   python -m unittest tests/test_pipeline.py -v
   ```

4. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```

---

#### 📊 For Your Report & Defense
* **Project ID**: BT-55
* **Institution**: IILM University, Greater Noida
* **Authors**: Adhikansh Gami, Kashish, Sneha
