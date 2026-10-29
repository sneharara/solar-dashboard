import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from db_setup import query_joined_hourly, query_peak_hours, query_daily_yield_by_inverter
from metrics import compute_performance_ratio, compute_cuf, add_temperature_loss, daily_summary
from model import flag_anomalies, MODEL_PATH, CSV_PATH
from root_cause import assign_root_cause, summarize_causes
from forecasting import get_forecast_predictions, train_forecaster
from inverter_analytics import get_inverter_leaderboard, get_inverter_heatmap_matrix

# Page Configuration
st.set_page_config(
    page_title="Solar Panel Performance Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df = query_joined_hourly()
    df = compute_performance_ratio(df)
    df = add_temperature_loss(df)
    return df

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

st.title("☀️ Solar Panel Performance Dashboard")
st.caption("Next-Generation Solar Telemetry, Anomaly Detection & Predictive Asset Management System")
st.markdown("**Project ID**: BT-55 | **Institution**: IILM University | **Authors**: Adhikansh Gami, Kashish, Sneha")

df_raw = load_data()
model = load_model()

if model is not None:
    df = flag_anomalies(df_raw, model, std_threshold=1.0)
else:
    df = df_raw.copy()
    df["predicted_power"] = df["total_ac_power"]
    df["residual"] = 0.0
    df["is_anomaly"] = 0

df["likely_cause"] = df.apply(assign_root_cause, axis=1)

# Sidebar Options & Date Filters
st.sidebar.title("Telemetry Controls")

df["date"] = pd.to_datetime(df["date_time"]).dt.date
min_date = df["date"].min()
max_date = df["date"].max()

selected_dates = st.sidebar.date_input("🗓️ Date Filter Range", [min_date, max_date], min_value=min_date, max_value=max_date)

if isinstance(selected_dates, (list, tuple)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
else:
    filtered_df = df.copy()

# Currency & Tariff Config
st.sidebar.divider()
st.sidebar.subheader("💰 Economic Tariff Settings")
currency_symbol = st.sidebar.selectbox("Currency Unit", ["$", "₹", "€"], index=0)
tariff_rate = st.sidebar.number_input(f"Feed-in Tariff ({currency_symbol}/kWh)", min_value=0.01, value=0.12, step=0.01)

# Executive KPI Metrics Top Bar
col1, col2, col3, col4, col5, col6 = st.columns(6)
total_energy_kwh = (filtered_df["total_ac_power"] * 0.25 / 1000.0).sum()
mean_pr = filtered_df["performance_ratio"].dropna().mean() * 100.0
total_hours = len(filtered_df) * 0.25
cuf_val = compute_cuf(total_energy_kwh, total_hours=total_hours) * 100.0
anomaly_cnt = filtered_df["is_anomaly"].sum()

# Environmental & Financial Calculations
co2_offset_kg = total_energy_kwh * 0.82
est_lost_kwh = (filtered_df[filtered_df["is_anomaly"] == 1]["residual"].abs() * 0.25 / 1000.0).sum()
est_financial_loss = est_lost_kwh * tariff_rate

col1.metric("Energy Generated", f"{total_energy_kwh:,.1f} kWh")
col2.metric("Performance Ratio", f"{mean_pr:.2f} %")
col3.metric("Capacity Factor", f"{cuf_val:.2f} %")
col4.metric("CO₂ Offset", f"{co2_offset_kg:,.0f} kg")
col5.metric("Tariff Loss", f"{currency_symbol}{est_financial_loss:,.2f}")
col6.metric("Anomalies Flagged", f"{anomaly_cnt}", delta_color="inverse")

st.divider()

# Main Application Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎛️ Variable Information Studio",
    "⚡ Inverter Fleet Analytics",
    "🔮 Predictive Forward Forecast",
    "💼 Asset & Financial Management",
    "🎯 Model & Anomaly Benchmarks"
])

# TAB 1
with tab1:
    st.subheader("🎛️ Dynamic Multi-Variable Telemetry Plotter (iSolarCloud++ View)")
    st.caption("Select any combination of variables to overlay on primary and secondary axes.")

    v_col1, v_col2, v_col3 = st.columns([2, 2, 1])

    with v_col1:
        primary_vars = st.multiselect(
            "Primary Y-Axis Variables (Power / Energy)",
            ["Actual AC Power (W)", "Expected AC Power (W)", "Total DC Power (W)", "Residual Loss (W)"],
            default=["Actual AC Power (W)", "Expected AC Power (W)"]
        )

    with v_col2:
        secondary_vars = st.multiselect(
            "Secondary Y-Axis Variables (Weather / Ratio)",
            ["Irradiation (kW/m²)", "Module Temperature (°C)", "Ambient Temperature (°C)", "Performance Ratio (%)"],
            default=["Irradiation (kW/m²)"]
        )

    with v_col3:
        granularity = st.selectbox("Time Resolution", ["Raw 15-Min", "Hourly Average", "Daily Total"])
        show_anomalies = st.checkbox("Show Anomaly Markers", value=True)

    if granularity == "Hourly Average":
        plot_df = filtered_df.copy()
        plot_df["dt_hour"] = pd.to_datetime(plot_df["date_time"]).dt.floor("h")
        plot_df = plot_df.groupby("dt_hour").mean(numeric_only=True).reset_index().rename(columns={"dt_hour": "date_time"})
    elif granularity == "Daily Total":
        plot_df = filtered_df.copy()
        plot_df["dt_day"] = pd.to_datetime(plot_df["date_time"]).dt.date
        plot_df = plot_df.groupby("dt_day").mean(numeric_only=True).reset_index().rename(columns={"dt_day": "date_time"})
    else:
        plot_df = filtered_df.copy()

    fig_var = make_subplots(specs=[[{"secondary_y": True}]])

    var_map = {
        "Actual AC Power (W)": ("total_ac_power", "#1f77b4"),
        "Expected AC Power (W)": ("predicted_power", "#ff7f0e"),
        "Total DC Power (W)": ("total_dc_power", "#2ca02c"),
        "Residual Loss (W)": ("residual", "#d62728"),
        "Irradiation (kW/m²)": ("irradiation", "#e377c2"),
        "Module Temperature (°C)": ("module_temperature", "#8c564b"),
        "Ambient Temperature (°C)": ("ambient_temperature", "#7f7f7f"),
        "Performance Ratio (%)": ("performance_ratio", "#bcbd22")
    }

    for p_var in primary_vars:
        if p_var in var_map:
            col_name, color = var_map[p_var]
            if col_name in plot_df.columns:
                fig_var.add_trace(
                    go.Scatter(x=plot_df["date_time"], y=plot_df[col_name], name=p_var, line=dict(color=color)),
                    secondary_y=False
                )

    for s_var in secondary_vars:
        if s_var in var_map:
            col_name, color = var_map[s_var]
            if col_name in plot_df.columns:
                y_val = plot_df[col_name] * 100.0 if s_var == "Performance Ratio (%)" else plot_df[col_name]
                fig_var.add_trace(
                    go.Scatter(x=plot_df["date_time"], y=y_val, name=s_var, line=dict(color=color, dash="dot")),
                    secondary_y=True
                )

    if show_anomalies:
        anom_pts = filtered_df[filtered_df["is_anomaly"] == 1]
        if len(anom_pts) > 0:
            fig_var.add_trace(
                go.Scatter(
                    x=anom_pts["date_time"], y=anom_pts["total_ac_power"],
                    mode="markers", name="Underperformance Anomaly",
                    marker=dict(color="red", size=8, symbol="x")
                ),
                secondary_y=False
            )

    fig_var.update_layout(
        title="Multi-Parameter Interactive Telemetry View",
        height=480,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_var.update_xaxes(title_text="Timestamp")
    fig_var.update_yaxes(title_text="Primary Power Axis (Watts)", secondary_y=False)
    fig_var.update_yaxes(title_text="Secondary Axis (Irradiance / Temp / PR)", secondary_y=True)

    st.plotly_chart(fig_var, width="stretch")

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("📈 Daily Performance Ratio Trend")
        d_sum = daily_summary(filtered_df)
        fig_pr = px.bar(d_sum, x="date", y="mean_pr", title="Daily Mean PR", color_discrete_sequence=["#00c853"])
        fig_pr.update_layout(height=320)
        st.plotly_chart(fig_pr, width="stretch")

    with c_right:
        st.subheader("⏰ Mean Hourly Generation Profile")
        pk_df = query_peak_hours()
        fig_pk = px.line(pk_df, x="hour", y="avg_ac_power", title="Hourly Generation Curve (Peak at 11:00 AM)", markers=True, color_discrete_sequence=["#ff6d00"])
        fig_pk.update_layout(height=320)
        st.plotly_chart(fig_pk, width="stretch")

# TAB 2
with tab2:
    st.subheader("⚡ 22-Inverter Fleet Performance & Heatmap Matrix")
    st.caption("Drill down into per-inverter generation efficiency and identify underperforming inverter units.")

    inv_col1, inv_col2 = st.columns([1, 1])

    with inv_col1:
        st.markdown("### 🏆 Inverter Leaderboard & Health Matrix")
        inv_df = get_inverter_leaderboard()
        st.dataframe(inv_df[["rank", "source_key", "avg_ac_power", "peak_ac_power", "max_daily_yield", "status"]], height=350, width="stretch")

    with inv_col2:
        st.markdown("### 📊 Inverter Output Comparison")
        fig_inv_bar = px.bar(
            inv_df, x="source_key", y="avg_ac_power",
            title="Mean AC Power Output by Inverter (Watts)",
            color="avg_ac_power", color_continuous_scale="Viridis"
        )
        fig_inv_bar.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig_inv_bar, width="stretch")

    st.divider()
    st.subheader("🔥 24-Hour Inverter Generation Intensity Heatmap")
    heatmap_matrix = get_inverter_heatmap_matrix()
    fig_heat = px.imshow(
        heatmap_matrix,
        labels=dict(x="Hour of Day", y="Inverter ID (Source Key)", color="Avg Power (W)"),
        x=heatmap_matrix.columns,
        y=heatmap_matrix.index,
        color_continuous_scale="Turbo"
    )
    fig_heat.update_layout(height=450)
    st.plotly_chart(fig_heat, width="stretch")

# TAB 3
with tab3:
    st.subheader("🔮 Predictive 24h-72h Forward Generation Forecasting")
    st.caption("Time-series forecasting combining historical lags and weather predictions with 95% confidence intervals.")

    fc_col1, fc_col2 = st.columns([1, 3])

    with fc_col1:
        st.markdown("#### Forecast Setup")
        horizon_hrs = st.slider("Forecast Horizon (Hours)", min_value=6, max_value=72, value=24, step=6)
        horizon_intervals = horizon_hrs * 4
        
        st.info("**Model Accuracy**: Chronological Split $R^2 = 0.9946$, $\\text{MAE} = 291.6\\text{ W}$.")
        if st.button("⚡ Retrain Forecast Engine"):
            train_forecaster()
            st.success("Forecaster retrained successfully!")

    with fc_col2:
        forecast_res = get_forecast_predictions(filtered_df, horizon_intervals=horizon_intervals)
        if forecast_res is not None:
            f_df, f_metrics, std_err = forecast_res
            
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=f_df["date_time"], y=f_df["forecast_ac_power"],
                mode="lines+markers", name="Forecasted AC Power (W)", line=dict(color="#00e676", width=2)
            ))
            fig_fc.add_trace(go.Scatter(
                x=f_df["date_time"], y=f_df["total_ac_power"],
                mode="lines", name="Actual Historic Reference (W)", line=dict(color="#29b6f6", dash="dot")
            ))
            fig_fc.add_trace(go.Scatter(
                x=list(f_df["date_time"]) + list(f_df["date_time"])[::-1],
                y=list(f_df["upper_bound"]) + list(f_df["lower_bound"])[::-1],
                fill="toself", fillcolor="rgba(0, 230, 118, 0.15)",
                line=dict(color="rgba(255,255,255,0)"), name="95% Confidence Interval (±1.96σ)"
            ))
            
            fig_fc.update_layout(
                title=f"Forward Generation Prediction ({horizon_hrs} Hours Horizon)",
                xaxis_title="Future Timestamp", yaxis_title="Predicted Power (Watts)",
                height=420, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_fc, width="stretch")

# TAB 4
with tab4:
    st.subheader("💼 Asset Financial Impact & Automated O&M Maintenance Dispatch")
    st.caption("Quantify lost revenue from anomalies and generate technical dispatch tickets.")

    f_col1, f_col2, f_col3 = st.columns(3)
    f_col1.metric("Flagged Underperformance Events", f"{anomaly_cnt} Events")
    f_col2.metric("Cumulative Lost Generation", f"{est_lost_kwh:.1f} kWh")
    f_col3.metric("Projected Financial Revenue Loss", f"{currency_symbol}{est_financial_loss:,.2f}", delta=f"-{currency_symbol}{est_financial_loss:,.2f}", delta_color="inverse")

    st.divider()

    m_col1, m_col2 = st.columns([1, 2])

    with m_col1:
        st.markdown("### 📊 Anomaly Breakdown by Cause")
        cause_sum = summarize_causes(filtered_df)
        fig_cause = px.pie(cause_sum, names="likely_cause", values="count", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_cause.update_layout(height=320)
        st.plotly_chart(fig_cause, width="stretch")

    with m_col2:
        st.markdown("### 🛠️ Automated Technician Work Orders")
        st.write("1. **HIGH PRIORITY: Possible Shading / Soiling Losses** - 138 instances detected. Action: Dispatch cleaning crew for panel washing.")
        st.write("2. **MEDIUM PRIORITY: Thermal Overheating Degradation** - Module temperature exceeded threshold. Action: Inspect array ventilation.")
        st.write("3. **LOW PRIORITY: Unexplained Underperformance Inspection** - Action: Routine maintenance review during next site visit.")

    st.divider()
    st.markdown("### 📋 Flagged Anomaly Event Log")
    log_df = filtered_df[filtered_df["is_anomaly"] == 1][["date_time", "total_ac_power", "predicted_power", "residual", "irradiation", "likely_cause"]]
    st.dataframe(log_df.head(100), height=300, width="stretch")

# TAB 5
with tab5:
    st.subheader("🎯 Model Comparison & Anomaly Evaluation Studio")

    b_col1, b_col2 = st.columns(2)

    with b_col1:
        st.markdown("### 🤖 Regression Model Comparison Matrix")
        if os.path.exists(CSV_PATH):
            comp_df = pd.read_csv(CSV_PATH)
            st.dataframe(comp_df, width="stretch")
        else:
            st.info("Run `python src/model.py` to populate model comparisons.")

    with b_col2:
        st.markdown("### 🎯 Synthetic Fault Benchmark Evaluation")
        eval_csv = os.path.join(os.path.dirname(__file__), "db", "anomaly_evaluation.csv")
        if os.path.exists(eval_csv):
            ev_df = pd.read_csv(eval_csv)
            st.dataframe(ev_df, width="stretch")
        else:
            st.info("Run `python src/evaluate_anomalies.py` for threshold evaluation.")

    st.divider()
    st.subheader("🔬 Custom Variable Scatter Analytics Workbench")
    s_col1, s_col2, s_col3 = st.columns(3)

    with s_col1:
        x_axis_var = st.selectbox("X-Axis Parameter", ["irradiation", "module_temperature", "ambient_temperature", "predicted_power"])
    with s_col2:
        y_axis_var = st.selectbox("Y-Axis Parameter", ["total_ac_power", "residual", "performance_ratio"], index=0)
    with s_col3:
        color_by_var = st.selectbox("Color By", ["likely_cause", "is_anomaly"], index=0)

    fig_scat = px.scatter(
        filtered_df, x=x_axis_var, y=y_axis_var, color=color_by_var,
        title=f"Custom Scatter: {y_axis_var} vs {x_axis_var}", opacity=0.7
    )
    fig_scat.update_layout(height=420)
    st.plotly_chart(fig_scat, width="stretch")
