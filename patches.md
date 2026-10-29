# Patches & Bug Fixes Log (solar-dashboard)

### D-1: ISO Date Parsing Fix (`src/db_setup.py`)
- **Issue**: `pd.to_datetime(dayfirst=True)` failed on ISO-formatted strings under `pandas` 3.x.
- **Fix**: Updated to `pd.to_datetime(format="mixed", dayfirst=not _is_iso)` to seamlessly parse both ISO and `DD-MM-YYYY` timestamp formats.

### D-6: SQLite Primary Key & Index Retention (`src/db_setup.py`)
- **Issue**: Using `to_sql(if_exists="replace")` dropped declared primary keys and indexes.
- **Fix**: Replaced with `DROP TABLE IF EXISTS` followed by explicit `CREATE TABLE` schema execution and `if_exists="append"`.

### D-8: Streamlit Deprecated Parameter Fix (`app.py`)
- **Issue**: `use_container_width=True` is deprecated in newer Streamlit releases.
- **Fix**: Updated all `st.plotly_chart` and `st.dataframe` calls to `width="stretch"`.

### D-5: Peak Irradiance Comment Fix (`src/metrics.py`)
- **Issue**: Source code comment referenced $1.0\text{ kW/m}^2$ peak irradiance.
- **Fix**: Updated comment to reflect actual dataset peak irradiance of $\approx 1.222\text{ kW/m}^2$.

### D-3: Anomaly Evaluation Asset Export (`src/evaluate_anomalies.py`)
- **Issue**: Missing evaluation export prevented UI benchmark rendering.
- **Fix**: Automatically exports precision/recall/F1 metrics to `db/anomaly_evaluation.csv`.
