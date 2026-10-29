import os
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "solar.db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SCHEMA = """
CREATE TABLE IF NOT EXISTS generation (
    date_time TEXT,
    plant_id INTEGER,
    source_key TEXT,
    dc_power REAL,
    ac_power REAL,
    daily_yield REAL,
    total_yield REAL,
    known_fault INTEGER DEFAULT 0,
    PRIMARY KEY (date_time, plant_id, source_key)
);

CREATE TABLE IF NOT EXISTS weather (
    date_time TEXT,
    plant_id INTEGER,
    ambient_temperature REAL,
    module_temperature REAL,
    irradiation REAL,
    PRIMARY KEY (date_time, plant_id)
);

CREATE INDEX IF NOT EXISTS idx_gen_datetime ON generation(date_time);
CREATE INDEX IF NOT EXISTS idx_weather_datetime ON weather(date_time);
"""

def get_connection(db_path=DB_PATH):
    return sqlite3.connect(db_path)

def build_db(data_dir=DATA_DIR, db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = get_connection(db_path)
    
    conn.executescript("DROP TABLE IF EXISTS generation; DROP TABLE IF EXISTS weather;")
    conn.executescript(SCHEMA)
    
    gen_csv = os.path.join(data_dir, "Plant_1_Generation_Data.csv")
    weather_csv = os.path.join(data_dir, "Plant_1_Weather_Sensor_Data.csv")
    
    if not os.path.exists(gen_csv) or not os.path.exists(weather_csv):
        print("CSV files missing in data/. Running generate_sample_data...")
        from generate_sample_data import generate_sample_data
        generate_sample_data(output_dir=data_dir)
        
    gen_df = pd.read_csv(gen_csv)
    weather_df = pd.read_csv(weather_csv)
    
    gen_df.columns = [c.lower() for c in gen_df.columns]
    weather_df.columns = [c.lower() for c in weather_df.columns]
    
    _is_iso_gen = gen_df["date_time"].astype(str).str.match(r"^\d{4}-").all()
    gen_df["date_time"] = pd.to_datetime(
        gen_df["date_time"], format="mixed" if hasattr(pd, "to_datetime") else None, dayfirst=not _is_iso_gen
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    
    _is_iso_w = weather_df["date_time"].astype(str).str.match(r"^\d{4}-").all()
    weather_df["date_time"] = pd.to_datetime(
        weather_df["date_time"], format="mixed" if hasattr(pd, "to_datetime") else None, dayfirst=not _is_iso_w
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    
    if "known_fault" not in gen_df.columns:
        gen_df["known_fault"] = 0
        
    gen_df.to_sql("generation", conn, if_exists="append", index=False)
    weather_df.to_sql("weather", conn, if_exists="append", index=False)
    
    conn.close()
    print("Database built successfully.")

def query_joined_hourly(db_path=DB_PATH):
    conn = get_connection(db_path)
    query = """
    SELECT 
        g.date_time,
        g.plant_id,
        SUM(g.dc_power) AS total_dc_power,
        SUM(g.ac_power) AS total_ac_power,
        MAX(g.known_fault) AS known_fault,
        w.ambient_temperature,
        w.module_temperature,
        w.irradiation
    FROM generation g
    JOIN weather w ON g.date_time = w.date_time AND g.plant_id = w.plant_id
    GROUP BY g.date_time, g.plant_id
    ORDER BY g.date_time
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def query_daily_yield_by_inverter(db_path=DB_PATH):
    conn = get_connection(db_path)
    query = """
    SELECT 
        source_key,
        DATE(date_time) as date,
        MAX(daily_yield) as max_daily_yield,
        LAG(MAX(daily_yield), 1) OVER (PARTITION BY source_key ORDER BY DATE(date_time)) as prev_daily_yield
    FROM generation
    GROUP BY source_key, DATE(date_time)
    ORDER BY source_key, date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def query_peak_hours(db_path=DB_PATH):
    conn = get_connection(db_path)
    query = """
    SELECT 
        STRFTIME('%H:00', date_time) as hour,
        AVG(ac_power) as avg_ac_power
    FROM generation
    GROUP BY hour
    ORDER BY hour
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def query_inverter_details(db_path=DB_PATH):
    conn = get_connection(db_path)
    query = """
    SELECT 
        source_key,
        COUNT(DISTINCT date_time) as total_intervals,
        AVG(dc_power) as avg_dc_power,
        AVG(ac_power) as avg_ac_power,
        MAX(ac_power) as peak_ac_power,
        MAX(daily_yield) as max_daily_yield,
        MAX(total_yield) - MIN(total_yield) as net_total_yield
    FROM generation
    GROUP BY source_key
    ORDER BY avg_ac_power DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["dc_ac_ratio"] = np.where(df["avg_ac_power"] > 0, df["avg_dc_power"] / df["avg_ac_power"], np.nan)
    return df

def query_inverter_heatmap(db_path=DB_PATH):
    conn = get_connection(db_path)
    query = """
    SELECT 
        source_key,
        STRFTIME('%H:00', date_time) as hour,
        AVG(ac_power) as avg_ac_power
    FROM generation
    GROUP BY source_key, hour
    ORDER BY source_key, hour
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

if __name__ == "__main__":
    build_db()
