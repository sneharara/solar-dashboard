import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sample_data(output_dir="data", days=34, freq_min=15):
    os.makedirs(output_dir, exist_ok=True)
    start_time = datetime(2020, 5, 15, 0, 0, 0)
    total_steps = days * (24 * 60 // freq_min)
    timestamps = [start_time + timedelta(minutes=i * freq_min) for i in range(total_steps)]
    
    # Weather generation
    weather_records = []
    for ts in timestamps:
        hour = ts.hour + ts.minute / 60.0
        if 6.0 <= hour <= 18.5:
            # Solar bell curve
            solar_factor = np.sin((hour - 6.0) / 12.5 * np.pi)
            irradiation = max(0.0, solar_factor * (1.0 + 0.15 * np.random.randn()))
            ambient_temp = 22.0 + 10.0 * solar_factor + 1.5 * np.random.randn()
            module_temp = ambient_temp + irradiation * 28.0 + 2.0 * np.random.randn()
        else:
            irradiation = 0.0
            ambient_temp = 20.0 + 2.0 * np.random.randn()
            module_temp = ambient_temp
        
        weather_records.append({
            "DATE_TIME": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "PLANT_ID": 4135001,
            "AMBIENT_TEMPERATURE": round(float(ambient_temp), 2),
            "MODULE_TEMPERATURE": round(float(module_temp), 2),
            "IRRADIATION": round(float(irradiation), 4)
        })
    
    weather_df = pd.DataFrame(weather_records)
    weather_df.to_csv(os.path.join(output_dir, "Plant_1_Weather_Sensor_Data.csv"), index=False)
    
    # Generation data for 4 inverters
    inverters = ["1BYS1P3SAB2M650", "1IF23P3SAB2M651", "33YS1P3SAB2M652", "44YS1P3SAB2M653"]
    faulty_inverters = ["1IF23P3SAB2M651", "44YS1P3SAB2M653"]
    fault_days = [10, 22] # days with derated performance
    
    gen_records = []
    for inv_idx, inv in enumerate(inverters):
        daily_yield = 0.0
        total_yield = 100000.0 + inv_idx * 25000.0
        
        for step_idx, ts in enumerate(timestamps):
            day_num = step_idx // (24 * 60 // freq_min)
            hour = ts.hour + ts.minute / 60.0
            
            # Reset daily yield at midnight
            if ts.hour == 0 and ts.minute == 0:
                daily_yield = 0.0
            
            w_row = weather_df.iloc[step_idx]
            irr = w_row["IRRADIATION"]
            
            # Base power (30kW rated capacity per inverter)
            dc_power = irr * 30000.0 * (1.0 + 0.03 * np.random.randn())
            
            # Inject fault
            is_faulty = False
            if inv in faulty_inverters and day_num in fault_days and irr > 0.05:
                dc_power *= np.random.uniform(0.40, 0.60)
                is_faulty = True
            
            ac_power = dc_power * 0.97 if dc_power > 0 else 0.0
            
            # Yield accumulation
            energy_step = ac_power * (freq_min / 60.0) / 1000.0 # kWh
            daily_yield += energy_step
            total_yield += energy_step
            
            gen_records.append({
                "DATE_TIME": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "PLANT_ID": 4135001,
                "SOURCE_KEY": inv,
                "DC_POWER": round(float(max(0.0, dc_power)), 2),
                "AC_POWER": round(float(max(0.0, ac_power)), 2),
                "DAILY_YIELD": round(float(daily_yield), 2),
                "TOTAL_YIELD": round(float(total_yield), 2),
                "KNOWN_FAULT": 1 if is_faulty else 0
            })
            
    gen_df = pd.DataFrame(gen_records)
    gen_df.to_csv(os.path.join(output_dir, "Plant_1_Generation_Data.csv"), index=False)
    print(f"Sample data generated: {len(gen_df)} generation rows, {len(weather_df)} weather rows.")

if __name__ == "__main__":
    generate_sample_data()
