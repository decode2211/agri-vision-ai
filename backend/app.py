import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# PATH SETUP
# ==========================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")

# ==========================================
# LOAD DATA
# ==========================================

master_df = pd.read_csv(os.path.join(DATA_PATH, "UP_Rice_Master_Dataset_Final.csv"))
climate_df = pd.read_csv(os.path.join(DATA_PATH, "UP_Kharif_Climate_AllFactors_2000_2014.csv"))
rain_df = pd.read_csv(os.path.join(DATA_PATH, "UP_Kharif_Seasonal_Rainfall_2000_2023.csv"))
sat_df = pd.read_csv(os.path.join(DATA_PATH, "UP_Crop_Features_2000_2014.csv"))

# Clean district names
master_df["district"] = master_df["district"].str.lower().str.strip()
climate_df["district"] = climate_df["ADM2_NAME"].str.lower().str.strip()
rain_df["district"] = rain_df["ADM2_NAME"].str.lower().str.strip()
sat_df["district"] = sat_df["ADM2_NAME"].str.lower().str.strip()

# Rename rainfall column
rain_df = rain_df.rename(columns={"mean": "seasonal_rainfall"})

# Keep only required columns
climate_df = climate_df[["district", "year", "max_temp"]]
rain_df = rain_df[["district", "year", "seasonal_rainfall"]]

# Satellite aggregation
sat_agg = sat_df.groupby(["district", "year"]).agg({
    "NDVI": "max",
    "EVI": "max",
    "LSWI": "min",
    "vap": "max",
    "pr": "sum"
}).reset_index()

sat_agg.columns = ["district", "year", "peak_ndvi", "peak_evi",
                   "max_water_stress", "max_atmospheric_thirst", "total_rain_sat"]

# Merge all datasets
df = master_df.merge(climate_df, on=["district", "year"], how="left")
df = df.merge(rain_df, on=["district", "year"], how="left")
df = df.merge(sat_agg, on=["district", "year"], how="left")

df = df.dropna()


# District baseline
district_means = df.groupby("district")["yield"].mean()
df["district_long_term_avg"] = df["district"].map(district_means)

df["yield_deviation_pct"] = (
    (df["yield"] - df["district_long_term_avg"]) 
    / df["district_long_term_avg"]
) * 100

def classify_risk(deviation):
    if deviation > 10:
        return "Low Risk"
    elif deviation >= -10:
        return "Moderate Risk"
    else:
        return "High Risk"

df["risk_category"] = df["yield_deviation_pct"].apply(classify_risk)


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(MODEL_PATH)

features = [
    "district_long_term_avg",
    "prev_year_yield",
    "peak_evi",
    "max_water_stress",
    "max_atmospheric_thirst",
    "year",
    "heat_stress_index",
    "rain_anomaly",
    "seasonal_rainfall",
    "max_temp"
]

# ==========================================
# PAST MODE
# ==========================================

def get_past_report(district_name, year_input):
    district_name = district_name.lower().strip()

    record = df[
        (df["district"] == district_name) &
        (df["year"] == year_input)
    ]

    if record.empty:
        return {"error": "Data not available"}

    record = record.iloc[0]

    return {
        "district": district_name.title(),
        "year": int(year_input),
        "actual_yield": float(round(record["yield"], 3)),
        "rainfall": float(round(record["seasonal_rainfall"], 2)),
        "max_temperature": float(round(record["max_temp"], 2)),
        "deviation_pct": float(round(record["yield_deviation_pct"], 2)),
        "risk": record["risk_category"]
    }

# ==========================================
# FUTURE MODE
# ==========================================

def predict_future(district_name, year_input, rainfall, max_temp, rain_anomaly):

    district_name = district_name.lower().strip()

    # Use latest known data for satellite + prev_year
    latest_record = df[df["district"] == district_name].iloc[-1]

    input_data = pd.DataFrame([{
        "district_long_term_avg": latest_record["district_long_term_avg"],
        "prev_year_yield": latest_record["yield"],
        "peak_evi": latest_record["peak_evi"],
        "max_water_stress": latest_record["max_water_stress"],
        "max_atmospheric_thirst": latest_record["max_atmospheric_thirst"],
        "year": year_input,
        "heat_stress_index": max_temp / (rainfall + 1.0),
        "rain_anomaly": rain_anomaly,
        "seasonal_rainfall": rainfall,
        "max_temp": max_temp
    }])

    predicted_yield = model.predict(input_data)[0]

    deviation = (
        (predicted_yield - latest_record["district_long_term_avg"]) /
        latest_record["district_long_term_avg"]
    ) * 100

    risk = classify_risk(deviation)

    return {
        "district": district_name.title(),
        "year": year_input,
        "predicted_yield": float(round(predicted_yield, 3)),
        "deviation_pct": float(round(deviation, 2)),
        "risk": risk
    }
