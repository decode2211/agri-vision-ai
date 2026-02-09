import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# PATH SETUP
# ==========================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "finaldata.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")

# ==========================================
# LOAD DATA + MODEL
# ==========================================

df = pd.read_csv(DATA_PATH)
df["district"] = df["district"].str.lower().str.strip()

model = joblib.load(MODEL_PATH)

# ==========================================
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# ==========================================

df = df.sort_values(["district", "year"])

df["prev_year_yield"] = df.groupby("district")["yield"].shift(1)
df["prev_year_rain"] = df.groupby("district")["seasonal_rainfall"].shift(1)

df["heat_stress_index"] = df["max_temp"] / (df["seasonal_rainfall"] + 1.0)

df["rain_anomaly"] = (
    df["seasonal_rainfall"] -
    df.groupby("district")["seasonal_rainfall"].transform("mean")
)

df = df.dropna()

# Spatial baseline (same logic as training)
district_means = df.groupby("district")["yield"].mean()
df["district_baseline"] = df["district"].map(district_means)

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
        "max_temperature": float(round(record["max_temp"], 2))
    }

# ==========================================
# FUTURE MODE
# ==========================================

def predict_future(district_name, year_input, rainfall, max_temp):

    district_name = district_name.lower().strip()

    district_data = df[df["district"] == district_name]

    if district_data.empty:
        return {"error": "District not found"}

    latest = district_data.iloc[-1]

    rain_anomaly = rainfall - district_data["seasonal_rainfall"].mean()

    input_data = pd.DataFrame([{
        "district_baseline": latest["district_baseline"],
        "prev_year_yield": latest["yield"],
        "peak_evi": latest["peak_evi"],
        "max_water_stress": latest["max_water_stress"],
        "max_atmospheric_thirst": latest["max_atmospheric_thirst"],
        "year": year_input,
        "heat_stress_index": max_temp / (rainfall + 1.0),
        "rain_anomaly": rain_anomaly,
        "seasonal_rainfall": rainfall,
        "max_temp": max_temp
    }])

    predicted = model.predict(input_data)[0]

    return {
        "district": district_name.title(),
        "year": year_input,
        "predicted_yield": float(round(predicted, 3))
    }
