import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# PATH SETUP
# ==========================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "finaldata.csv")

# NEW MODEL PATH
MODEL_PATH = os.path.join(BASE_DIR, "models", "trial1.pkl")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)
df["district"] = df["district"].str.lower().str.strip()

df = df.sort_values(["district", "year"])

df["prev_year_yield"] = df.groupby("district")["yield"].shift(1)
df["prev_year_rain"] = df.groupby("district")["seasonal_rainfall"].shift(1)

df["heat_stress_index"] = df["max_temp"] / (df["seasonal_rainfall"] + 1.0)

df["rain_anomaly"] = (
    df["seasonal_rainfall"] -
    df.groupby("district")["seasonal_rainfall"].transform("mean")
)

df = df.dropna()

district_means = df.groupby("district")["yield"].mean()
df["district_baseline"] = df["district"].map(district_means)

# ==========================================
# LOAD HYBRID MODEL
# ==========================================

model_package = joblib.load(MODEL_PATH)

struct_model = model_package["struct_model"]
climate_model = model_package["climate_model"]

structural_features = model_package["structural_features"]
climate_features = model_package["climate_features"]

heat_threshold = model_package["heat_threshold"]
flood_threshold = model_package["flood_threshold"]
drought_threshold = model_package["drought_threshold"]

global_mean = model_package["global_mean"]

# ==========================================
# PAST MODE (UNCHANGED)
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
# FUTURE MODE (UPDATED TO HYBRID MODEL)
# ==========================================

def predict_future(district_name, year_input, rainfall, max_temp):

    district_name = district_name.lower().strip()

    district_data = df[df["district"] == district_name]

    if district_data.empty:
        return {"error": "District not found"}

    latest = district_data.iloc[-1]

    # ------------------------------
    # Build Input Row
    # ------------------------------

    input_row = pd.DataFrame([{
        "district_baseline": latest["district_baseline"],
        "prev_year_yield": latest["yield"],
        "year": year_input,
        "peak_evi": latest["peak_evi"],
        "max_water_stress": latest["max_water_stress"],
        "max_atmospheric_thirst": latest["max_atmospheric_thirst"],
        "heat_stress_index": max_temp / (rainfall + 1.0),
        "rain_anomaly": rainfall - district_data["seasonal_rainfall"].mean(),
        "seasonal_rainfall": rainfall,
        "max_temp": max_temp,
        "rainfall_variability": district_data["seasonal_rainfall"].std()
    }])

    # Damage Features
    input_row["heat_damage"] = np.maximum(max_temp - heat_threshold, 0)
    input_row["flood_damage"] = np.maximum(rainfall - flood_threshold, 0)
    input_row["drought_damage"] = np.maximum(drought_threshold - rainfall, 0)

    # ------------------------------
    # Predict
    # ------------------------------

    struct_pred = struct_model.predict(input_row[structural_features])[0]
    climate_adj = climate_model.predict(input_row[climate_features])[0]

    final_pred = struct_pred + climate_adj

    return {
        "district": district_name.title(),
        "year": year_input,
        "structural_yield": float(round(struct_pred, 3)),
        "climate_adjustment": float(round(climate_adj, 3)),
        "final_predicted_yield": float(round(final_pred, 3))
    }
