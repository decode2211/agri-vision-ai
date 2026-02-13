import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

# -------------------------
# Create model directory
# -------------------------
MODEL_DIR = r"E:\agri-vision-ai\models\for_nerds"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("finaldata_with_climate.csv")

# Add year as feature
features = [
    "district", "year", "area",
    "mean_ndvi", "peak_ndvi_x", "long_term_ndvi",
    "ndvi_anomaly", "max_temp", "seasonal_rainfall",
    "peak_ndvi_y", "peak_evi",
    "max_water_stress", "max_atmospheric_thirst",
    "total_rain_sat", "prev_year_yield",
    "heat_stress_index", "rain_anomaly"
]

target = "yield"

X = df[features]
y = np.log1p(df[target])  # Log transform

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Preprocessing
# -------------------------
categorical = ["district"]
numeric = [col for col in features if col != "district"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric)
    ]
)

# -------------------------
# Models
# -------------------------

ridge = Ridge()

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    objective="reg:squarederror"
)

# Pipelines
ridge_pipe = Pipeline([("pre", preprocessor), ("model", ridge)])
rf_pipe = Pipeline([("pre", preprocessor), ("model", rf)])
xgb_pipe = Pipeline([("pre", preprocessor), ("model", xgb)])

# -------------------------
# Train
# -------------------------
print("⏳ Training models with optimized parameters...")

ridge_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)
xgb_pipe.fit(X_train, y_train)

# Voting Ensemble
voting = VotingRegressor(
    estimators=[
        ("ridge", ridge_pipe),
        ("rf", rf_pipe),
        ("xgb", xgb_pipe)
    ]
)

voting.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
def evaluate(name, model):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    return [name, mse, rmse, r2]

results = []
results.append(evaluate("Ridge Regression", ridge_pipe))
results.append(evaluate("Random Forest", rf_pipe))
results.append(evaluate("XGBoost", xgb_pipe))
results.append(evaluate("Ensemble (Voting)", voting))

results_df = pd.DataFrame(
    results,
    columns=["Model", "MSE", "RMSE", "R2"]
).sort_values(by="R2", ascending=False)

results_df.to_csv(os.path.join(MODEL_DIR, "model_metrics.csv"), index=False)

# Save models
joblib.dump(ridge_pipe, os.path.join(MODEL_DIR, "ridge_model.pkl"))
joblib.dump(rf_pipe, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(xgb_pipe, os.path.join(MODEL_DIR, "xgb_model.pkl"))
joblib.dump(voting, os.path.join(MODEL_DIR, "voting_model.pkl"))

print("\n📊 Model Comparison Results (Sorted by Accuracy):")
print(results_df)
print("✅ All models trained and saved successfully.")
