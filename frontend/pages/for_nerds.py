import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kiro - Technical Documentation", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models", "for_nerds")
DATA_PATH = os.path.join(BASE_DIR, "finaldata_with_climate.csv")

# ----------------------------------------------------------
# Utility: Dynamic Grid Layout for Graphs
# ----------------------------------------------------------
def display_graphs_in_grid(figures, cols=3):
    for i in range(0, len(figures), cols):
        row_figs = figures[i:i+cols]
        columns = st.columns(len(row_figs))
        for col, fig in zip(columns, row_figs):
            with col:
                st.pyplot(fig)

# ----------------------------------------------------------

st.title("🧠 Kiro — Technical & Modeling Documentation")
st.markdown("---")

# =========================================================
# 1️⃣ DATASET OVERVIEW
# =========================================================

st.header("1️⃣ Dataset Structure")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    col1, col2, col3 = st.columns(3)
    col1.write(f"Shape: {df.shape}")
    col2.write(f"Year Range: {df['year'].min()} - {df['year'].max()}")
    col3.write(f"District Count: {df['district'].nunique()}")


    st.subheader("Columns")
    st.write(list(df.columns))

    st.subheader("Sample Data")
    st.dataframe(df.head())

else:
    st.error("Dataset not found.")

st.markdown("""
### Feature Groups

Temporal → year, prev_year_yield  
Climate → max_temp, rainfall, stress indices  
Vegetation → NDVI, EVI metrics  
Agricultural → area, production, yield  
""")

st.markdown("---")

# =========================================================
# 2️⃣ PREPROCESSING
# =========================================================

st.header("2️⃣ Preprocessing Pipeline")

st.markdown("""
- OneHotEncoding → district  
- StandardScaler → numeric features  
- Target transform → log1p(yield)  
- Train/Test split → 80/20  

Why log transform?
- Reduces skewness
- Improves linear model stability
- Reduces heteroscedasticity
""")

st.markdown("---")

# =========================================================
# 3️⃣ MODEL PERFORMANCE
# =========================================================

st.header("3️⃣ Model Benchmarking")

metrics_path = os.path.join(MODEL_DIR, "model_metrics.csv")

if os.path.exists(metrics_path):

    metrics_df = pd.read_csv(metrics_path)
    metrics_df = metrics_df.sort_values(by="R2", ascending=False)

    st.dataframe(metrics_df, use_container_width=True)

    figures = []

    # R2 Plot
    fig1, ax1 = plt.subplots()
    ax1.bar(metrics_df["Model"], metrics_df["R2"])
    ax1.set_title("R² Score Comparison")
    ax1.set_ylabel("R²")
    ax1.tick_params(axis='x', rotation=45)
    figures.append(fig1)

    # RMSE Plot
    fig2, ax2 = plt.subplots()
    ax2.bar(metrics_df["Model"], metrics_df["RMSE"])
    ax2.set_title("RMSE Comparison")
    ax2.set_ylabel("RMSE")
    ax2.tick_params(axis='x', rotation=45)
    figures.append(fig2)

    # MSE Plot
    fig3, ax3 = plt.subplots()
    ax3.bar(metrics_df["Model"], metrics_df["MSE"])
    ax3.set_title("MSE Comparison")
    ax3.set_ylabel("MSE")
    ax3.tick_params(axis='x', rotation=45)
    figures.append(fig3)

    display_graphs_in_grid(figures, cols=3)

else:
    st.error("Model metrics not found.")

st.markdown("""
Metric Meaning:

• MSE → Squared error penalty  
• RMSE → Error magnitude in log-yield  
• R² → Variance explained  

Higher R² = better explanatory power  
Lower RMSE = better predictive accuracy  
""")

st.markdown("---")

# =========================================================
# 4️⃣ MODEL DIFFERENCES
# =========================================================

st.header("4️⃣ Algorithmic Differences")

st.markdown("""
Ridge Regression:
- Linear
- L2 regularization
- Assumes additive relationships

Random Forest:
- Bagging ensemble
- Non-linear splits
- Handles interactions automatically

XGBoost:
- Gradient boosting
- Sequential tree correction
- Strong bias-variance control

Voting Regressor:
- Averaging ensemble
- Reduces prediction variance
""")

st.markdown("---")

# =========================================================
# 5️⃣ RESIDUAL ANALYSIS
# =========================================================

st.header("5️⃣ Residual Diagnostics (XGBoost)")

xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")

if os.path.exists(xgb_path) and os.path.exists(DATA_PATH):

    model = joblib.load(xgb_path)
    df = pd.read_csv(DATA_PATH)

    features = [col for col in df.columns if col != "yield"]
    X = df[features]
    y = np.log1p(df["yield"])

    preds = model.predict(X)
    residuals = y - preds

    figures = []

    # Residual Histogram
    fig4, ax4 = plt.subplots()
    ax4.hist(residuals, bins=30)
    ax4.set_title("Residual Distribution")
    figures.append(fig4)

    # Residual vs Prediction
    fig5, ax5 = plt.subplots()
    ax5.scatter(preds, residuals, alpha=0.4)
    ax5.set_title("Residual vs Prediction")
    ax5.set_xlabel("Predicted")
    ax5.set_ylabel("Residual")
    figures.append(fig5)

    # Actual vs Predicted
    fig6, ax6 = plt.subplots()
    ax6.scatter(y, preds, alpha=0.4)
    ax6.set_title("Actual vs Predicted")
    ax6.set_xlabel("Actual")
    ax6.set_ylabel("Predicted")
    figures.append(fig6)

    display_graphs_in_grid(figures, cols=3)

    st.write("Mean Residual:", round(np.mean(residuals), 6))

else:
    st.info("Residual analysis unavailable.")

st.markdown("---")

# =========================================================
# 6️⃣ PROJECT STRUCTURE
# =========================================================

st.header("6️⃣ Project Architecture")

st.code("""
agri-vision-ai/
│
├── train_models.py
├── finaldata_with_climate.csv
│
├── models/
│    └── for_nerds/
│
└── frontend/
     ├── app.py
     └── pages/
          2_For_Nerds.py
""")

st.markdown("""
Pipeline:

1. Data ingestion  
2. Feature engineering  
3. Encoding & scaling  
4. Model training  
5. Ensemble construction  
6. Evaluation & benchmarking  
7. Deployment  
""")

st.markdown("---")

# =========================================================
# 7️⃣ SAVED ARTIFACTS
# =========================================================

st.header("7️⃣ Saved Artifacts")

if os.path.exists(MODEL_DIR):
    for file in os.listdir(MODEL_DIR):
        st.write("✔", file)
else:
    st.warning("Model directory not found.")

st.markdown("End of technical documentation.")
