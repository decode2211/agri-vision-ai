import streamlit as st
from backend.api import get_past_report, predict_future, df
import matplotlib.pyplot as plt

st.set_page_config(page_title="Agri Vision AI", layout="centered")

st.title("🌾 District Rice Yield Intelligence System")

mode = st.radio("Select Mode:", ["Study Past", "Predict Future"])

district = st.text_input("Enter District Name:")

# ==========================================
# STUDY PAST MODE
# ==========================================

if mode == "Study Past":

    year = st.number_input("Enter Year", min_value=2000, max_value=2023, step=1)

    if st.button("Get Report"):

        result = get_past_report(district, year)

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("📊 District Summary")

            col1, col2, col3 = st.columns(3)
            col1.metric("Yield (t/ha)", result["actual_yield"])
            col2.metric("Rainfall (mm)", result["rainfall"])
            col3.metric("Max Temp (°C)", result["max_temperature"])

            district_data = df[df["district"] == district.lower()].copy()

            # 1️⃣ Yield Trend
            st.subheader("📈 Yield Trend Over Time")

            fig, ax = plt.subplots()
            ax.plot(district_data["year"], district_data["yield"])
            ax.axvline(year, linestyle="--")
            ax.set_xlabel("Year")
            ax.set_ylabel("Yield (t/ha)")
            ax.set_title("Yield Trend")
            st.pyplot(fig)

            # 2️⃣ Rainfall vs Yield
            st.subheader("🌧 Rainfall vs Yield Relationship")

            fig2, ax2 = plt.subplots()
            ax2.scatter(district_data["seasonal_rainfall"], district_data["yield"])
            ax2.set_xlabel("Seasonal Rainfall (mm)")
            ax2.set_ylabel("Yield (t/ha)")
            ax2.set_title("Rainfall Impact on Yield")
            st.pyplot(fig2)

            # 3️⃣ Temperature vs Yield
            st.subheader("🌡 Temperature vs Yield Relationship")

            fig3, ax3 = plt.subplots()
            ax3.scatter(district_data["max_temp"], district_data["yield"])
            ax3.set_xlabel("Maximum Temperature (°C)")
            ax3.set_ylabel("Yield (t/ha)")
            ax3.set_title("Temperature Impact on Yield")
            st.pyplot(fig3)

# ==========================================
# PREDICT FUTURE MODE
# ==========================================

elif mode == "Predict Future":

    year = st.number_input("Prediction Year", min_value=2024, max_value=2035, step=1)
    rainfall = st.number_input("Expected Rainfall (mm)", min_value=0.0)
    max_temp = st.number_input("Expected Max Temperature (°C)", min_value=0.0)

    if st.button("Predict Yield"):

        result = predict_future(district, year, rainfall, max_temp)

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("📊 Prediction Breakdown")

            col1, col2, col3 = st.columns(3)
            col1.metric("Structural Yield", result["structural_yield"])
            col2.metric("Climate Adjustment", result["climate_adjustment"])
            col3.metric("Final Yield (t/ha)", result["final_predicted_yield"])
            adj = result["climate_adjustment"]

            if adj < -0.25:
             st.error("🚨 Severe climate stress")
            elif adj < -0.1:
             st.warning("⚠ Moderate climate stress")
            elif adj < -0.03:
             st.info("ℹ Mild climate variation")
            elif adj <= 0.03:
             st.success("✅ Near-normal climate conditions")
            else:
             st.success("🌧 Favorable climate boost")
