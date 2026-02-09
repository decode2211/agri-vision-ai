import streamlit as st
from backend.api import get_past_report, predict_future

st.set_page_config(page_title="Agri Vision AI", layout="centered")

st.title("🌾 District Rice Yield Intelligence System")

mode = st.radio("Select Mode:", ["Study Past", "Predict Future"])

district = st.text_input("Enter District Name:")
import pandas as pd
import matplotlib.pyplot as plt

from backend.api import get_past_report
from backend.api import df  # use backend dataframe directly


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

            # ==============================
            # 1️⃣ Yield Trend
            # ==============================
            st.subheader("📈 Yield Trend Over Time")

            fig, ax = plt.subplots()
            ax.plot(district_data["year"], district_data["yield"])
            ax.axvline(year, linestyle="--")  # highlight selected year
            ax.set_xlabel("Year")
            ax.set_ylabel("Yield (t/ha)")
            ax.set_title("Yield Trend")
            st.pyplot(fig)

            st.markdown("""
**How to interpret this graph:**
- The line shows how rice yield has changed over the years.
- An upward trend indicates productivity improvement.
- A sudden drop may indicate drought, extreme weather, or stress conditions.
- The vertical dashed line marks the selected year.
""")

            # ==============================
            # 2️⃣ Rainfall vs Yield
            # ==============================
            st.subheader("🌧 Rainfall vs Yield Relationship")

            fig2, ax2 = plt.subplots()
            ax2.scatter(district_data["seasonal_rainfall"], district_data["yield"])
            ax2.set_xlabel("Seasonal Rainfall (mm)")
            ax2.set_ylabel("Yield (t/ha)")
            ax2.set_title("Rainfall Impact on Yield")
            st.pyplot(fig2)

            st.markdown("""
**How to interpret this graph:**
- Each dot represents one year.
- If dots slope upward → higher rainfall increases yield.
- If scattered randomly → rainfall may not strongly influence yield.
- Extreme low rainfall points may correspond to lower yields (drought years).
""")

            # ==============================
            # 3️⃣ Temperature vs Yield
            # ==============================
            st.subheader("🌡 Temperature vs Yield Relationship")

            fig3, ax3 = plt.subplots()
            ax3.scatter(district_data["max_temp"], district_data["yield"])
            ax3.set_xlabel("Maximum Temperature (°C)")
            ax3.set_ylabel("Yield (t/ha)")
            ax3.set_title("Temperature Impact on Yield")
            st.pyplot(fig3)

            st.markdown("""
**How to interpret this graph:**
- If yield decreases as temperature increases → heat stress is affecting crops.
- A downward slope suggests negative heat impact.
- If stable → crop may be heat-resilient.
""")
elif mode == "Predict Future":
    year = st.number_input("Prediction Year", min_value=2024, max_value=2035, step=1)
    rainfall = st.number_input("Expected Rainfall (mm)")
    max_temp = st.number_input("Expected Max Temperature (°C)")
    
    if st.button("Predict Yield"):
        result = predict_future(district, year, rainfall, max_temp)
        st.json(result)
