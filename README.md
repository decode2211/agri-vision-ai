🌍 Drought Impact Analysis – India (2000–2023)

A multi-source drought and agricultural productivity dataset covering 27 drought-prone districts across India.

This project combines satellite-based groundwater indicators with district-level crop productivity data to enable drought impact analysis and forecasting.

📌 Project Overview

This dataset integrates:

GRACE – Groundwater storage anomalies

GLDAS – Root zone soil moisture

ICRISAT – District-level agricultural statistics

NDVI, SPEI, Rainfall – Climate & vegetation indicators

It supports:

Time-series drought analysis

Crop yield correlation studies

Regional comparison

Forecasting models (ARIMA, ML, LSTM)

🗺 Study Regions (27 Districts)
1️⃣ Marathwada (Maharashtra) – 8 districts

Aurangabad, Beed, Hingoli, Jalna, Latur, Nanded, Osmanabad, Parbhani

2️⃣ Bundelkhand (UP & MP) – 13 districts

Banda, Chitrakoot, Hamirpur, Jalaun, Jhansi, Lalitpur, Mahoba, Chhatarpur, Damoh, Datia, Panna, Sagar, Tikamgarh

3️⃣ Eastern Tamil Nadu – 6 districts

Cuddalore, Nagapattinam, Ramanathapuram, Thanjavur, Tiruvarur, Pudukkottai

📂 Dataset Contents
📁 Primary Data

GRACE groundwater anomalies (2003–2017)

GLDAS soil moisture (2000–2023)

District-level agricultural statistics (1966–2014)

NDVI & climate indices

Total: 35,000+ records | 100+ variables | 58-year span

🔎 Key Variables
Satellite Indicators

Groundwater anomaly (cm)

Soil moisture (kg/m²)

NDVI

SPEI

Rainfall

Agricultural Metrics

Area (1000 ha)

Production (1000 tons)

Yield (kg/ha)

🚀 How To Use
Install Requirements
pip install pandas numpy matplotlib seaborn statsmodels

Load Data Example
import pandas as pd

grace = pd.read_csv("drought_regions_grace_2003_2008.csv")
crops = pd.read_csv("ICRISAT-District Level Data (1).csv")

print(grace.head())
print(crops.head())

📊 Recommended Analyses

Time series trend visualization

Drought event detection

Crop yield vs groundwater correlation

Seasonal decomposition

Machine learning forecasting

Regional vulnerability comparison

📚 Data Sources

NASA GRACE (via Google Earth Engine)

NASA GLDAS

ICRISAT Agricultural Data

IMD & CHIRPS rainfall data

🎯 Ideal For

Research projects

Climate & agriculture modeling

Machine learning experiments

Time series forecasting

Academic assignments

📬 Contact

Kevin George
Email: kmgs452003@gmail.com

Kaggle: https://www.kaggle.com/kevinmathewsgeorge

LinkedIn: www.linkedin.com/in/kevin-m-george

🌱 Making drought research simple, structured, and accessible.

🔥 Result

Now your README is:

60% shorter

Clear

Professional

Recruiter-friendly

Hackathon-ready

Not overwhelming
