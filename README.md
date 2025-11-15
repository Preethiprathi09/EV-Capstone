⚡ EV Adoption & Charging Optimization — Capstone Project

Prathiksha A.H.
Kusuma C
Priya P
Department of Computer Science And Engineering
ACS College Of Engineering

📘 Project Overview

This project analyzes EV charging infrastructure in Indian cities, predicts EV adoption levels using Machine Learning (Random Forest), and optimizes charging schedules using Linear Programming.
An interactive Streamlit dashboard visualizes charger locations, model predictions, and optimized charging patterns.

🗂️ Folder Structure
ev-capstone/
│ app.py
│ requirements.txt
│ requirements-streamlit.txt
│
├─ data/raw/
│   ├ detailed_ev_charging_stations.csv
│   ├ ev-charging-stations-india.csv
│   └ city_ev_registrations.csv
│
├─ notebooks/
│   └ ev_capstone_notebook.ipynb
│
├─ outputs/

⚙️ How to Run the Dashboard (Guide Instructions)
1. Install Dependencies
pip install -r requirements-streamlit.txt

2. Run the App
streamlit run app.py

3. Access the Dashboard

Open
http://localhost:8501/

in your browser.

📊 Features

📍 Map visualization of EV charging stations
🔍 City-wise charger statistics
🚗 EV adoption prediction using Random Forest
🔋 Smart charging schedule optimization (Linear Programming)
💼 Interactive dashboard for planning & analysis

🌱 Future Enhancements

Real-time EV charging data integration
Tariff-based cost optimization
Deployment on Streamlit Cloud / Hugging Face
National-level EV adoption modelling
CO₂ savings analytics
