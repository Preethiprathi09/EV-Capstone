# app.py (beautified)
"""
Beautified Streamlit app for:
EV Adoption & Charge Optimization — Noor's Capstone
Save as app.py (replace existing)
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import pulp
import os

st.set_page_config(page_title="EV Adoption & Charge Optimization — Noor", layout="wide", initial_sidebar_state="expanded")

# -----------------------
# Small CSS for nicer look
# -----------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#0f172a 0%, #081029 100%); color: #e6eef8; }
    .title { color: #fff; }
    .card { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 12px; }
    .metric { font-weight:700; color:#fff; }
    .small { color: #cbd5e1; font-size:0.9rem; }
    .streamlit-expanderHeader { color: #fff; }
    </style>
    """, unsafe_allow_html=True
)

# -------------------------
# Paths
# -------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
MODEL_PATH = OUTPUTS / "model_artifacts" / "ev_adoption_rf.pkl"

# -------------------------
# Utilities
# -------------------------
@st.cache_data
def load_csv_safe(fname):
    p = DATA_RAW / fname
    if not p.exists():
        return None
    return pd.read_csv(p)

def detect_and_rename_coords(df):
    """
    Robustly detect latitude/longitude-like columns (handles typos and duplicate columns),
    rename them to 'latitude'/'longitude' and coerce to numeric safely.
    """
    import pandas as _pd
    import numpy as _np

    if df is None:
        return None
    if not isinstance(df, _pd.DataFrame):
        # Unexpected type — try to convert
        try:
            df = _pd.DataFrame(df)
        except Exception:
            return None

    df = df.copy()

    # find candidate columns (case-insensitive)
    lat_candidates = [c for c in df.columns if any(k in c.lower() for k in ('lat', 'latt', 'latitude', 'latit'))]
    lon_candidates = [c for c in df.columns if any(k in c.lower() for k in ('lon', 'lng', 'long', 'longitude'))]

    # pick the first candidate if any
    lat_col = lat_candidates[0] if lat_candidates else None
    lon_col = lon_candidates[0] if lon_candidates else None

    # If duplicates exist (multiple columns matched), log small debug note by renaming using first column
    if len(lat_candidates) > 1:
        st.write(f"⚠️ Multiple latitude-like columns found: {lat_candidates}. Using '{lat_col}'.")
    if len(lon_candidates) > 1:
        st.write(f"⚠️ Multiple longitude-like columns found: {lon_candidates}. Using '{lon_col}'.")

    # Rename chosen columns to standard names
    if lat_col and lat_col != 'latitude':
        df = df.rename(columns={lat_col: 'latitude'})
    if lon_col and lon_col != 'longitude':
        df = df.rename(columns={lon_col: 'longitude'})

    # If after rename the selection returns a DataFrame (duplicate column names), pick first sub-column
    # e.g., df['latitude'] could be a DataFrame if duplicates existed with same name
    try:
        lat_series = df['latitude']
        if isinstance(lat_series, _pd.DataFrame):
            # pick first column
            df['latitude'] = lat_series.iloc[:, 0]
            st.write("ℹ️ Note: 'latitude' column had duplicates; using the first one.")
    except KeyError:
        pass

    try:
        lon_series = df['longitude']
        if isinstance(lon_series, _pd.DataFrame):
            df['longitude'] = lon_series.iloc[:, 0]
            st.write("ℹ️ Note: 'longitude' column had duplicates; using the first one.")
    except KeyError:
        pass

    # Coerce to numeric safely only if the column exists and is convertible
    if 'latitude' in df.columns:
        try:
            df['latitude'] = _pd.to_numeric(df['latitude'], errors='coerce')
        except TypeError:
            # fallback: try converting via astype if possible
            try:
                df['latitude'] = df['latitude'].astype(float)
            except Exception:
                df['latitude'] = _np.nan
    if 'longitude' in df.columns:
        try:
            df['longitude'] = _pd.to_numeric(df['longitude'], errors='coerce')
        except TypeError:
            try:
                df['longitude'] = df['longitude'].astype(float)
            except Exception:
                df['longitude'] = _np.nan

    return df

@st.cache_data
def build_charger_summary():
    detailed = load_csv_safe("detailed_ev_charging_stations.csv")
    india = load_csv_safe("ev-charging-stations-india.csv")
    detailed = detect_and_rename_coords(detailed) if detailed is not None else pd.DataFrame()
    india = detect_and_rename_coords(india) if india is not None else pd.DataFrame()
    combined = pd.concat([df for df in [india, detailed] if df is not None and not df.empty], ignore_index=True, sort=False)
    # detect city col
    city_col = None
    for c in combined.columns:
        if c.strip().lower() == 'city':
            city_col = c; break
    if not city_col:
        for c in combined.columns:
            if any(k in c.lower() for k in ['city','town','address','place','location']):
                city_col = c; break
    if city_col:
        combined['city_name'] = combined[city_col].astype(str).str.title().str.strip()
    else:
        combined['city_name'] = 'Unknown'
    if 'latitude' in combined.columns and 'longitude' in combined.columns:
        summary = combined.groupby('city_name').agg(
            charger_count=('latitude','count'),
            avg_lat=('latitude','mean'),
            avg_lon=('longitude','mean')
        ).reset_index().sort_values('charger_count', ascending=False)
    else:
        summary = combined.groupby('city_name').size().reset_index(name='charger_count')
        summary['avg_lat'] = np.nan; summary['avg_lon'] = np.nan
    return combined, summary

@st.cache_data
def load_city_ev():
    city_ev = load_csv_safe("city_ev_registrations.csv")
    if city_ev is None:
        return None
    if 'city' in city_ev.columns and 'City' not in city_ev.columns:
        city_ev = city_ev.rename(columns={'city':'City'})
    city_ev['City'] = city_ev['City'].astype(str).str.title().str.strip()
    if 'EV_Per_1000_Vehicles' not in city_ev.columns:
        if {'EV_Registrations','Total_Vehicles'}.issubset(set(city_ev.columns)):
            city_ev['EV_Per_1000_Vehicles'] = (city_ev['EV_Registrations'] / (city_ev['Total_Vehicles'] / 1000)).round(2)
    return city_ev

def train_or_load_model(X, y):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "model_artifacts").mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH); return model, True
        except Exception:
            pass
    if X.shape[0] < 3:
        return None, False
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, MODEL_PATH)
    return rf, False

def run_scheduling_lp(num_vehicles, charger_power, seed=42):
    np.random.seed(seed)
    demands = np.random.randint(8,25,size=num_vehicles).tolist()
    avail = []
    for _ in range(num_vehicles):
        s = np.random.randint(0,18); e = s + np.random.randint(4,8)
        avail.append(range(s, min(e,24)))
    hours = range(24)
    prob = pulp.LpProblem("ev_scheduling", pulp.LpMinimize)
    x = pulp.LpVariable.dicts('x', (range(num_vehicles), hours), 0, 1, cat='Binary')
    P = pulp.LpVariable('peak_load', lowBound=0); prob += P
    for h in hours:
        prob += pulp.lpSum(x[v][h] * charger_power for v in range(num_vehicles)) <= P
    for v in range(num_vehicles):
        prob += pulp.lpSum(x[v][h] * charger_power for h in avail[v]) >= demands[v]
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    schedule = {v: [h for h in hours if pulp.value(x[v][h]) > 0.5] for v in range(num_vehicles)}
    peak = round(pulp.value(P),2)
    return demands, avail, peak, schedule

# -------------------------
# Load data
# -------------------------
combined_india, chargers_by_city = build_charger_summary()
city_ev = load_city_ev()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Controls")
st.sidebar.markdown("Pick a city → see map, prediction & run demo optimization.")

# select city
city_list = []
if city_ev is not None:
    city_list = sorted(city_ev['City'].unique().tolist())
elif not chargers_by_city.empty:
    city_list = list(chargers_by_city['city_name'].head(100).astype(str).tolist())
if not city_list:
    city_list = ['Bengaluru','Delhi','Mumbai']
selected_city = st.sidebar.selectbox("Select City", city_list)

# optimization inputs
st.sidebar.markdown("---")
num_vehicles = st.sidebar.slider("Demo: Number of EVs", 2, 12, 5)
charger_power = st.sidebar.slider("Charger power (kW)", 3, 22, 7)
run_opt = st.sidebar.button("Run Optimization")

# -------------------------
# Header
# -------------------------
st.markdown("<h1 class='title'>⚡ EV Adoption & Charging Optimization — Noor</h1>", unsafe_allow_html=True)
st.markdown("<div class='small'>Interactive capstone dashboard • Mapping • Prediction • Scheduling optimization</div>", unsafe_allow_html=True)
st.write("")

# -------------------------
# Main layout (metrics + map/chart)
# -------------------------
left_col, right_col = st.columns([2,1])

# Left: Map + top metrics
with left_col:
    # Top metric cards (row)
    total_chargers = combined_india['city_name'].count() if combined_india is not None else 0
    city_chargers = int(chargers_by_city.loc[chargers_by_city['city_name'].str.contains(selected_city, case=False), 'charger_count'].sum()) if not chargers_by_city.empty else 0
    # get predicted value if possible
    pred_text = "N/A"
    chargers_per_10k = "N/A"

    # Prepare merged model data for quick prediction
    if city_ev is not None and not city_ev.empty and not chargers_by_city.empty:
        merged = city_ev.merge(chargers_by_city[['city_name','charger_count']].rename(columns={'city_name':'City'}), on='City', how='left')
        merged['charger_count'] = merged['charger_count'].fillna(0).astype(int)
        for c in ['Total_Vehicles','Population','EV_Registrations']:
            if c not in merged.columns:
                merged[c] = 0
        if 'EV_Per_1000_Vehicles' not in merged.columns and {'EV_Registrations','Total_Vehicles'}.issubset(set(merged.columns)):
            merged['EV_Per_1000_Vehicles'] = (merged['EV_Registrations'] / (merged['Total_Vehicles'] / 1000)).round(2)
        merged['chargers_per_10000_vehicles'] = (merged['charger_count'] / (merged['Total_Vehicles'] / 10000)).replace([np.inf,-np.inf],0).fillna(0)
        feature_cols = [c for c in ['charger_count','Total_Vehicles','Population','chargers_per_10000_vehicles'] if c in merged.columns]
        X_all = merged[feature_cols].fillna(0)
        y_all = merged['EV_Per_1000_Vehicles'].fillna(0)
        model, loaded = train_or_load_model(X_all, y_all)
        if model is not None:
            # attempt city prediction
            city_row = merged[merged['City'].str.contains(selected_city, case=False)]
            if not city_row.empty:
                vec = city_row.iloc[0][feature_cols].values.reshape(1, -1)
                pred_val = float(model.predict(vec)[0])
                pred_text = round(pred_val,2)
                chargers_per_10k = round(city_row.iloc[0]['chargers_per_10000_vehicles'],2)
    # metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='card'><div class='metric'>🔋 {total_chargers}</div><div class='small'>Total charger rows</div></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='card'><div class='metric'>📍 {city_chargers}</div><div class='small'>Chargers in {selected_city}</div></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='card'><div class='metric'>🚗 {pred_text}</div><div class='small'>Pred EV per 1000</div></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='card'><div class='metric'>⚙️ {chargers_per_10k}</div><div class='small'>Chargers / 10k vehicles</div></div>", unsafe_allow_html=True)

    st.markdown("### Map — chargers in selected city")
    subset = combined_india[combined_india['city_name'].str.contains(selected_city, case=False, na=False)]
    if subset.empty:
        st.info(f"No charger rows for {selected_city}")
    else:
        center = [subset['latitude'].mean(), subset['longitude'].mean()] if 'latitude' in subset.columns and subset['latitude'].notna().any() else [20.5937,78.9629]
        m = folium.Map(location=center, zoom_start=12, tiles='CartoDB positron')
        mc = MarkerCluster().add_to(m)
        for _, r in subset.iterrows():
            lat = r.get('latitude', None); lon = r.get('longitude', None)
            if pd.notna(lat) and pd.notna(lon):
                popup = str(r.get('name', r.get('address', 'EV Charger')))
                folium.Marker([lat, lon], popup=popup).add_to(mc)
        st_folium(m, width=700, height=420)

    st.markdown("#### Top cities by charger count")
    st.dataframe(chargers_by_city.head(12).reset_index(drop=True))

# Right: charts, model details and optimization
with right_col:
    st.markdown("### EV adoption vs Charger density")
    # scatter: charger_count vs EV per 1000 (if city_ev exists)
    if city_ev is not None and not city_ev.empty:
        merged_chart = merged.copy()
        merged_chart = merged_chart.dropna(subset=['EV_Per_1000_Vehicles'])
        if not merged_chart.empty:
            fig = px.scatter(merged_chart, x='charger_count', y='EV_Per_1000_Vehicles', hover_data=['City','Total_Vehicles'],
                             labels={'charger_count':'Charger count','EV_Per_1000_Vehicles':'EV per 1000'},
                             title="Charger count vs EV penetration (city-level)")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No EV penetration values in city_ev_registrations.csv to chart.")
    else:
        st.info("Provide city_ev_registrations.csv in data/raw/ to enable prediction charts.")

    st.markdown("### City data & prediction")
    if city_ev is not None and selected_city:
        if not merged[merged['City'].str.contains(selected_city, case=False)].empty:
            row = merged[merged['City'].str.contains(selected_city, case=False)].iloc[0]
            st.write(f"**{row['City']}**")
            st.write({
                "Charger count (merged)": int(row.get('charger_count',0)),
                "Total vehicles": int(row.get('Total_Vehicles',0)),
                "Population": int(row.get('Population',0)) if 'Population' in row.index else None,
                "EV registrations": int(row.get('EV_Registrations',0)) if 'EV_Registrations' in row.index else None,
                "Predicted EV per 1000": pred_text
            })
        else:
            st.info(f"{selected_city} not present in city_ev_registrations.csv merged table.")
    else:
        st.info("City EV registration file required for detailed prediction display.")

    st.markdown("### Optimization demo (scheduling)")
    if run_opt:
        with st.spinner("Solving optimization..."):
            demands, avail, peak, schedule = run_scheduling_lp(num_vehicles, charger_power)
            st.success(f"Optimized peak load: {peak} kW")
            st.write("Demands (kWh):", demands)
            st.write("Availability windows:")
            for i,w in enumerate(avail, start=1):
                st.write(f" - V{i}: {list(w)}")
            st.markdown("**Schedule (vehicle -> hours)**")
            st.json(schedule)
    else:
        st.info("Press 'Run Optimization' on the left to solve a demo scheduling problem.")

# Footer
st.markdown("---")
st.caption("Built for Noor's capstone — EV Adoption & Charge Optimization. Tip: export outputs from ../outputs for slides.")
