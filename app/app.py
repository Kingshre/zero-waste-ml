import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Zero Waste ML Predictor", layout="wide")

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(APP_DIR, 'facilities_clean.csv'))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(APP_DIR, 'model.pkl'))

df = load_data()
model = load_model()

# Header
st.title("🏭 Zero Waste ML Predictor")
st.markdown("Predicting U.S. industrial facilities at risk of missing zero-waste targets using EPA TRI data.")

# Sidebar filters
st.sidebar.header("Filters")
year = st.sidebar.selectbox("Year", sorted(df['year'].unique(), reverse=True))
state = st.sidebar.multiselect("State", sorted(df['state'].dropna().unique()), default=[])

filtered = df[df['year'] == year]
if state:
    filtered = filtered[filtered['state'].isin(state)]

# Map
st.subheader("📍 Facility Risk Map")
map_df = filtered.dropna(subset=['latitude', 'longitude'])
fig = px.scatter_mapbox(
    map_df,
    lat='latitude',
    lon='longitude',
    color=map_df['at_risk'].astype(str),
    color_discrete_map={'0': 'green', '1': 'red'},
    hover_name='facility_name',
    hover_data={'state': True, 'diversion_rate': ':.2f', 'at_risk': True},
    zoom=3,
    height=500,
    labels={'at_risk': 'At Risk'}
)
fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)

# Stats
col1, col2, col3 = st.columns(3)
col1.metric("Total Facilities", len(filtered))
col2.metric("At-Risk Facilities", filtered['at_risk'].sum())
col3.metric("Compliance Rate", f"{(1 - filtered['at_risk'].mean())*100:.1f}%")

# Table
st.subheader("📋 Facility Risk Table")
st.dataframe(
    filtered[['facility_name', 'state', 'industry_sector', 'diversion_rate', 'at_risk']]
    .sort_values('diversion_rate')
    .reset_index(drop=True),
    use_container_width=True
)

# SHAP
st.subheader("🔍 Feature Importance (SHAP)")
shap_img = Image.open(os.path.join(APP_DIR, 'shap_summary.png'))
st.image(shap_img, use_container_width=True)

# Prediction tool
st.subheader("⚡ Predict Risk for a Facility")
col1, col2 = st.columns(2)
with col1:
    waste = st.number_input("Total Waste (lbs)", min_value=0.0, value=10000.0)
    trend = st.slider("Diversion Rate Trend", -1.0, 1.0, 0.0)
    releases = st.number_input("Total Releases (lbs)", min_value=0.0, value=500.0)
with col2:
    state_input = st.selectbox("State", sorted(df['state'].dropna().unique()))
    sector_input = st.selectbox("Industry Sector", sorted(df['industry_sector'].dropna().unique()))
    year_input = st.selectbox("Year", sorted(df['year'].unique()))

if st.button("Predict Risk"):
    from sklearn.preprocessing import LabelEncoder
    le_state = LabelEncoder().fit(df['state'].fillna('Unknown'))
    le_sector = LabelEncoder().fit(df['industry_sector'].fillna('Unknown'))

    input_data = np.array([[
        np.log1p(waste),
        trend,
        releases,
        le_state.transform([state_input])[0],
        le_sector.transform([sector_input])[0],
        year_input
    ]])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"⚠️ AT RISK — {prob*100:.1f}% probability of missing zero-waste target")
    else:
        st.success(f"✅ ON TRACK — {prob*100:.1f}% risk score")