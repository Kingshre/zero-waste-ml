# zero-waste-ml
ML classifier that predicts which industrial facilities are at risk of missing zero-waste targets using EPA TRI data, Random Forest, and a Streamlit dashboard.

# Zero Waste ML Predictor

A machine learning project that predicts which U.S. industrial facilities 
are at risk of missing zero-waste targets (diversion rate < 90%) using 
EPA Toxics Release Inventory (TRI) data.

## Features
- Random Forest classifier trained on 88,000+ EPA facility records (2020–2023)
- Feature engineering on diversion rate trends, waste volume, and industry sector
- Interactive Streamlit dashboard with Plotly choropleth map
- SHAP explainability to surface top predictors of compliance failure

## Tech Stack
Python · pandas · scikit-learn · Streamlit · Plotly · SHAP

## Data Source
[EPA Toxics Release Inventory (TRI)](https://www.epa.gov/toxics-release-inventory-tri-program)
