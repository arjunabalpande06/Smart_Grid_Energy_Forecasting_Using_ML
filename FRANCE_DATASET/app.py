# ==============================================================================
# Streamlit Web Application: Smart Grid Power Forecasting (Linear Regression)
# ==============================================================================
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Smart Grid Power Forecasting - Linear Regression",
    page_icon="⚡",
    layout="wide"
)

CUSTOM_CSS = """
<style>
.sg-header-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #eff6ff 50%, #f5f3ff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 4px 20px -2px rgba(148, 163, 184, 0.15);
}
.sg-badge {
    display: inline-flex;
    align-items: center;
    background: #ffffff;
    color: #2563eb;
    border: 1px solid #bfdbfe;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 14px;
    border-radius: 9999px;
    margin-bottom: 10px;
}
.sg-title {
    font-size: 26px;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 4px 0;
}
.sg-title-sub {
    font-size: 16px;
    font-weight: 600;
    color: #059669;
    margin: 0 0 8px 0;
}
.sg-model-tag {
    display: inline-block;
    background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%);
    color: #1e40af;
    font-size: 13px;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 8px;
    border: 1px solid #bfdbfe;
    margin-bottom: 8px;
}
.sg-subtitle {
    font-size: 13px;
    color: #64748b;
    margin: 0;
}
.sg-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px -2px rgba(148, 163, 184, 0.1);
}
.sg-card-header {
    font-size: 14px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 8px;
}
.sg-meta-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    padding: 4px 0;
    border-bottom: 1px dashed #f1f5f9;
}
.sg-unit-box {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1px solid #a7f3d0;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
}
.sg-unit-val {
    font-size: 17px;
    font-weight: 800;
    color: #065f46;
}
.sg-unit-desc {
    font-size: 11px;
    color: #047857;
    margin-top: 2px;
}
.sg-footer {
    text-align: center;
    padding: 20px 10px 10px;
    margin-top: 24px;
    border-top: 1px solid #e2e8f0;
    color: #64748b;
    font-size: 12px;
    line-height: 1.5;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Header Banner Card
st.markdown("""
<div class="sg-header-card">
    <div class="sg-badge">⚡ Smart Grid Power Forecasting • Machine Learning Baseline</div>
    <div class="sg-title">⚡ Smart Grid Power Forecasting</div>
    <div class="sg-title-sub">Household Electricity Consumption Forecasting</div>
    <div class="sg-model-tag">⚡ Linear Regression Model</div>
    <p class="sg-subtitle">AI-powered predictive forecasting of household electricity consumption</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = None
    candidates = [
        Path("linear_regression_model.pkl"),
        Path("Linear_Regression_Output/linear_regression_model.pkl"),
        Path("RFR_Outputs/linear_regression_model.pkl"),
        Path("XGBoost_Output/linear_regression_model.pkl"),
        Path("Linear_Regression_Output/linear_regression_energy_model.pkl")
    ]
    for p in candidates:
        if p.exists():
            try:
                model = joblib.load(p)
                break
            except Exception:
                continue
    return model, None

model, scaler = load_artifacts()

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="sg-card-header">🔮 Previous Power Consumption (Lag Features)</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        power_lag_1 = st.number_input("Power 1 Min Ago (kW)", value=1.0, format="%.4f")
    with c2:
        power_lag_5 = st.number_input("Power 5 Min Ago (kW)", value=1.0, format="%.4f")
    with c3:
        power_lag_15 = st.number_input("Power 15 Min Ago (kW)", value=1.0, format="%.4f")
        
    c4, c5 = st.columns(2)
    with c4:
        power_lag_60 = st.number_input("Power 1 Hour Ago (kW)", value=1.0, format="%.4f")
    with c5:
        power_lag_1440 = st.number_input("Power 1 Day Ago (kW)", value=1.0, format="%.4f")

with col2:
    st.markdown('<div class="sg-card-header">📅 Date & Time Parameters</div>', unsafe_allow_html=True)
    cd1, cd2, cd3 = st.columns(3)
    with cd1:
        year = st.number_input("Year", min_value=2000, max_value=2030, value=2007, step=1)
    with cd2:
        month = st.slider("Month", min_value=1, max_value=12, value=1)
    with cd3:
        day = st.slider("Day", min_value=1, max_value=31, value=1)
        
    cd4, cd5, cd6 = st.columns(3)
    with cd4:
        hour = st.slider("Hour", min_value=0, max_value=23, value=12)
    with cd5:
        minute = st.slider("Minute", min_value=0, max_value=59, value=0)
    with cd6:
        day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=0)

predict_btn = st.button("⚡ Predict Power Consumption", type="primary", use_container_width=True)

st.markdown('<div class="sg-card-header" style="margin-top: 14px;">📊 Prediction Result</div>', unsafe_allow_html=True)
result_placeholder = st.empty()

if predict_btn:
    try:
        vals = [power_lag_1, power_lag_5, power_lag_15, power_lag_60, power_lag_1440, year, month, day, hour, minute, day_of_week]
        if any(v is None for v in vals):
            result_placeholder.warning("⚠️ Please enter all required numerical values.")
        elif model is None:
            result_placeholder.error("⚠️ Trained model artifact not found. Please train the model in the notebook first.")
        else:
            is_weekend = 1 if day_of_week >= 5 else 0
            input_data = pd.DataFrame({
                'Power_Lag_1': [power_lag_1],
                'Power_Lag_5': [power_lag_5],
                'Power_Lag_15': [power_lag_15],
                'Power_Lag_60': [power_lag_60],
                'Power_Lag_1440': [power_lag_1440],
                'Year': [year],
                'Month': [month],
                'Day': [day],
                'Hour': [hour],
                'Minute': [minute],
                'DayOfWeek': [day_of_week],
                'IsWeekend': [is_weekend]
            })
            pred = model.predict(input_data)[0]
            result_placeholder.success(f"### Predicted Global Active Power: **{pred:.4f} kW**")
    except Exception as e:
        result_placeholder.error(f"⚠️ Error calculating prediction: {str(e)}")
else:
    result_placeholder.info("Prediction will appear here (e.g. 1.2450 kW)...")

# Info Cards Grid
ic1, ic2, ic3 = st.columns(3)
with ic1:
    st.markdown("""
    <div class="sg-card">
        <div class="sg-card-header">⚡ Electricity Power Unit</div>
        <div class="sg-unit-box">
            <div class="sg-unit-val">kW — Kilowatts</div>
            <div class="sg-unit-desc">Global Active Power consumption per minute</div>
        </div>
        <p style="font-size: 12px; color: #64748b; margin: 4px 0 0 0; line-height: 1.5;">
            Global Active Power represents household power consumption measured per minute.
        </p>
    </div>
    """, unsafe_allow_html=True)
with ic2:
    st.markdown(f"""
    <div class="sg-card">
        <div class="sg-card-header">🤖 Model Information</div>
        <div class="sg-meta-row"><span>Model:</span> <strong>Linear Regression</strong></div>
        <div class="sg-meta-row"><span>Dataset:</span> <strong>France Household Power (UCI)</strong></div>
        <div class="sg-meta-row"><span>Target:</span> <strong>Global Active Power</strong></div>
        <div class="sg-meta-row"><span>Unit:</span> <strong>kW (Kilowatts)</strong></div>
        <div class="sg-meta-row"><span>Forecast Type:</span> <strong>Lag Feature Forecast</strong></div>
    </div>
    """, unsafe_allow_html=True)
with ic3:
    st.markdown("""
    <div class="sg-card">
        <div class="sg-card-header">🇫🇷 Dataset</div>
        <div class="sg-meta-row"><span>Source:</span> <strong>France Household Electric Power</strong></div>
        <div class="sg-meta-row"><span>Target:</span> <strong>Global Active Power</strong></div>
        <div class="sg-meta-row"><span>Frequency:</span> <strong>1-Minute Intervals</strong></div>
        <div class="sg-meta-row"><span>Region:</span> <strong>Sceaux, France</strong></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="sg-footer">
    <div style="font-weight: 700; color: #334155; font-size: 13px; margin-bottom: 4px;">⚡ Smart Grid Power Forecasting</div>
    <div>Machine Learning &amp; Deep Learning Project • Linear Regression | SVR | Random Forest | LSTM | XGBoost</div>
    <div style="margin-top: 4px; color: #94a3b8;">Developed for Academic PBL Project</div>
</div>
""", unsafe_allow_html=True)
