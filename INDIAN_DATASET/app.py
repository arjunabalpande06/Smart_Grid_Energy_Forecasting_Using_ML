# ==============================================================================
# Streamlit Web Application: Indian Electricity Load Forecasting (Linear Regression)
# ==============================================================================
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Indian Electricity Load Forecasting - Linear Regression",
    page_icon="⚡",
    layout="wide"
)

CUSTOM_CSS = """
<style>
.sg-header-glass {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.6);
}
.sg-badge-glass {
    display: inline-flex;
    background: rgba(56, 189, 248, 0.1);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 14px;
    border-radius: 9999px;
    margin-bottom: 10px;
}
.sg-title-glass {
    font-size: 26px;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 4px 0;
}
.sg-title-sub-glass {
    font-size: 16px;
    font-weight: 600;
    color: #38bdf8;
    margin: 0 0 8px 0;
}
.sg-model-tag-glass {
    display: inline-block;
    background: rgba(14, 165, 233, 0.12);
    color: #7dd3fc;
    font-size: 13px;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 8px;
    border: 1px solid rgba(56, 189, 248, 0.25);
    margin-bottom: 8px;
}
.sg-subtitle-glass {
    font-size: 13px;
    color: #94a3b8;
    margin: 0;
}
.sg-card-glass {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 14px;
}
.sg-card-header-glass {
    font-size: 14px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 8px;
}
.sg-meta-row-glass {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    padding: 4px 0;
    border-bottom: 1px dashed rgba(255, 255, 255, 0.08);
}
.sg-unit-box-glass {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 10px;
    padding: 10px;
    text-align: center;
}
.sg-unit-val-glass {
    font-size: 17px;
    font-weight: 800;
    color: #34d399;
}
.sg-unit-desc-glass {
    font-size: 11px;
    color: #a7f3d0;
    margin-top: 2px;
}
.sg-footer-glass {
    text-align: center;
    padding: 20px 10px 10px;
    margin-top: 24px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    color: #64748b;
    font-size: 12px;
    line-height: 1.5;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Futuristic Glass Header
st.markdown("""
<div class="sg-header-glass">
    <div class="sg-badge-glass">⚡ SMART GRID TELEMETRY • Machine Learning Baseline</div>
    <div class="sg-title-glass">🇮🇳 Indian Smart Grid</div>
    <div class="sg-title-sub-glass">National Electricity Demand &amp; Load Forecasting</div>
    <div class="sg-model-tag-glass">⚡ Linear Regression Model Engine</div>
    <p class="sg-subtitle-glass">AI-powered predictive forecasting of India's national electricity demand</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = None
    candidates = [
        Path("linear_regression_model.pkl"),
        Path("Linear Regression Model Data/linear_regression_model.pkl"),
        Path("RFR Model Data/linear_regression_model.pkl"),
        Path("XGBoost Model Data/linear_regression_model.pkl"),
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
    st.markdown('<div class="sg-card-header-glass">🔮 Prediction Input Parameters</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        date_input = st.text_input("📅 Date (YYYY-MM-DD)", value="2024-04-01", help="Date format: YYYY-MM-DD")
    with c2:
        time_input = st.text_input("🕐 Time (24-Hour Format)", value="18:00", help="Time format: 24-hour (HH:00)")
    predict_btn = st.button("⚡ Predict Electricity Demand", type="primary", use_container_width=True)

with col2:
    st.markdown('<div class="sg-card-header-glass">📊 Telemetry Prediction Result</div>', unsafe_allow_html=True)
    result_placeholder = st.empty()
    
    if predict_btn:
        if not date_input or not str(date_input).strip() or not time_input or not str(time_input).strip():
            result_placeholder.warning("⚠️ Please enter a valid date and time.")
        else:
            try:
                datetime_val = pd.to_datetime(f"{date_input} {time_input}")
                if model is None:
                    result_placeholder.error("⚠️ Trained model artifact not found. Please run the notebook training cells first.")
                else:
                    input_data = pd.DataFrame({
                        'hour': [datetime_val.hour],
                        'day': [datetime_val.day],
                        'day_of_week': [datetime_val.dayofweek],
                        'month': [datetime_val.month],
                        'year': [datetime_val.year],
                        'day_of_year': [datetime_val.dayofyear]
                    })
                    pred = model.predict(input_data)[0]
                    result_placeholder.markdown(f"""
                    <div class="sg-card-glass">
                        <div style="font-size: 13px; color: #94a3b8;">📅 Target: <strong style="color: #f8fafc;">{datetime_val.strftime('%Y-%m-%d %H:%M')}</strong></div>
                        <div style="font-size: 20px; font-weight: 800; color: #34d399; margin-top: 8px;">⚡ {pred:,.2f} MW</div>
                        <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">Predicted National Electricity Demand</div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                result_placeholder.error(f"⚠️ Error: {str(e)}")
    else:
        result_placeholder.info("Prediction will appear here (e.g. 175,240.50 MW)...")

ic1, ic2, ic3 = st.columns(3)
with ic1:
    st.markdown("""
    <div class="sg-card-glass">
        <div class="sg-card-header-glass">⚡ Electricity Power Unit</div>
        <div class="sg-unit-box-glass">
            <div class="sg-unit-val-glass">MW — Megawatts</div>
            <div class="sg-unit-desc-glass">National grid demand aggregate</div>
        </div>
        <p style="font-size: 12px; color: #94a3b8; margin: 6px 0 0 0; line-height: 1.5;">
            National Electricity Demand representing all-India aggregate load measured in Megawatts (MW).
        </p>
    </div>
    """, unsafe_allow_html=True)
with ic2:
    st.markdown(f"""
    <div class="sg-card-glass">
        <div class="sg-card-header-glass">🤖 Model Information</div>
        <div class="sg-meta-row-glass"><span style="color: #94a3b8;">Model:</span> <strong style="color: #38bdf8;">Linear Regression</strong></div>
        <div class="sg-meta-row-glass"><span style="color: #94a3b8;">Dataset:</span> <strong style="color: #38bdf8;">National Grid India</strong></div>
        <div class="sg-meta-row-glass"><span style="color: #94a3b8;">Target:</span> <strong style="color: #38bdf8;">National Demand</strong></div>
        <div class="sg-meta-row-glass"><span style="color: #94a3b8;">Unit:</span> <strong style="color: #38bdf8;">MW (Megawatts)</strong></div>
        <div class="sg-meta-row-glass"><span style="color: #94a3b8;">Resolution:</span> <strong style="color: #38bdf8;">Hourly Telemetry</strong></div>
    </div>
    """, unsafe_allow_html=True)
with ic3:
    st.markdown("""
    <div class="sg-card-glass">
        <div class="sg-card-header-glass">🇮🇳 Dataset</div>
        <div class="sg-meta-row-glass"><span style="color: #94a3b8;">Source:</span> <strong style="color: #38bdf8;">Power System Operation Corp (POSOCO)</strong></div>
        <div class="sg-meta-row-glass"><span style="color: #94a3b8;">Region:</span> <strong style="color: #38bdf8;">National Grid India</strong></div>
        <div class="sg-meta-row-glass"><span style="color: #94a3b8;">Frequency:</span> <strong style="color: #38bdf8;">Hourly Intervals</strong></div>
        <div class="sg-meta-row-glass"><span style="color: #94a3b8;">Type:</span> <strong style="color: #38bdf8;">Time Series Load Telemetry</strong></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="sg-footer-glass">
    <div style="font-weight: 700; color: #cbd5e1; font-size: 13px; margin-bottom: 4px;">⚡ Indian Smart Grid Power Forecasting</div>
    <div>Machine Learning &amp; Deep Learning Project • Linear Regression | SVR | Random Forest | LSTM | XGBoost</div>
    <div style="margin-top: 4px; color: #64748b;">Academic PBL Project • Clean &amp; Renewable Energy Intelligence</div>
</div>
""", unsafe_allow_html=True)
