import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import time
from drift.data_drift import detect_data_drift
from drift.feature_drift import detect_feature_drift
from drift.prediction_drift import detect_prediction_drift

# Page config
st.set_page_config(page_title="Loan Drift Dashboard", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stAlert {
        border-radius: 10px;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #3e4451;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Real-Time Loan Risk Drift Detection")
st.markdown("---")

# Load Reference Data and Model
@st.cache_resource
def load_assets():
    ref_df = pd.read_csv("loan_drift_detection/data/reference.csv")
    model = joblib.load("loan_drift_detection/model/loan_model.joblib")
    features = joblib.load("loan_drift_detection/model/feature_names.joblib")
    return ref_df, model, features

try:
    reference_df, model, feature_names = load_assets()
except Exception as e:
    st.error("Assets not found. Please ensure data is generated and model is trained.")
    st.stop()

# Dashboard Layout
col1, col2 = st.columns([1, 1])

# Sidebar control
st.sidebar.header("Monitoring Controls")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2)
batch_window = st.sidebar.slider("Batch Window Size", 20, 200, 50)

placeholder = st.empty()

# ... (Logic continues)
if not os.path.exists("loan_drift_detection/data/live_stream.csv"):
    st.warning("Waiting for live data stream...")
    time.sleep(2)
    st.rerun()

# Load live data
live_df = pd.read_csv("loan_drift_detection/data/live_stream.csv")
if len(live_df) < 5:
    st.info("Collecting initial live samples...")
    time.sleep(2)
    st.rerun()

current_batch = live_df.tail(batch_window)

# Run Drift Detection
ks_results = detect_data_drift(reference_df, current_batch)
psi_results = detect_feature_drift(reference_df, current_batch)
pred_drift = detect_prediction_drift("loan_drift_detection/model/loan_model.joblib", reference_df, current_batch)

# Summary Metrics
m1, m2, m3, m4 = st.columns(4)

# Predict drift status
drift_status = "🔴 DRIFT DETECTED" if pred_drift['drift_detected'] else "🟢 STABLE"

m1.metric("Status", drift_status)
m2.metric("PSI (Max)", f"{max([v['psi_score'] for v in psi_results.values()]):.4f}")
m3.metric("Avg Prob Default", f"{pred_drift['cur_preds_mean']:.2%}")
m4.metric("Samples Processed", len(live_df))

if pred_drift['drift_detected']:
    st.error(f"⚠️ **CRITICAL PREDICTION DRIFT:** The model output distribution has shifted significantly (p={pred_drift['p_value']:.4f}). Retraining might be required.")

st.markdown("### 📊 Feature-Level Drift Analysis")

# Feature Comparison Table
drift_data = []
for feat in feature_names:
    drift_data.append({
        "Feature": feat,
        "PSI Score": f"{psi_results[feat]['psi_score']:.4f}",
        "PSI Status": psi_results[feat]['drift_level'],
        "KS P-Value": f"{ks_results[feat]['p_value']:.4f}",
        "Data Drift": "YES" if ks_results[feat]['drift_detected'] else "NO"
    })

drift_df = pd.DataFrame(drift_data)
st.table(drift_df)

# Visualizations
v_col1, v_col2 = st.columns(2)

# 1. Prediction Distribution Comparison
ref_preds = model.predict_proba(reference_df[feature_names])[:, 1]
cur_preds = model.predict_proba(current_batch[feature_names])[:, 1]

fig_pred = go.Figure()
fig_pred.add_trace(go.Histogram(x=ref_preds, name='Reference', opacity=0.6, histnorm='probability'))
fig_pred.add_trace(go.Histogram(x=cur_preds, name='Live', opacity=0.6, histnorm='probability'))
fig_pred.update_layout(title="Prediction Probability Distribution (Default)", barmode='overlay')
v_col1.plotly_chart(fig_pred, use_container_width=True, key="pred_dist")

# 2. Top-Drifting Feature Comparison
top_feat = max(psi_results, key=lambda x: psi_results[x]['psi_score'])
fig_feat = go.Figure()
fig_feat.add_trace(go.Violin(y=reference_df[top_feat], name='Reference', box_visible=True))
fig_feat.add_trace(go.Violin(y=current_batch[top_feat], name='Live', box_visible=True))
fig_feat.update_layout(title=f"Feature Distribution: {top_feat} (Highest Drift)")
v_col2.plotly_chart(fig_feat, use_container_width=True, key="feat_dist")

# Time Series of Drifting Feature (Simulated)
st.markdown("### 📈 Live Feature Trend (Full Dataset)")
fig_trend = px.line(live_df, x=live_df.index, y=top_feat, title=f"{top_feat} Trend over Time")
st.plotly_chart(fig_trend, use_container_width=True, key="trend_chart")

# Auto-refresh
time.sleep(refresh_rate)
st.rerun()
