# Real-Time Drift Detection System for Loan Risk

## Overview
This project implements a production-grade monitoring system for credit risk models. It detects **Data Drift**, **Feature Drift (PSI)**, and **Prediction Drift** in real-time by comparing incoming live loan applications against a historical baseline.

## Why Drift Matters in Financial ML
In credit scoring, models are trained on historical economic conditions. If interest rates rise or average income drops (as simulated here), the model's accuracy can degrade rapidly. This system ensures:
1. **Model Reliability**: Detects when the model is scoring outside its 'comfort zone'.
2. **Regulatory Compliance**: Provides transparency into feature shifts.
3. **Proactive Maintenance**: Alerts engineers to retrain BEFORE financial losses occur.

## Architecture
- **Reference Data**: Historical loan data used for training.
- **Model Layer**: Random Forest classifier trained on reference data.
- **Streaming Layer**: Simulates real-time loan applications with injected economic drift.
- **Drift Engine**: 
  - **KS Test**: Statistical check for numerical feature distribution shifts.
  - **PSI (Population Stability Index)**: Measures how much the feature distribution has shifted.
  - **Prediction Drift**: Monitors change in the 'Probability of Default' output.
- **Dashboard**: Real-time Streamlit UI for visualization and alerting.

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data & Train Model**:
   ```bash
   python data/data_generator.py
   python model/model.py
   ```

3. **Start Live Simulation** (Open a separate terminal):
   ```bash
   python streaming/live_data_simulator.py
   ```

4. **Launch Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

## Resume-Ready Summary
"Developed a real-time drift detection system for credit risk modeling using Python and Streamlit. Implemented KS-tests and Population Stability Index (PSI) to monitor feature and prediction shifts. Simulated economic drift scenarios to validate detection logic, providing a proactive alerting mechanism for model retraining in production environments."
