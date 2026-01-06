import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import joblib

def detect_prediction_drift(model_path, reference_df, current_df, threshold=0.05):
    """
    Detects if the distribution of model predictions has shifted significantly.
    """
    # Load model
    model = joblib.load(model_path)
    feature_names = joblib.load("loan_drift_detection/model/feature_names.joblib")
    
    # Ensure current_df has same features (excluding extra columns like 'timestamp')
    ref_features = reference_df[feature_names]
    cur_features = current_df[feature_names]
    
    # Get probability of default (class 1)
    ref_preds = model.predict_proba(ref_features)[:, 1]
    cur_preds = model.predict_proba(cur_features)[:, 1]
    
    # KS Test on prediction distributions
    stat, p_value = ks_2samp(ref_preds, cur_preds)
    drift_detected = p_value < threshold
    
    return {
        'statistic': stat,
        'p_value': p_value,
        'drift_detected': drift_detected,
        'ref_preds_mean': np.mean(ref_preds),
        'cur_preds_mean': np.mean(cur_preds)
    }

if __name__ == "__main__":
    ref = pd.read_csv("loan_drift_detection/data/reference.csv")
    cur = ref.copy()
    # Inducing drift in features to see if prediction drifts
    cur['interest_rate'] = cur['interest_rate'] + 10.0
    
    results = detect_prediction_drift("loan_drift_detection/model/loan_model.joblib", ref, cur)
    print(f"Prediction Drift: {results['drift_detected']}, P-value: {results['p_value']:.4f}")
    print(f"Avg Prob Default (Ref): {results['ref_preds_mean']:.4f}")
    print(f"Avg Prob Default (Current): {results['cur_preds_mean']:.4f}")
