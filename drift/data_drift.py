import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def detect_data_drift(reference_df, current_df, threshold=0.05):
    """
    Detects data drift using Kolmogorov-Smirnov test for numerical features.
    Returns a dictionary of drift results per feature.
    """
    drift_results = {}
    numerical_cols = reference_df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        if col in current_df.columns:
            # KS Test: Null hypothesis is that both samples come from the same distribution
            stat, p_value = ks_2samp(reference_df[col], current_df[col])
            
            # If p-value is less than threshold, we reject the null hypothesis (drift detected)
            drift_detected = p_value < threshold
            drift_results[col] = {
                'statistic': stat,
                'p_value': p_value,
                'drift_detected': drift_detected
            }
            
    return drift_results

if __name__ == "__main__":
    # Example usage
    ref = pd.read_csv("loan_drift_detection/data/reference.csv")
    cur = ref.copy()
    cur['interest_rate'] = cur['interest_rate'] + 2.0 # Inducing drift
    
    results = detect_data_drift(ref, cur)
    for feat, res in results.items():
        print(f"Feature: {feat}, Drift: {res['drift_detected']}, P-value: {res['p_value']:.4f}")
