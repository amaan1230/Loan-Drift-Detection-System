import numpy as np
import pandas as pd

def calculate_psi(expected, actual, buckets=10):
    """
    Calculates the Population Stability Index (PSI) between two distributions.
    """
    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= (np.max(input) / (max - min))
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    breakpoints = np.percentile(expected, breakpoints)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # To avoid division by zero
    expected_percents = np.clip(expected_percents, a_min=0.0001, a_max=None)
    actual_percents = np.clip(actual_percents, a_min=0.0001, a_max=None)

    psi_values = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    psi_score = np.sum(psi_values)

    return psi_score

def get_drift_level(psi_score):
    if psi_score < 0.1:
        return "Stable"
    elif psi_score < 0.25:
        return "Warning"
    else:
        return "Severe Drift"

def detect_feature_drift(reference_df, current_df):
    results = {}
    numerical_cols = reference_df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        if col in current_df.columns:
            psi = calculate_psi(reference_df[col], current_df[col])
            level = get_drift_level(psi)
            results[col] = {
                'psi_score': psi,
                'drift_level': level
            }
    return results

if __name__ == "__main__":
    ref = pd.read_csv("loan_drift_detection/data/reference.csv")
    cur = ref.copy()
    cur['credit_score'] = cur['credit_score'] - 50 # Inducing drift
    
    results = detect_feature_drift(ref, cur)
    for feat, res in results.items():
        print(f"Feature: {feat}, PSI: {res['psi_score']:.4f}, Level: {res['drift_level']}")
