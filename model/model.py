import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def train_model():
    # Load reference data
    df = pd.read_csv("loan_drift_detection/data/reference.csv")
    
    X = df.drop('default', axis=1)
    y = df['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Baseline Model Performance:")
    print(classification_report(y_test, y_pred))
    
    # Save model and feature names
    os.makedirs("loan_drift_detection/model", exist_ok=True)
    joblib.dump(model, "loan_drift_detection/model/loan_model.joblib")
    joblib.dump(X.columns.tolist(), "loan_drift_detection/model/feature_names.joblib")
    
    # Save baseline statistics for drift comparison
    baseline_stats = X.describe()
    baseline_stats.to_csv("loan_drift_detection/model/baseline_stats.csv")
    
    print("Model and baseline statistics saved.")

if __name__ == "__main__":
    train_model()
