import pandas as pd
import numpy as np
import time
import os

def simulate_streaming_data(n_batches=50, batch_size=20, drift_severity=1.0):
    """
    Simulates real-time stream by appending batches to live_stream.csv.
    Adds gradual economic drift.
    """
    output_path = "loan_drift_detection/data/live_stream.csv"
    
    # Initial clear if file exists
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"Starting live data simulation... Drift severity: {drift_severity}")
    
    for i in range(n_batches):
        # Gradual drift: Increase interest rates, decrease income, decrease credit score
        drift_factor = (i / n_batches) * drift_severity
        
        batch = {
            'timestamp': [time.time()] * batch_size,
            'loan_amount': np.random.randint(5000, 60000, batch_size),
            'income': np.random.normal(60000 - (10000 * drift_factor), 15000, batch_size).clip(15000, 150000),
            'credit_score': np.random.normal(700 - (100 * drift_factor), 50, batch_size).clip(300, 850),
            'interest_rate': np.random.normal(5.0 + (5.0 * drift_factor), 1.5, batch_size).clip(2.0, 25.0),
            'employment_years': np.random.randint(0, 30, batch_size),
            'debt_to_income': np.random.uniform(0.1 + (0.2 * drift_factor), 0.6 + (0.3 * drift_factor), batch_size).clip(0.1, 0.95),
        }
        
        df_batch = pd.DataFrame(batch)
        
        # Write to CSV (append mode)
        df_batch.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
        
        print(f"Batch {i+1}/{n_batches} written to {output_path}")
        time.sleep(2) # Wait 2 seconds between batches

if __name__ == "__main__":
    os.makedirs("loan_drift_detection/data", exist_ok=True)
    simulate_streaming_data()
