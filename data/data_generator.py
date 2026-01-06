import pandas as pd
import numpy as np
import os

def generate_reference_data(output_path, n_samples=5000):
    """
    Generates synthetic 'historical' loan data.
    """
    np.random.seed(42)
    
    data = {
        'loan_amount': np.random.randint(5000, 50000, n_samples),
        'income': np.random.normal(60000, 15000, n_samples).clip(20000, 150000),
        'credit_score': np.random.normal(700, 50, n_samples).clip(300, 850),
        'interest_rate': np.random.normal(5.0, 1.5, n_samples).clip(2.0, 15.0),
        'employment_years': np.random.randint(0, 30, n_samples),
        'debt_to_income': np.random.uniform(0.1, 0.6, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate probability of default based on features
    # Probability increases with: higher interest rate, higher debt_to_income, lower credit_score
    logit = (
        0.1 * (df['interest_rate'] - 5.0) +
        -0.01 * (df['credit_score'] - 700) +
        1.5 * (df['debt_to_income'] - 0.3) +
        -0.00001 * (df['income'] - 60000)
    )
    prob_default = 1 / (1 + np.exp(-logit))
    df['default'] = (np.random.rand(n_samples) < prob_default).astype(int)
    
    df.to_csv(output_path, index=False)
    print(f"Reference data generated at {output_path}")

if __name__ == "__main__":
    os.makedirs("loan_drift_detection/data", exist_ok=True)
    generate_reference_data("loan_drift_detection/data/reference.csv")
