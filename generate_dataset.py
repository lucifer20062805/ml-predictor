import pandas as pd
import numpy as np
import os

def generate_commercial_dataset(output_path="commercial_fraud_dataset.csv", n_samples=79000):
    print(f"Generating commercial-grade dataset with {n_samples} entries...")
    np.random.seed(42)
    
    # 95% Safe, 5% Fraud
    n_fraud = int(n_samples * 0.05)
    n_safe = n_samples - n_fraud
    
    # Generate Safe Transactions
    safe_data = {
        'transaction_amount': np.random.lognormal(mean=np.log(200), sigma=1.0, size=n_safe).round(2),
        'income': np.random.normal(70000, 25000, n_safe).clip(10000, 250000),
        'device_fraud_count': np.random.poisson(0.5, n_safe),
        'account_age_days': np.random.randint(90, 3650, n_safe),
        'transaction_hour': np.random.randint(6, 23, n_safe),  # Mostly daytime
        'is_foreign_ip': np.random.choice([0, 1], p=[0.98, 0.02], size=n_safe),
        'month': np.random.randint(1, 13, n_safe),
        'fraud_bool': 0
    }
    
    # Generate Fraud Transactions
    fraud_data = {
        'transaction_amount': np.random.lognormal(mean=np.log(2500), sigma=1.2, size=n_fraud).round(2),
        'income': np.random.normal(50000, 30000, n_fraud).clip(10000, 250000),
        'device_fraud_count': np.random.poisson(4.0, n_fraud),
        'account_age_days': np.random.randint(0, 60, n_fraud), # Often new accounts
        'transaction_hour': np.random.choice(
            list(range(0, 6)) + list(range(23, 24)), 
            size=n_fraud
        ), # Often nighttime
        'is_foreign_ip': np.random.choice([0, 1], p=[0.20, 0.80], size=n_fraud),
        'month': np.random.randint(1, 13, n_fraud),
        'fraud_bool': 1
    }
    
    df_safe = pd.DataFrame(safe_data)
    df_fraud = pd.DataFrame(fraud_data)
    
    # Combine and shuffle
    df = pd.concat([df_safe, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # To ensure complex, tree-based ML logic, we use a continuous risk score
    # rather than hard boolean logic, plus interactions.
    
    # Base risk score
    risk = np.zeros(len(df))
    
    # 1. High amounts add risk, scaled down
    risk += (df['transaction_amount'] / 10000.0) 
    
    # 2. Income discrepancy (high tx amount vs low income)
    ratio = df['transaction_amount'] / (df['income'] + 1)
    risk += ratio * 5 
    
    # 3. Known device fraud is a heavy multiplier
    risk += df['device_fraud_count'] * 1.5 
    
    # 4. Brand new accounts are risky (inverse age)
    risk += np.where(df['account_age_days'] < 30, 1.5, 0)
    risk += np.where(df['account_age_days'] < 5, 2.0, 0) 
    
    # 5. Suspicious hours (12AM - 5AM) add risk
    suspicious_hours = ((df['transaction_hour'] >= 0) & (df['transaction_hour'] <= 5))
    risk += np.where(suspicious_hours, 1.0, 0)
    
    # 6. Foreign IPs
    risk += (df['is_foreign_ip'] * 2.0)
    
    # Define a threshold for fraud (top ~5-15% of risk scores)
    threshold = np.percentile(risk, 88) 
    
    df['fraud_bool'] = np.where(risk > threshold, 1, 0)
    
    # Add random background noise (1% chance of random flip)
    noise_mask = np.random.rand(len(df)) < 0.01 
    df.loc[noise_mask, 'fraud_bool'] = 1 - df.loc[noise_mask, 'fraud_bool']
    
    print("Class distribution:")
    print(df['fraud_bool'].value_counts(normalize=True))
    
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path}")

if __name__ == "__main__":
    generate_commercial_dataset()
