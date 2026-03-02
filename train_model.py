import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np
import joblib
import os

def clean_data(df):
    """Clean dataset before training."""
    # Ensure fraud_bool is numeric if it exists
    if 'fraud_bool' in df.columns:
        df['fraud_bool'] = pd.to_numeric(df['fraud_bool'], errors='coerce').fillna(0).astype(int)
    else:
        df['fraud_bool'] = 0
    
    # Select our required features
    features = [
        'transaction_amount', 'income', 'device_fraud_count', 
        'account_age_days', 'transaction_hour', 'is_foreign_ip', 'month'
    ]
    
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
            
    df = df[features + ['fraud_bool']]
    df = df.dropna(subset=['fraud_bool'])
    
    return df

def train_and_save_model(csv_path="commercial_fraud_dataset.csv", model_path="model.pkl"):
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    # Read only needed columns to save memory if possible
    # Base.csv headers might have fraud_bool, income, device_fraud_count, month
    try:
            df = pd.read_csv(csv_path, usecols=[
                'fraud_bool', 'transaction_amount', 'income', 
                'device_fraud_count', 'account_age_days', 
                'transaction_hour', 'is_foreign_ip', 'month'
            ])
    except ValueError:
        # Fallback to loading all if usecols fails
        df = pd.read_csv(csv_path)
    
    print("Cleaning data...")
    df = clean_data(df)
    
    if len(df) == 0:
        print("Dataset is empty after cleaning.")
        return
        
    features = [
        'transaction_amount', 'income', 'device_fraud_count', 
        'account_age_days', 'transaction_hour', 'is_foreign_ip', 'month'
    ]
    X = df[features]
    y = df['fraud_bool']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Balancing with SMOTE...")
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"SMOTE omitted, using standard RF class_weight: {e}")

    print("Training RandomForest model...")
    # Increase min_samples_leaf to smooth out probabilities across leaf nodes
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_leaf=10, class_weight='balanced_subsample')
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model trained successfully! Accuracy: {accuracy:.4f}")
    
    print(f"Saving model to {model_path}...")
    # Save a tuple with model and imputer to ensure consistent input shaping
    joblib.dump({"model": model, "imputer": imputer}, model_path)
    print("Done.")

if __name__ == "__main__":
    train_and_save_model()