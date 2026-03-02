import joblib
import numpy as np
import os
import pandas as pd

# Global loaded model cache
_model_cache = None
_imputer_cache = None

def load_model(model_path="model.pkl"):
    """Loads the model and imputer from disk."""
    global _model_cache, _imputer_cache
    if _model_cache is not None:
        return _model_cache, _imputer_cache
        
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Returning dummy model.")
        return None, None
        
    try:
        data = joblib.load(model_path)
        if isinstance(data, dict):
            _model_cache = data.get("model")
            _imputer_cache = data.get("imputer")
        else:
            _model_cache = data
            _imputer_cache = None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
        
    return _model_cache, _imputer_cache

def predict_fraud(data):
    """
    Accepts input data (list/array of 7 features:
    ['transaction_amount', 'income', 'device_fraud_count', 'account_age_days', 'transaction_hour', 'is_foreign_ip', 'month']).
    Returns prediction (0 or 1) and probability of fraud.
    """
    model, imputer = load_model()
    
    if model is None:
        # Dummy fallback
        return 0, 0.05
        
    data = np.array(data).reshape(1, -1)
    if imputer:
        data = imputer.transform(data)
        
    # RandomForest outputs array of predictions
    prediction = int(model.predict(data)[0])
    
    # probability of class 1 (fraud)
    probabilities = model.predict_proba(data)[0]
    probability = float(probabilities[1]) if len(probabilities) > 1 else float(prediction)
    
    return prediction, probability

def batch_predict(df):
    """
    Accepts a pandas DataFrame with features and returns predictions and probabilities.
    """
    model, imputer = load_model()
    if model is None:
        return [0] * len(df), [0.0] * len(df)
        
    features = [
        'transaction_amount', 'income', 'device_fraud_count', 
        'account_age_days', 'transaction_hour', 'is_foreign_ip', 'month'
    ]
    # Safely extract features or use 0
    X = []
    for col in features:
        if col in df.columns:
            X.append(df[col])
        else:
            X.append(pd.Series([0] * len(df)))
            
    X_df = pd.concat(X, axis=1)
    
    if imputer:
        X_array = imputer.transform(X_df)
    else:
        X_array = X_df.values
        
    predictions = model.predict(X_array)
    probs_array = model.predict_proba(X_array)
    
    # Grab probability of class 1
    probabilities = [p[1] if len(p) > 1 else float(predictions[i]) for i, p in enumerate(probs_array)]
    
    return predictions, probabilities