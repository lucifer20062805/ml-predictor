import predict as _pred
import pandas as pd
import numpy as np

STAGE_DEFINITIONS = {
    0: {"icon": "✅", "label": "Normal", "description": "Standard transactional behaviour. No anomalies detected."},
    1: {"icon": "🔓", "label": "Compromised", "description": "Account shows signs of compromise or early mule readiness."},
    2: {"icon": "🔄", "label": "Layering", "description": "Active participation in rapid money movement/obfuscation."},
    3: {"icon": "⏳", "label": "Pre-Cashout", "description": "Funds pooling. One hop away from exit node."},
    4: {"icon": "🚨", "label": "Exit Imminent", "description": "Critical phase. Funds are actively attempting to leave the tracked network."}
}

def predict(dna_df):
    """Wrapper around predict.py to output predictions with stage/probability matching app needs."""
    # Check if we have anything to predict
    if dna_df is None or dna_df.empty:
        return dna_df
        
    df = dna_df.copy()
    
    # We need to map DNA df features to predict features if possible.
    # The real model uses 7 features.
    mock_df = pd.DataFrame({
        'transaction_amount': df.get('amount_anomaly', 0.0) * 15000 + 100,
        'income': df.get('burst_score', 0.0) * 80000 + 30000, 
        'device_fraud_count': df.get('dna_score', 0.0) / 10.0,
        'account_age_days': np.where(df.get('velocity_score', 0.0) > 0.5, 15, 300),  # High velocity -> newer account mock
        'transaction_hour': 2, # Nighttime penalty baseline
        'is_foreign_ip': (df.get('hop_proximity', 0.0) > 0.5).astype(int),
        'month': 1
    })
    
    predictions, probabilities = _pred.batch_predict(mock_df)
    
    df['prediction'] = predictions
    df['cashout_probability'] = [p * 100 for p in probabilities]
    
    # Map probability to stage (0-4)
    stages = []
    for prob in probabilities:
        if prob > 0.90: stages.append(4)
        elif prob > 0.70: stages.append(3)
        elif prob > 0.50: stages.append(2)
        elif prob > 0.20: stages.append(1)
        else: stages.append(0)
        
    df['stage'] = stages
    df['stage_label'] = [STAGE_DEFINITIONS[s]['label'] for s in stages]
    
    # Mock time to cashout based on stage
    df['time_to_cashout_min'] = df['stage'].apply(lambda x: max(0, (4 - x) * 15) if x > 0 else -1)
    
    return df

def predict_ring_summary(pred_df, ring_accounts: list) -> dict:
    """Summarize ring probabilities."""
    if pred_df is None or pred_df.empty or not ring_accounts:
        return {}
        
    ring_df = pred_df[pred_df['node'].isin(ring_accounts)]
    if ring_df.empty:
        return {}
        
    max_stage = int(ring_df['stage'].max())
    max_prob = float(ring_df['cashout_probability'].max())
    min_ttc = float(ring_df[ring_df['time_to_cashout_min'] >= 0]['time_to_cashout_min'].min()) if not ring_df[ring_df['time_to_cashout_min'] >= 0].empty else -1
    
    return {
        "max_stage": max_stage,
        "max_cashout_probability": max_prob,
        "min_time_to_cashout": min_ttc,
        "dominant_label": STAGE_DEFINITIONS[max_stage]['label'],
        "estimated_loss_usd": 50000.0 * len(ring_accounts) * (max_prob/100)
    }
