import sys
import predict as ml_predict
import database as db
import alerts as alert_engine
import gemini_layer

def test_pipeline():
    print("=== TESTING AI PIPELINE ENTIRE FLOW ===")
    print("1. Preparing DB test...")
    db.init_db()
    
    # Mock data [tx_amount, income, item_count, age, hour, foreign_ip, month]
    test_data_for_prediction = [25000.0, 45000.0, 8, 2, 3, 1, 1] 
    
    print("\n--- Running Single Test ---")
    
    print("2. Predicting Fraud...")
    pred, prob = ml_predict.predict_fraud(test_data_for_prediction)
    print(f"Prediction result: {pred} (Fraud Prob: {prob*100:.2f}%)")
    
    print("3. Generating Alert...")
    alert = alert_engine.check_prediction_alert(prob)
    print(f"Alert: {alert['severity']} - {alert['message']}")
    
    print("4. Getting Gemini Explanation...")
    explanation_data = {
        'transaction_amount': test_data_for_prediction[0],
        'income': test_data_for_prediction[1],
        'device_fraud_count': test_data_for_prediction[2], # This was item_count in the mock data, mapping to device_fraud_count for explanation
        'account_age_days': test_data_for_prediction[3],
        'transaction_hour': test_data_for_prediction[4],
        'is_foreign_ip': test_data_for_prediction[5],
        'month': test_data_for_prediction[6]
    }
    explanation = gemini_layer.explain_prediction(pred, prob, explanation_data)
    print(f"Explanation: {explanation}")
    
    print("5. Saving to Database...")
    # Using a string representation of the input features for saving
    db.save_predictions(str(test_data_for_prediction), pred)
        
    print("\n--- Verifying Database ---")
    df = db.get_predictions(limit=5)
    print("Recent DB entries:")
    print(df)
    
if __name__ == "__main__":
    test_pipeline()
    print("\nALL TESTS COMPLETED SUCCESSFULLY!")