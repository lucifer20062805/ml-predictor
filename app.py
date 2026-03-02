import streamlit as st
import predict as ml_predict
import database as db
import alerts as alert_engine
import gemini_layer

st.set_page_config(
    page_title="ChronoTrace | Manual Predictor",
    page_icon="🤖",
    layout="wide",
)

# Reuse the CSS
st.markdown("""
<style>
/* Base Dark Theme Overrides */
[data-testid="stAppViewContainer"] { background-color: #060b13; }
[data-testid="stHeader"] { background-color: rgba(6, 11, 19, 0.8); }

/* Custom AI Panel Styling */
.ai-panel {
    background: linear-gradient(135deg, #0d1520, #0a111a);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: inset 0 1px 1px rgba(255,255,255,0.02);
}

.sec-hdr {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #475569;
    border-bottom: 1px solid #1e2a3a;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

.sec-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #3b82f6;
    display: inline-block;
    margin-right: 0.5rem;
}

/* Alert Styling */
.alert-card {
    background: #0d1117;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    border: 1px solid #1e2a3a;
    border-left: 4px solid var(--alert-color);
}
.alert-meta {
    font-size: 0.7rem; color: #64748b; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 0.3rem;
}

/* Number inputs to match aesthetic */
div[data-baseweb="input"] {
    background-color: #090e17;
    border-color: #1e2a3a;
}
div[data-baseweb="input"] > input {
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Manual AI Fraud Prediction")
st.markdown("<p style='color:#94a3b8;'>Directly query the commercial-grade predictive AI model to evaluate isolated transactions.</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown("""
    <div class="ai-panel">
        <h4 style="margin-top:0; color:#e2e8f0;">Transaction Details</h4>
        <p style="font-size:0.8rem; color:#94a3b8;">Enter details to predict fraud probability instantly.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form_standalone"):
        ca, cb = st.columns(2)
        with ca:
            tx_amount = st.number_input("Transfer Amount ($)", min_value=0.0, value=1500.0, step=100.0)
            income = st.number_input("Income", min_value=0.0, value=75000.0, step=1000.0)
            acct_age = st.number_input("Account Age (Days)", min_value=0, value=365, step=10)
            device_fraud = st.number_input("Device Fraud", min_value=0, value=0, step=1)
        with cb:
            tx_hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=14, step=1)
            foreign_ip = st.selectbox("Foreign IP", options=[0, 1], index=0, format_func=lambda x: "Yes" if x==1 else "No")
            month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=1, step=1)
            
        submit = st.form_submit_button("Predict Fraud Risk", use_container_width=True)
        
    if submit:
        with st.spinner("AI Evaluating..."):
            features = [tx_amount, income, device_fraud, acct_age, tx_hour, foreign_ip, month]
            pred, prob = ml_predict.predict_fraud(features)
            input_data = f"Amt: {tx_amount}, Inc: {income}, F_IP: {foreign_ip}, Hr: {tx_hour}"
            
            # Save to DB
            db.save_predictions(input_data, pred)
            
            alert_info = alert_engine.check_prediction_alert(prob)
            explanation_data = {
                'transaction_amount': tx_amount, 'income': income, 'device_fraud_count': device_fraud,
                'account_age_days': acct_age, 'transaction_hour': tx_hour, 'is_foreign_ip': foreign_ip, 'month': month
            }
            explanation = gemini_layer.explain_prediction(pred, prob, explanation_data)
            
            st.markdown(f"""
            <div class="alert-card" style="--alert-color:{alert_info['color']}; margin-top: 1rem;">
                <div class="alert-meta">{alert_info['badge']} {alert_info['severity']} · PREDICTION RESULT</div>
                <div class="alert-msg" style="font-size:1.1rem; font-weight:600; color:{alert_info['color']};">{alert_info['message']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="ai-panel" style="margin-top: 1rem;">
                <div class="ai-badge" style="background:rgba(59,130,246,0.15); color:#3b82f6; display:inline-block; padding:0.2rem 0.5rem; border-radius:4px; font-size:0.75rem; font-weight:600; margin-bottom:0.8rem;">
                    ✨ Gemini AI Insight
                </div>
                <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.6;">
                    {explanation}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
with col2:
    st.markdown('<div class="sec-hdr"><span class="sec-dot"></span>PREDICTION HISTORY</div>', unsafe_allow_html=True)
    history_df = db.get_predictions(limit=15)
    
    if not history_df.empty:
        st.dataframe(
            history_df[['id', 'timestamp', 'input_data', 'prediction']], 
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No prediction history found.")
