import streamlit as st
import joblib
import pandas as pd


model = joblib.load("fraud_model.sav")
feature_list = joblib.load("model_features.sav")

st.title("Fraud Detection System")


amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, step=0.01)
oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, step=0.01)

type_selected = st.selectbox("Transaction Type", 
                             ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])


input_data = {
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "oldbalanceDest": oldbalanceDest,
}

for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
    input_data[f"type_{t}"] = 1 if t == type_selected else 0

input_df = pd.DataFrame([input_data]).reindex(columns=feature_list, fill_value=0)

def rule_based_risk():
    score = 0
    reasons = []

    if amount > oldbalanceOrg:
        score += 3
        reasons.append("Transaction amount exceeds sender balance")
    
    if amount > 100000:
        score += 3
        reasons.append("Very large transaction amount")
    
    if oldbalanceDest == 0 and amount > 10000:
        score += 3
        reasons.append("Receiver balance is zero but receiving high amount")

    if type_selected in ["TRANSFER", "CASH_OUT"]:
        score += 2
        reasons.append("High-risk transaction type")

    if amount > 50000:
        score += 2
        reasons.append("Large transaction amount")

    if oldbalanceOrg == 0:
        score += 2
        reasons.append("Sender has zero starting balance")

    if amount > 10000:
        score += 1
        reasons.append("Amount higher than typical daily usage")

    if oldbalanceDest == 0:
        score += 1
        reasons.append("Receiver account is empty")

    if score == 0:
        reasons.append("No strong indicators of fraud detected")

    return min(score * 8, 80), reasons

if st.button("Analyze Transaction"):
    ml_risk = round(model.predict_proba(input_df)[0][1] * 100, 2)
    rule_risk, reasons = rule_based_risk()
    final_risk = round((ml_risk + rule_risk) / 2, 2)

    st.markdown(f"### Model Risk: **{ml_risk}%**")
    st.markdown(f"### Rule-Based Risk: **{rule_risk}%**")
    st.markdown(f"### Overall Risk Score: **{final_risk}%**")

    st.markdown("---")
    st.markdown("### Risk Factors Identified:")
    for r in reasons:
        st.write(f"- {r}")

    st.markdown("---")
    st.markdown("### Recommendation:")

    if final_risk >= 70:
        st.write("Block Transaction. High likelihood of fraud.")
    elif final_risk >= 50:
        st.write("Hold Transaction. Verify customer credentials before processing.")
    elif final_risk >= 30:
        st.write("Proceed with caution. Additional manual review suggested.")
    else:
        st.write("Transaction appears low-risk and may proceed normally.")
