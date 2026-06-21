import streamlit as st
import pandas as pd
import pickle
import os

# Load trained model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

# Page Settings
st.set_page_config(
    page_title="AI Heart Disease Risk Analyzer",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ AI-Powered Heart Disease Risk Analyzer")
st.info(
    "This system predicts heart disease risk and provides AI-based healthcare alerts."
)

# -----------------------
# User Inputs
# -----------------------

age = st.slider("Age", 18, 100, 30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

sex = 1 if gender == "Male" else 0

cp_name = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ]
)

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}

cp = cp_map[cp_name]

trestbps = st.slider(
    "Resting Blood Pressure",
    80,
    220,
    120
)

chol = st.slider(
    "Cholesterol Level",
    100,
    600,
    200
)

fbs_text = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["No", "Yes"]
)

fbs = 1 if fbs_text == "Yes" else 0

restecg = st.selectbox(
    "Rest ECG Result",
    [0, 1, 2]
)

thalach = st.slider(
    "Maximum Heart Rate",
    60,
    220,
    150
)

exang_text = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

exang = 1 if exang_text == "Yes" else 0

oldpeak = st.slider(
    "Old Peak",
    0.0,
    6.5,
    1.0
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels (CA)",
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)

# -----------------------
# Prediction
# -----------------------

if st.button("🔍 Predict Risk"):

    sample = pd.DataFrame(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]],
        columns=[
            'age',
            'sex',
            'cp',
            'trestbps',
            'chol',
            'fbs',
            'restecg',
            'thalach',
            'exang',
            'oldpeak',
            'slope',
            'ca',
            'thal'
        ]
    )

    result = model.predict(sample)

    probability = model.predict_proba(sample)

    risk_score = round(probability[0][1] * 100, 2)

    st.subheader("📊 Risk Analysis")

    st.write(f"Risk Score: **{risk_score}%**")

    # Risk Level
    if risk_score < 40:
        st.success("🟢 LOW RISK")

    elif risk_score < 70:
        st.warning("🟡 MEDIUM RISK")

    else:
        st.error("🔴 HIGH RISK")

    # AI Agent
    st.subheader("🤖 AI Healthcare Agent Assessment")

    if result[0] == 1:

        st.error("🚨 HIGH-RISK PATIENT DETECTED")

        st.warning("""
        Emergency Healthcare Alert Generated

        Recommended Actions:
        • Consult a cardiologist immediately
        • Monitor blood pressure regularly
        • Reduce cholesterol intake
        • Avoid smoking and alcohol
        • Follow a heart-healthy diet
        """)

    else:

        st.success("✅ Patient appears to be at lower risk")

        st.info("""
        AI Recommendations:
        • Continue regular exercise
        • Maintain healthy diet
        • Monitor health periodically
        • Maintain healthy cholesterol levels
        """)

    # Save Patient Records
    record = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "BloodPressure": trestbps,
        "Cholesterol": chol,
        "RiskScore": risk_score,
        "Prediction": int(result[0])
    }])

    file_name = "patient_records.csv"

    if os.path.exists(file_name):
        record.to_csv(
            file_name,
            mode="a",
            header=False,
            index=False
        )
    else:
        record.to_csv(
            file_name,
            index=False
        )

    st.success("📁 Patient assessment record saved successfully.")