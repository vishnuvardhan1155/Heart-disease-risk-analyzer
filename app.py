import streamlit as st
import pandas as pd
import pickle

# Load Model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️"
)

st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter patient details below and click Predict.")

# Age
age = st.slider("Age", 18, 100, 30)

# Gender
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)
sex = 1 if gender == "Male" else 0

# Chest Pain
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

# Blood Pressure
trestbps = st.slider(
    "Resting Blood Pressure",
    80,
    220,
    120
)

# Cholesterol
chol = st.slider(
    "Cholesterol Level",
    100,
    600,
    200
)

# Fasting Blood Sugar
fbs_text = st.selectbox(
    "Fasting Blood Sugar",
    ["No", "Yes"]
)

fbs = 1 if fbs_text == "Yes" else 0

# ECG
restecg = st.selectbox(
    "Rest ECG Result",
    [0, 1, 2]
)

# Heart Rate
thalach = st.slider(
    "Maximum Heart Rate",
    60,
    220,
    150
)

# Exercise Angina
exang_text = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

exang = 1 if exang_text == "Yes" else 0

# Old Peak
oldpeak = st.slider(
    "Old Peak",
    0.0,
    6.5,
    1.0
)

# Slope
slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

# CA
ca = st.selectbox(
    "Major Vessels (CA)",
    [0, 1, 2, 3, 4]
)

# Thal
thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)

if st.button("🔍 Predict"):

    sample = pd.DataFrame(
        [[age, sex, cp, trestbps, chol,
          fbs, restecg, thalach,
          exang, oldpeak, slope,
          ca, thal]],
        columns=[
            'age','sex','cp','trestbps',
            'chol','fbs','restecg',
            'thalach','exang',
            'oldpeak','slope',
            'ca','thal'
        ]
    )

    result = model.predict(sample)

    if result[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
st.info(
    "This project uses Machine Learning (Logistic Regression) to predict heart disease risk."
)