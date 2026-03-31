import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

st.title('❤️ Heart Disease Prediction')
st.markdown('Provide the following details')

# Inputs
age = st.slider('Age', 18, 100, 40)
sex = st.selectbox("SEX", ['M', 'F'])
cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
trestbps = st.slider('Resting Blood Pressure', 80, 200, 120)
chol = st.slider('Serum Cholestoral in mg/dl', 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['True', 'False'])
restecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
maxhr = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ['Yes', 'No'])
oldpeak = st.slider('ST Depression', 0.0, 10.0, 1.0)
slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

if st.button('Predict'):

    # Create input (MATCH TRAINING)
    input_dict = {
        'Age': age,
        'RestingBP': trestbps,
        'Cholesterol': chol,
        'MaxHR': maxhr,
        'Oldpeak': oldpeak,
        'Sex': 1 if sex == 'M' else 0,
        'FastingBS': 1 if fbs == 'True' else 0,
        'ExerciseAngina': 1 if exang == 'Yes' else 0
    }

    input_df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # One-hot columns
    input_df['ChestPainType_' + cp] = 1
    input_df['RestingECG_' + restecg] = 1
    input_df['ST_Slope_' + slope] = 1

    # Remove target if exists
    if 'HeartDisease' in input_df.columns:
        input_df = input_df.drop('HeartDisease', axis=1)

    # Reorder
    input_df = input_df[columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # ✅ Probability
    proba = model.predict_proba(input_scaled)[0]
    risk_percent = round(proba[1] * 100, 2)

    # 🔥 OUTPUT SECTION
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"High Risk of Heart Disease: {risk_percent}%")
    else:
        st.success(f"Low Risk of Heart Disease: {risk_percent}%")

    # 📊 Risk meter
    st.metric(label="Risk Score (%)", value=f"{risk_percent}%")
    st.progress(int(risk_percent))

    # 🧠 Interpretation
    if risk_percent < 30:
        st.info("Low risk. Maintain a healthy lifestyle.")
    elif risk_percent < 70:
        st.warning("Moderate risk. Consider consulting a doctor.")
    else:
        st.error("High risk. Strongly recommend medical attention.")

    # 🔍 Explain factors
    st.subheader("Key Factors Influencing Prediction")

    reasons = []

    if oldpeak > 2:
        reasons.append("High ST depression (Oldpeak)")
    if exang == 'Yes':
        reasons.append("Exercise-induced angina")
    if cp == 'ASY':
        reasons.append("Asymptomatic chest pain")
    if maxhr < 120:
        reasons.append("Low maximum heart rate")
    if trestbps > 140:
        reasons.append("High resting blood pressure")

    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("No major risk factors detected.")

    # ⚠️ Disclaimer
    st.caption("⚠️ This prediction is based on a machine learning model and is not a medical diagnosis.")