import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load(r"C:\Users\abc\Desktop\Thyroid_Cancer\model\thyroid_recurrence_model.pkl")
encoders = joblib.load(r"C:\Users\abc\Desktop\Thyroid_Cancer\model\label_encoders.pkl")
target_encoder = encoders['Recurred']

st.title("🩺 Thyroid Cancer Recurrence Prediction")

# Collect user input
age = st.number_input("Age", min_value=1, max_value=120, value=45)
gender = st.selectbox("Gender", encoders["Gender"].classes_)
smoking = st.selectbox("Smoking", encoders["Smoking"].classes_)
hx_smoking = st.selectbox("History of Smoking", encoders["Hx Smoking"].classes_)
hx_radio = st.selectbox("History of Radiotherapy", encoders["Hx Radiothreapy"].classes_)
thyroid_func = st.selectbox("Thyroid Function", encoders["Thyroid Function"].classes_)
physical_exam = st.selectbox("Physical Examination", encoders["Physical Examination"].classes_)
adenopathy = st.selectbox("Adenopathy", encoders["Adenopathy"].classes_)
pathology = st.selectbox("Pathology", encoders["Pathology"].classes_)
focality = st.selectbox("Focality", encoders["Focality"].classes_)
risk = st.selectbox("Risk", encoders["Risk"].classes_)
T_val = st.selectbox("T Stage", encoders["T"].classes_)
N_val = st.selectbox("N Stage", encoders["N"].classes_)
M_val = st.selectbox("M Stage", encoders["M"].classes_)
stage = st.selectbox("Stage", encoders["Stage"].classes_)
response = st.selectbox("Response", encoders["Response"].classes_)

if st.button("Predict Recurrence"):
    # Prepare input data
    user_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Smoking": smoking,
        "Hx Smoking": hx_smoking,
        "Hx Radiothreapy": hx_radio,
        "Thyroid Function": thyroid_func,
        "Physical Examination": physical_exam,
        "Adenopathy": adenopathy,
        "Pathology": pathology,
        "Focality": focality,
        "Risk": risk,
        "T": T_val,
        "N": N_val,
        "M": M_val,
        "Stage": stage,
        "Response": response
    }])

    # Encode using saved encoders
    for col in user_data.columns:
        if col in encoders:
            le = encoders[col]
            user_data[col] = le.transform(user_data[col].astype(str))

    # Predict
    pred_class = model.predict(user_data)[0]
    pred_prob = model.predict_proba(user_data)[0][1]
    pred_label = target_encoder.inverse_transform([pred_class])[0]

    st.subheader(f"Prediction: **{pred_label}**")
    st.write(f"Probability of Recurrence: **{pred_prob:.2f}**")
