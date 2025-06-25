import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from streamlit_lottie import st_lottie_spinner

# ✅ Set Streamlit config
st.set_page_config(page_title="Credit Card Approval Predictor", layout="centered")

# ✅ Load Lottie animation
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

# ✅ Load model and pipeline
@st.cache_resource
def load_model_pipeline():
    model = joblib.load("gradient_boosting_model.sav")  # Update with correct path if needed
    pipeline = joblib.load("pipeline.pkl")
    return model, pipeline

model, pipeline = load_model_pipeline()

# ✅ Input form
st.title("💳 Credit Card Approval Predictor")
st.markdown("Fill out the information below to predict whether a credit card application will be approved.")

col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 70, 30)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    car_owner = st.radio("Own a Car?", ["Yes", "No"])

with col2:
    employment_status = st.selectbox("Employment Status", ["Working", "Unemployed", "Pensioner", "Student"])
    employment_years = st.slider("Years at Current Job", 0, 40, 5)
    education = st.selectbox("Education Level", ["Secondary", "Higher", "Incomplete", "Academic Degree"])
    dwelling = st.selectbox("Dwelling Type", ["House", "Rented", "Municipal", "With Parents"])
    prop_owner = st.radio("Own Property?", ["Yes", "No"])

# ✅ Additional contact info
st.subheader("📞 Contact Information")
col3, col4 = st.columns(2)

with col3:
    work_phone = st.radio("Has Work Phone?", ["Yes", "No"])
    phone = st.radio("Has Phone?", ["Yes", "No"])

with col4:
    email = st.radio("Has Email?", ["Yes", "No"])
    family_members = st.slider("Number of Family Members", 1, 10, 3)

# ✅ Predict button
if st.button("🚀 Predict"):
    user_data = {
        "Gender": gender[0],  # 'M' or 'F'
        "Age": -age * 365.25,
        "Marital status": marital_status,
        "Income": income,
        "Has a car": car_owner[0],  # 'Y' or 'N'
        "Employment status": employment_status,
        "Employment length": -employment_years * 365.25,
        "Education level": education,
        "Dwelling": dwelling,
        "Has a property": prop_owner[0],  # 'Y' or 'N'
        "Has a work phone": 1 if work_phone == "Yes" else 0,
        "Has a phone": 1 if phone == "Yes" else 0,
        "Has a mobile phone": 1 if phone == "Yes" else 0,  # assume same as phone
        "Has an email": 1 if email == "Yes" else 0,
        "Family member count": family_members,
        "Children count": 0,         # optional default
        "Account age": -3 * 365.25,  # optional default
        "Job title": "Unknown"       # optional default
    }

    input_df = pd.DataFrame([user_data])

    # 🎬 Animation
    lottie_loading = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_szlepvdh.json")

    with st_lottie_spinner(lottie_loading, height=200, width=200):
        try:
            input_processed = pipeline.transform(input_df)
            prediction = model.predict(input_processed)[0]
            prob = model.predict_proba(input_processed)[0][1]

            if prediction == 1:
                st.success(f"✅ Approved! (Confidence: {prob:.2%})")
            else:
                st.error(f"❌ Not Approved. (Confidence: {prob:.2%})")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
