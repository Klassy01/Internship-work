import streamlit as st
import pandas as pd
import joblib
import warnings

# Ignore sklearn version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load trained model
MODEL_PATH = "mobile_price_predictor (1).pkl"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"❌ Model file not found at path: {MODEL_PATH}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Streamlit app setup
st.set_page_config(page_title="📱 Mobile Price Predictor", layout="centered")
st.title("📱 Mobile Price Predictor")
st.markdown("Enter your phone specs to get the **predicted price in ₹**.")

# Input fields
company = st.selectbox("Brand", ["Samsung", "Xiaomi", "Realme", "Infinix", "Vivo", "Oppo", "Motorola", "Apple", "Others"])
processor = st.text_input("Processor Name (e.g., Snapdragon 720G)", "Snapdragon 720G")
ram = st.number_input("RAM (GB)", min_value=1, max_value=24, value=6)
storage = st.number_input("Inbuilt Storage (GB)", min_value=8, max_value=512, value=128)
battery = st.number_input("Battery Capacity (mAh)", min_value=1000, max_value=10000, value=5000)
display = st.number_input("Display Size (inches)", min_value=4.0, max_value=7.5, value=6.5, step=0.1)
camera = st.number_input("Total Camera Megapixels (e.g., sum of rear/front)", min_value=5, max_value=200, value=64)

# Predict button
if st.button("📊 Predict Price"):
    input_df = pd.DataFrame([{
        "company": company,
        "Processor_name": processor,
        "RAM": ram,
        "Battery_mAh": battery,
        "Display_inch": display,
        "Camera_MP": camera,
        "Storage_GB": storage
    }])

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Mobile Price: ₹{round(prediction, 2)}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
