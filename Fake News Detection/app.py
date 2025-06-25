import streamlit as st
import pickle
import os

# ✅ Correct path to your model and vectorizer
model_path = "C:/Users/david/OneDrive/Desktop/Internship-Works/Fake News Detection/"

# Load model and vectorizer
with open(os.path.join(model_path, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(model_path, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's **Real** or **Fake**.")

# Text input
news_input = st.text_area("📝 Paste your news article here:", height=300)

# Predict button
if st.button("🔍 Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Transform input and predict
        news_vec = vectorizer.transform([news_input])
        prediction = model.predict(news_vec)[0]

        # Show result
        if prediction == "fake":
            st.error("❌ This news article is predicted to be **FAKE**.")
        else:
            st.success("✅ This news article is predicted to be **REAL**.")
