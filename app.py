"""
app.py
------
Streamlit demo app for the Spam Email/SMS Classifier.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import joblib
from preprocess import clean_text

st.set_page_config(page_title="Spam SMS/Email Classifier", page_icon="📩", layout="centered")

# ---------------------------------------------------------------------
# Load trained model + vectorizer
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/spam_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    with open("model/best_model_name.txt") as f:
        model_name = f.read().strip()
    return model, vectorizer, model_name


model, vectorizer, model_name = load_artifacts()

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("📩 Spam Email/SMS Classifier")
st.caption(f"Model in use: **{model_name}** (TF-IDF + ML)")

st.write("Enter a message below and check whether it's **Spam** or **Ham** (not spam).")

message = st.text_area("Message", height=150, placeholder="Type or paste an SMS/email message here...")

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("Check Message", type="primary")

if predict_clicked:
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(message)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        # get confidence if the model supports probability estimates
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            confidence = max(proba) * 100

        if pred == 1:
            st.error("🚨 This message looks like **SPAM**.")
        else:
            st.success("✅ This message looks like **HAM** (not spam).")

        if confidence is not None:
            st.write(f"Confidence: **{confidence:.1f}%**")

        with st.expander("See cleaned text used for prediction"):
            st.code(cleaned if cleaned else "(empty after cleaning)")

st.divider()
st.caption(
    "Note: This demo is trained on a sample dataset built for this project. "
    "For production-grade accuracy, retrain using the full real-world "
    "SMS Spam Collection dataset (see README.md)."
)
