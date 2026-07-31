"""
app.py
------
Streamlit demo app for the Spam Email/SMS Classifier.
Polished UI version — built for live presentations/demos.

Run locally with:
    streamlit run app.py
"""

import time
import pandas as pd
import streamlit as st
import joblib
from preprocess import clean_text

st.set_page_config(
    page_title="Spam Shield | AI Message Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Load trained model + vectorizer + metrics
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/spam_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    with open("model/best_model_name.txt") as f:
        model_name = f.read().strip()
    return model, vectorizer, model_name


@st.cache_data
def load_metrics():
    try:
        df = pd.read_csv("outputs/model_comparison.csv")
        return df
    except Exception:
        return None


model, vectorizer, model_name = load_artifacts()
metrics_df = load_metrics()

# ---------------------------------------------------------------------
# Custom CSS — theme, hero header, cards, buttons
# ---------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cambria&family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header {visibility: hidden;}

    /* ---------- Animated 3D background ---------- */
    .stApp {
        background: radial-gradient(circle at 15% 0%, #1f2761 0%, #12153a 45%, #0b0c24 100%);
        position: relative;
        overflow-x: hidden;
    }
    .orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(70px);
        opacity: 0.35;
        z-index: 0;
        pointer-events: none;
    }
    .orb1 { width: 420px; height: 420px; background: #3B5BDB; top: -120px; right: -100px; animation: floatA 14s ease-in-out infinite; }
    .orb2 { width: 340px; height: 340px; background: #7c4dff; bottom: -100px; left: -80px; animation: floatB 17s ease-in-out infinite; }
    .orb3 { width: 260px; height: 260px; background: #1B9C55; top: 45%; right: 10%; opacity: 0.18; animation: floatA 20s ease-in-out infinite reverse; }
    @keyframes floatA {
        0%, 100% { transform: translate(0,0) scale(1); }
        50% { transform: translate(-30px, 40px) scale(1.08); }
    }
    @keyframes floatB {
        0%, 100% { transform: translate(0,0) scale(1); }
        50% { transform: translate(40px, -30px) scale(1.1); }
    }

    /* ---------- Hero with 3D depth ---------- */
    .hero {
        position: relative;
        z-index: 1;
        padding: 2.6rem 2.2rem 2.2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(59,91,219,0.32) 0%, rgba(30,39,97,0.6) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1.8rem;
        box-shadow:
            0 1px 0 rgba(255,255,255,0.15) inset,
            0 20px 45px rgba(0,0,0,0.45),
            0 2px 10px rgba(59,91,219,0.3);
        transform: perspective(1000px) rotateX(0.5deg);
    }
    .hero-kicker {
        color: #9db4ff;
        font-weight: 700;
        letter-spacing: 3px;
        font-size: 0.78rem;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        color: #ffffff;
        font-size: 2.7rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 18px rgba(59,91,219,0.55);
    }
    .hero-title .shield-icon {
        display: inline-block;
        animation: pulseGlow 2.4s ease-in-out infinite;
    }
    @keyframes pulseGlow {
        0%, 100% { filter: drop-shadow(0 0 4px rgba(157,180,255,0.4)); transform: scale(1); }
        50% { filter: drop-shadow(0 0 20px rgba(157,180,255,0.9)); transform: scale(1.08); }
    }
    .hero-sub {
        color: #c9d3f5;
        font-size: 1.02rem;
        max-width: 700px;
    }

    .badge-row { margin-top: 1.1rem; }
    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.09);
        color: #dfe6ff;
        border: 1px solid rgba(255,255,255,0.16);
        padding: 0.32rem 0.9rem;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        font-weight: 500;
        box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        transition: transform 0.2s ease;
    }
    .badge:hover { transform: translateY(-2px); }

    /* ---------- 3D glass cards ---------- */
    .glass-card {
        position: relative;
        z-index: 1;
        background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow:
            0 1px 0 rgba(255,255,255,0.12) inset,
            0 14px 34px rgba(0,0,0,0.4);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-card:hover {
        transform: perspective(900px) rotateX(1.5deg) translateY(-4px);
        box-shadow:
            0 1px 0 rgba(255,255,255,0.16) inset,
            0 22px 48px rgba(0,0,0,0.5);
    }

    .section-label {
        color: #9db4ff;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 2px;
        margin-bottom: 0.6rem;
    }

    .stTextArea textarea {
        background: rgba(255,255,255,0.06) !important;
        color: #f3f5ff !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3) inset !important;
    }
    .stTextArea textarea::placeholder { color: #8992b8 !important; }
    .stTextArea textarea:focus {
        border: 1px solid #3B5BDB !important;
        box-shadow: 0 0 0 3px rgba(59,91,219,0.35) !important;
    }

    /* ---------- 3D press-down buttons ---------- */
    div[data-testid="stButton"] button {
        border-radius: 12px;
        font-weight: 700;
        padding: 0.6rem 1.4rem;
        border: none;
        box-shadow: 0 6px 0 rgba(0,0,0,0.35), 0 10px 20px rgba(0,0,0,0.35);
        transition: transform 0.1s ease, box-shadow 0.1s ease;
    }
    div[data-testid="stButton"] button:hover { transform: translateY(-2px); }
    div[data-testid="stButton"] button:active {
        transform: translateY(4px);
        box-shadow: 0 2px 0 rgba(0,0,0,0.35), 0 4px 10px rgba(0,0,0,0.3);
    }

    .example-btn button {
        background: rgba(255,255,255,0.07) !important;
        color: #dfe6ff !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        box-shadow: 0 4px 0 rgba(0,0,0,0.25), 0 8px 14px rgba(0,0,0,0.3) !important;
    }

    /* ---------- Result cards: pop + glow + confetti/shake ---------- */
    .result-card {
        position: relative;
        overflow: hidden;
        border-radius: 20px;
        padding: 1.7rem 1.9rem;
        margin-top: 1rem;
        animation: popIn 0.5s cubic-bezier(.26,1.4,.44,1);
    }
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.85) translateY(10px); }
        60% { opacity: 1; transform: scale(1.03) translateY(-2px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }
    .result-spam {
        background: linear-gradient(135deg, rgba(214,69,69,0.26), rgba(214,69,69,0.07));
        border: 1px solid rgba(214,69,69,0.5);
        box-shadow: 0 0 0 rgba(214,69,69,0.5), 0 16px 36px rgba(214,69,69,0.25);
        animation: popIn 0.5s cubic-bezier(.26,1.4,.44,1), shake 0.5s ease 0.5s, alarmGlow 1.6s ease-in-out 1s infinite;
    }
    .result-ham {
        background: linear-gradient(135deg, rgba(27,156,85,0.26), rgba(27,156,85,0.07));
        border: 1px solid rgba(27,156,85,0.5);
        box-shadow: 0 16px 36px rgba(27,156,85,0.25);
        animation: popIn 0.5s cubic-bezier(.26,1.4,.44,1), safeGlow 1.8s ease-in-out 0.5s infinite;
    }
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20% { transform: translateX(-6px); }
        40% { transform: translateX(6px); }
        60% { transform: translateX(-4px); }
        80% { transform: translateX(4px); }
    }
    @keyframes alarmGlow {
        0%, 100% { box-shadow: 0 16px 36px rgba(214,69,69,0.25); }
        50% { box-shadow: 0 16px 46px rgba(214,69,69,0.55); }
    }
    @keyframes safeGlow {
        0%, 100% { box-shadow: 0 16px 36px rgba(27,156,85,0.25); }
        50% { box-shadow: 0 16px 46px rgba(27,156,85,0.5); }
    }
    .result-title { font-size: 1.55rem; font-weight: 800; margin-bottom: 0.2rem; }
    .result-spam .result-title { color: #ff8080; }
    .result-ham .result-title { color: #5be0a0; }
    .result-sub { color: #d9def0; font-size: 0.92rem; }

    /* confetti burst for Ham */
    .confetti-piece {
        position: absolute;
        top: -12px;
        font-size: 1.1rem;
        animation: confettiFall 1.6s ease-in forwards;
        z-index: 2;
    }
    @keyframes confettiFall {
        0% { transform: translateY(0) rotate(0deg); opacity: 1; }
        100% { transform: translateY(140px) rotate(280deg); opacity: 0; }
    }

    .metric-box {
        background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1rem 0.5rem;
        text-align: center;
        box-shadow: 0 10px 22px rgba(0,0,0,0.35);
        transition: transform 0.2s ease;
    }
    .metric-box:hover { transform: translateY(-3px) scale(1.02); }
    .metric-value { font-size: 1.6rem; font-weight: 800; color: #ffffff; }
    .metric-label { font-size: 0.72rem; color: #9db4ff; font-weight: 600; letter-spacing: 1px; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12153a 0%, #0b0c24 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * { color: #dfe6ff; }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }
</style>

<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar — project & model info
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ Spam Shield")
    st.caption("AI-powered SMS/Email spam detector")
    st.divider()

    st.markdown("**Active Model**")
    st.markdown(f"`{model_name}`")

    st.markdown("**Pipeline**")
    st.markdown("Text Cleaning → TF-IDF → ML Classifier")

    if metrics_df is not None:
        st.divider()
        st.markdown("**Model Performance**")
        best_row = metrics_df.sort_values("F1", ascending=False).iloc[0]
        st.metric("Accuracy", f"{best_row['Accuracy']*100:.1f}%")
        st.metric("F1-Score", f"{best_row['F1']*100:.1f}%")
        with st.expander("Compare all models"):
            st.dataframe(
                metrics_df.set_index("Model").style.format("{:.2%}"),
                use_container_width=True,
            )

    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("Python · Scikit-learn · TF-IDF · Streamlit")

    st.divider()
    st.caption("B.Tech Major Project — AI & ML")
    st.caption("Indresh Shukla · Shiv Kumar Tiwari")

# ---------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <div class="hero-kicker">AI · NLP · MACHINE LEARNING</div>
    <div class="hero-title">🛡️ Spam Shield</div>
    <div class="hero-sub">
        An intelligent message classifier that detects spam SMS and emails in real time
        using TF-IDF feature extraction and machine learning — trained, evaluated, and
        deployed end-to-end.
    </div>
    <div class="badge-row">
        <span class="badge">⚙️ Model: {model_name}</span>
        <span class="badge">⚡ Real-time prediction</span>
        <span class="badge">🌐 Live &amp; deployed</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------
left, right = st.columns([2.1, 1], gap="large")

if "message_text" not in st.session_state:
    st.session_state.message_text = ""

EXAMPLES = {
    "✅ Try a normal message": "Hey, are we still meeting for lunch at 1pm today?",
    "🚨 Try a spam message": "Congratulations! You have WON a free iPhone 15! Click here to claim your prize now!",
    "🚨 Try a phishing SMS": "URGENT: Your bank account has been suspended. Verify your details immediately at this link.",
}

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">MESSAGE INPUT</div>', unsafe_allow_html=True)

    message = st.text_area(
        "Message",
        value=st.session_state.message_text,
        height=150,
        placeholder="Type or paste an SMS/email message here...",
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES))
    for c, (label, text) in zip(cols, EXAMPLES.items()):
        with c:
            st.markdown('<div class="example-btn">', unsafe_allow_html=True)
            if st.button(label, key=label, use_container_width=True):
                st.session_state.message_text = text
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Analyze Message", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            with st.spinner("Analyzing message..."):
                time.sleep(0.4)
                cleaned = clean_text(message)
                vec = vectorizer.transform([cleaned])
                pred = model.predict(vec)[0]

                confidence = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vec)[0]
                    confidence = max(proba) * 100

            if pred == 1:
                st.markdown(f"""
                <div class="result-card result-spam">
                    <div class="result-title">🚨 SPAM DETECTED</div>
                    <div class="result-sub">This message shows strong characteristics of spam content.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-ham">
                    <div class="result-title">✅ LEGITIMATE MESSAGE</div>
                    <div class="result-sub">This message looks like a normal, safe communication (Ham).</div>
                </div>
                """, unsafe_allow_html=True)

            if confidence is not None:
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                st.progress(int(confidence), text=f"Model confidence: {confidence:.1f}%")

            with st.expander("🔬 See cleaned text used for prediction"):
                st.code(cleaned if cleaned else "(empty after cleaning)")

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">HOW IT WORKS</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Text Cleaning", "Lowercase, remove URLs, punctuation & stopwords"),
        ("2", "TF-IDF Vectorization", "Convert text into numeric feature vectors"),
        ("3", "ML Prediction", f"{model_name} classifies the message"),
        ("4", "Instant Result", "Spam / Ham shown with confidence score"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style="display:flex; gap:0.8rem; margin-bottom:0.9rem; align-items:flex-start;">
            <div style="background:#3B5BDB; color:white; width:26px; height:26px; border-radius:50%;
                        display:flex; align-items:center; justify-content:center; font-weight:700;
                        font-size:0.8rem; flex-shrink:0;">{num}</div>
            <div>
                <div style="color:#fff; font-weight:600; font-size:0.9rem;">{title}</div>
                <div style="color:#a9b2d9; font-size:0.78rem;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if metrics_df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">MODEL SNAPSHOT</div>', unsafe_allow_html=True)
        best_row = metrics_df.sort_values("F1", ascending=False).iloc[0]
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""<div class="metric-box"><div class="metric-value">{best_row['Accuracy']*100:.0f}%</div>
            <div class="metric-label">ACCURACY</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-box"><div class="metric-value">{best_row['Precision']*100:.0f}%</div>
            <div class="metric-label">PRECISION</div></div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        m3, m4 = st.columns(2)
        with m3:
            st.markdown(f"""<div class="metric-box"><div class="metric-value">{best_row['Recall']*100:.0f}%</div>
            <div class="metric-label">RECALL</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-box"><div class="metric-value">{best_row['F1']*100:.0f}%</div>
            <div class="metric-label">F1-SCORE</div></div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.caption(
    "⚠️ Note: This demo is trained on a sample dataset built for this project. "
    "For production-grade accuracy, retrain using the full real-world "
    "SMS Spam Collection dataset (see README.md)."
)
