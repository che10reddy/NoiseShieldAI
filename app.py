# =========================
# NoiseShield AI · Quantum-Inspired Diagnostics
# Optimized for Streamlit Cloud / Codespaces / Ubuntu
# =========================

import os
# 🔧 Disable directory watchers to prevent "inotify watch limit reached" error
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import numpy as np, pandas as pd, datetime as dt
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Optional PDF report (lightweight)
try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

# ========== App Meta ==========
st.set_page_config(page_title="NoiseShield AI", page_icon="🔰", layout="centered")

# ========== Session Init ==========
for key, default in {
    "results": {"Soil": None, "Health": None, "Water": None},
    "history": {"Soil": [], "Health": [], "Water": []},
    "last_stable": {"Soil": None, "Health": None, "Water": None},
    "theme_mode": "Dark",
}.items():
    st.session_state.setdefault(key, default)

# ========== Theme ==========
theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"],
    index=0 if st.session_state["theme_mode"] == "Light" else 1)
st.session_state["theme_mode"] = theme_choice
is_dark = st.session_state["theme_mode"] == "Dark"

bg_color   = "#0E1117" if is_dark else "#FFFFFF"
text_color = "#FAFAFA" if is_dark else "#111111"
accent     = "#00B4B4" if is_dark else "#0A84FF"
panel_bg   = "#151922" if is_dark else "#F7F9FC"

st.markdown(f"""
<style>
body, .stApp {{ background-color: {bg_color}; color: {text_color}; }}
.stButton>button {{ background-color: {accent}; color: white; border-radius: 8px; font-weight: 600; border: 0; }}
.stProgress > div > div {{ background-color: {accent} !important; }}
.block-container {{ padding-top: 1.0rem; max-width: 980px; }}
h2, h3, h4, h5, h6 {{ color: {accent}; }}
label, .stTextInput label, .stNumberInput label {{ color: {'#EDEDED' if is_dark else '#1F1F1F'} !important; font-weight: 600; }}
.panel {{ background:{panel_bg}; padding:12px 14px; border-radius:10px; border:1px solid {accent}30; }}
* {{ transition: background-color .25s ease, color .25s ease; }}
</style>
""", unsafe_allow_html=True)

# ========== Language Dictionary (English only shown for brevity) ==========
L = {
    "title": "NoiseShield AI · Quantum-Inspired Diagnostics",
    "tabs": ["Soil", "Health", "Water", "Quantum View", "Reports", "Dashboard", "Cross-Domain Stability"],
    "controls": "App Controls",
    "noise": "Simulated Sensor Noise (%)",
    "predicted": "Predicted Result",
    "confidence": "Confidence",
    "baseline": "Baseline prob",
    "var": "Disagreement var",
    "download_pdf": "Download PDF Report",
    "pdf_missing": "Install 'fpdf' to enable PDF report.",
    "why": "Why this result?",
    "unstable": "Data unstable — showing last safe reading",
    "domain": "Domain", "prediction": "Prediction",
    "prob": "Probability", "conf": "Confidence",
    "time": "Timestamp",
    "overall": "Overall Sustainability Confidence",
    "excellent": "🟢 Excellent", "moderate": "🟡 Moderate", "needs": "🔴 Needs Work",
    "trend": "Confidence Trends", "caption": "Quantum-inspired, offline diagnostics tool.",
    "note_panel": "Designed for YCS & College Admissions: explainable, robust, and offline."
}

# ========== Banner ==========
st.markdown(f"<h2 style='text-align:center; font-weight:800;'>{L['title']}</h2>", unsafe_allow_html=True)

# ========== Helpers ==========
def seed_rng(seed=42): return np.random.default_rng(seed)

def inject_noise(X, pct, rng=None):
    if pct <= 0: return X.copy()
    rng = rng or seed_rng(123)
    return X * (1.0 + rng.normal(0, pct/100.0, size=X.shape))

def make_submodels_from(base_lr, eps=0.03, n=3):
    scaler = base_lr.named_steps['standardscaler']
    lr = base_lr.named_steps['logisticregression']
    subs = []
    for k in range(n):
        lr_k = LogisticRegression()
        lr_k.classes_ = lr.classes_
        lr_k.coef_ = lr.coef_ * (1 + (k-1)*eps)
        lr_k.intercept_ = lr.intercept_ * (1 + (k-1)*eps)
        pipe_k = make_pipeline(StandardScaler())
        pipe_k.fit(np.zeros((1, scaler.scale_.shape[0])), [0])
        pipe_k.named_steps['standardscaler'].mean_ = scaler.mean_.copy()
        pipe_k.named_steps['standardscaler'].scale_ = scaler.scale_.copy()
        pipe_k.steps.append(('logisticregression', lr_k))
        subs.append(pipe_k)
    return subs

def ensemble_predict_proba(submodels, X_row):
    probs = np.array([m.predict_proba(X_row)[0,1] for m in submodels])
    var = float(np.var(probs))
    w = np.ones_like(probs) / len(probs) if var > 0.02 else np.exp(-(probs - probs.mean())**2 / (2*0.0025))
    w /= w.sum()
    return float(w @ probs), probs, w, var

def df_to_csv_bytes(df):
    sio = StringIO()
    df.to_csv(sio, index=False)
    return sio.getvalue().encode()

def pdf_report_bytes(domain, inputs, label, conf, noise):
    if not HAS_FPDF: return None
    pdf = FPDF(); pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "NoiseShield Diagnostic Report", ln=1, align='C')
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Domain: {domain}\nInputs: {inputs}\nPrediction: {label}\nConfidence: {conf}%\nNoise: {noise}%")
    return pdf.output(dest='S').encode('latin-1')

# ========== Training ==========
def synth_health_data(n=400, rng=None):
    rng = rng or seed_rng(1)
    Hb = rng.normal(13, 2.2, n).clip(6, 20)
    WBC = rng.normal(7000, 2500, n).clip(2000, 30000)
    PLT = rng.normal(250000, 80000, n).clip(70000, 800000)
    Temp = rng.normal(36.8, 0.7, n).clip(34.5, 41.5)
    Pulse = rng.normal(80, 15, n).clip(45, 160)
    X = np.column_stack([Hb, WBC, PLT, Temp, Pulse])
    y = ((Hb < 11) | ((Temp > 37.8) & (WBC > 10000)) | (PLT < 120000)).astype(int)
    return X, y

def synth_water_data(n=400, rng=None):
    rng = rng or seed_rng(2)
    pH = rng.normal(7.1, 0.6, n).clip(4.5, 9.5)
    turb = np.abs(rng.normal(5, 15, n)).clip(0, 200)
    tds = np.abs(rng.normal(300, 250, n)).clip(50, 2500)
    ec = np.abs(rng.normal(600, 400, n)).clip(50, 4500)
    temp = rng.normal(24, 6, n).clip(5, 45)
    X = np.column_stack([pH, turb, tds, ec, temp])
    y = ((turb > 10) | (tds > 1000) | (ec > 2000) | (pH < 6.0) | (pH > 8.5)).astype(int)
    return X, y

@st.cache_resource
def train_baselines():
    Xh, yh = synth_health_data()
    Xw, yw = synth_water_data()
    hp = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    wp = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    hp.fit(Xh, yh); wp.fit(Xw, yw)
    return hp, wp

health_pipe, water_pipe = train_baselines()

# ========== Controls ==========
st.sidebar.header(L["controls"])
noise_pct = st.sidebar.slider(L["noise"], 0, 100, 0, step=5)
tabs = st.tabs(L["tabs"])

# ========== Health Tab (example) ==========
with tabs[1]:
    st.subheader("Health Diagnostics")
    hb = st.number_input("Hemoglobin (g/dL)", 0.0, 25.0, 12.5)
    wbc = st.number_input("WBC (cells/µL)", 0.0, 30000.0, 7000.0)
    pltlt = st.number_input("Platelets (cells/µL)", 0.0, 900000.0, 250000.0)
    temp = st.number_input("Body Temp (°C)", 30.0, 45.0, 36.8)
    pulse = st.number_input("Pulse Rate (bpm)", 30.0, 200.0, 80.0)
    X0 = np.array([[hb, wbc, pltlt, temp, pulse]])
    Xn = inject_noise(X0, noise_pct)

    if st.button("Run Health Analysis"):
        p_lr = health_pipe.predict_proba(Xn)[0,1]
        subs = make_submodels_from(health_pipe, eps=0.04)
        p_ens, sub_probs, _, var = ensemble_predict_proba(subs, Xn)
        y_pred = int(p_ens >= 0.5)
        label = "Possible Condition" if y_pred else "Healthy"
        conf = round(p_ens * 100, 2)
        st.write(f"**{L['predicted']}**: {label}")
        st.progress(int(conf))
        st.write(f"{L['confidence']}: {conf}% · {L['baseline']}: {p_lr:.2f} · {L['var']}: {var:.4f}")

        fig, ax = plt.subplots()
        ax.bar([f"Sub{i+1}" for i in range(len(sub_probs))], sub_probs)
        ax.axhline(p_ens, color='r', linestyle='--', label='Final')
        ax.set_ylim(0,1); ax.legend(); st.pyplot(fig)

        if HAS_FPDF:
            pdf_bytes = pdf_report_bytes("Health", {"Hemoglobin":hb, "WBC":wbc}, label, conf, noise_pct)
            st.download_button(f"📄 {L['download_pdf']}", pdf_bytes, "noiseshield_health_report.pdf")
        else:
            st.caption(L["pdf_missing"])

# ========== Quantum View ==========
with tabs[3]:
    st.markdown(f"<div class='panel'>{L['note_panel']}</div>", unsafe_allow_html=True)

# ========== Dashboard ==========
with tabs[5]:
    st.subheader(L["overall"])
    res = st.session_state["results"]
    vals = [r["confidence"] if r else 0 for r in res.values()]
    overall = round(np.mean(vals), 1)
    status = L["excellent"] if overall >= 80 else (L["moderate"] if overall >= 50 else L["needs"])
    st.metric(L["overall"], f"{overall}%", status)
    st.caption(L["caption"])
