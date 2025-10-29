import streamlit as st
import numpy as np, pandas as pd, datetime as dt, uuid
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

st.set_page_config(page_title="NoiseShield AI", page_icon="🔰", layout="centered")

# ---------- THEME & BANNER ----------

if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "🌙 Dark"

theme_choice = st.sidebar.radio("Theme", ["🌞 Light", "🌙 Dark"],
                                index=0 if "Light" in st.session_state["theme_mode"] else 1)
st.session_state["theme_mode"] = theme_choice

is_dark = "Dark" in st.session_state["theme_mode"]
bg_color = "#0E1117" if is_dark else "#FFFFFF"
text_color = "#FAFAFA" if is_dark else "#000000"
accent = "#00B4B4" if is_dark else "#0077CC"
banner_color = "#00B4B4" if is_dark else "#0077CC"

st.markdown(f"""
    <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stButton>button {{
            background-color: {accent};
            color: white;
            border-radius: 8px;
            font-weight: 600;
        }}
        .stProgress > div > div {{
            background-color: {accent} !important;
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 1rem;
            max-width: 900px;
        }}
        h2, h3, h4, h5 {{
            color: {banner_color};
        }}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
* {
  transition: background-color 0.4s ease, color 0.4s ease;
}
</style>
""", unsafe_allow_html=True)

# ---- SDG Banner ----
st.markdown(f"""
<div style='text-align:center; margin-bottom:20px'>
  <h2 style='font-weight:700; color:{banner_color};'>
    🌍 NoiseShield AI | Quantum-Inspired Diagnostics
  </h2>
  <p style='font-size:15px; margin-top:-10px; color:{text_color};'>
    Supporting <strong>SDG 2 (Zero Hunger)</strong>, 
    <strong>SDG 3 (Good Health and Well-Being)</strong>, and 
    <strong>SDG 6 (Clean Water and Sanitation)</strong>
  </p>
  <div style='margin-top:6px;'>
    <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Sustainable_Development_Goal_2.svg/120px-Sustainable_Development_Goal_2.svg.png' width='55' style='margin:4px'/>
    <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Sustainable_Development_Goal_3.svg/120px-Sustainable_Development_Goal_3.svg.png' width='55' style='margin:4px'/>
    <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Sustainable_Development_Goal_6.svg/120px-Sustainable_Development_Goal_6.svg.png' width='55' style='margin:4px'/>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def seed_rng(seed=42):
    return np.random.default_rng(seed)

def inject_noise(X, pct, rng=None):
    if pct <= 0: return X.copy()
    if rng is None: rng = seed_rng(123)
    scale = pct / 100.0
    return X * (1.0 + rng.normal(0, scale, size=X.shape))

def make_submodels_from(base_lr, eps=0.03, n=3):
    scaler = base_lr.named_steps['standardscaler']
    lr = base_lr.named_steps['logisticregression']
    subs = []
    for k in range(n):
        lr_k = LogisticRegression()
        lr_k.classes_ = lr.classes_
        lr_k.coef_ = lr.coef_ * (1 + (k-1)*eps)
        lr_k.intercept_ = lr.intercept_ * (1 + (k-1)*eps)
        pipe_k = make_pipeline(StandardScaler(with_mean=scaler.with_mean, with_std=scaler.with_std))
        pipe_k.fit(np.zeros((1, scaler.scale_.shape[0])), [0])
        pipe_k.named_steps['standardscaler'].mean_ = scaler.mean_.copy()
        pipe_k.named_steps['standardscaler'].scale_ = scaler.scale_.copy()
        pipe_k.steps.append(('logisticregression', lr_k))
        subs.append(pipe_k)
    return subs

def ensemble_predict_proba(submodels, X_row):
    probs = np.array([m.predict_proba(X_row)[0,1] for m in submodels])
    var = np.var(probs)
    if var > 0.02:
        w = np.ones_like(probs) / len(probs)
    else:
        centered = np.exp(-(probs - probs.mean())**2 / (2*0.0025))
        w = centered / centered.sum()
    return float(w @ probs), probs, w, var

def linear_contribs(pipe, x_row, feature_names):
    scaler = pipe.named_steps['standardscaler']
    lr = pipe.named_steps['logisticregression']
    x_std = (x_row - scaler.mean_) / scaler.scale_
    contrib = (lr.coef_.ravel() * x_std.ravel())
    order = np.argsort(-np.abs(contrib))
    return [(feature_names[i], float(contrib[i])) for i in order]

def robustness_curve(pipe_base, subs, x_row, noise_levels):
    base_p, ens_p = [], []
    for nl in noise_levels:
        xn = inject_noise(x_row, nl)
        base_p.append(pipe_base.predict_proba(xn)[0,1])
        ens, _, _, _ = ensemble_predict_proba(subs, xn)
        ens_p.append(ens)
    return np.array(base_p), np.array(ens_p)

def make_csv_report(mode_label, features, x_row, pred_label, prob, conf, noise_pct):
    cols = list(features) + ["prediction","probability","confidence","noise_pct","timestamp","mode"]
    vals = list(map(lambda z: float(z), x_row.ravel())) + [pred_label, float(prob), float(conf), int(noise_pct), dt.datetime.now().isoformat(), mode_label]
    df = pd.DataFrame([vals], columns=cols)
    sio = StringIO(); df.to_csv(sio, index=False); return sio.getvalue().encode()

# ---------- Synthetic training ----------
def synth_health_data(n=400, rng=None):
    if rng is None: rng = seed_rng(1)
    Hb = rng.normal(13, 2.2, n).clip(6, 20)
    WBC = rng.normal(7000, 2500, n).clip(2000, 30000)
    PLT = rng.normal(250000, 80000, n).clip(70000, 800000)
    Temp = rng.normal(36.8, 0.7, n).clip(34.5, 41.5)
    Pulse = rng.normal(80, 15, n).clip(45, 160)
    X = np.column_stack([Hb, WBC, PLT, Temp, Pulse])
    y = ((Hb < 11) | ((Temp > 37.8) & (WBC > 10000)) | (PLT < 120000)).astype(int)
    return X, y

def synth_water_data(n=400, rng=None):
    if rng is None: rng = seed_rng(2)
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
    health_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    water_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    health_pipe.fit(Xh, yh); water_pipe.fit(Xw, yw)
    return health_pipe, water_pipe

health_pipe, water_pipe = train_baselines()

# ---------- UI ----------
st.markdown("## 🔰 NoiseShield AI")
st.caption("Quantum-inspired, offline, low-compute diagnostics for SDG 2, 3 & 6.")

lang = st.sidebar.selectbox("Language", ["English (EN)"])
noise_pct = st.sidebar.slider("Simulated Sensor Noise (%)", 0, 100, 0, step=5)

tab_soil, tab_health, tab_water, tab_quantum, tab_report, tab_dashboard = st.tabs(
    ["🌾 Soil (SDG 2)", "🧬 Health (SDG 3)", "💧 Water (SDG 6)", "⚛ Quantum View", "🧾 Reports", "📊 SDG Dashboard"]
)

if "last_stable" not in st.session_state:
    st.session_state["last_stable"] = None

# ---- SOIL TAB ----
with tab_soil:
    st.subheader("🌾 Soil Fertility Analysis (Offline)")

    col1, col2 = st.columns(2)
    with col1:
        spH = st.number_input("Soil pH", 3.0, 10.0, 6.5)
        N = st.number_input("Nitrogen (mg/kg)", 0.0, 200.0, 40.0)
        P = st.number_input("Phosphorus (mg/kg)", 0.0, 200.0, 30.0)
    with col2:
        K = st.number_input("Potassium (mg/kg)", 0.0, 300.0, 120.0)
        moist = st.number_input("Soil Moisture (%)", 0.0, 100.0, 25.0)

    X0s = np.array([[spH, N, P, K, moist]])
    Xns = inject_noise(X0s, noise_pct)

    @st.cache_resource
    def train_soil_baseline():
        rng = seed_rng(3)
        n = 400
        pH = rng.normal(6.5, 0.8, n).clip(3.5, 9.5)
        N = rng.normal(50, 25, n).clip(0, 200)
        P = rng.normal(40, 20, n).clip(0, 200)
        K = rng.normal(150, 60, n).clip(0, 300)
        M = rng.normal(30, 15, n).clip(0, 100)
        X = np.column_stack([pH, N, P, K, M])
        y = ((N < 30) | (P < 20) | (K < 80) | (pH < 5.5) | (pH > 8.5)).astype(int)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        pipe.fit(X, y)
        return pipe

    soil_pipe = train_soil_baseline()

    if st.button("Run Soil Analysis"):
        p_lr = soil_pipe.predict_proba(Xns)[0, 1]
        subs_s = make_submodels_from(soil_pipe, eps=0.04)
        p_ens, sub_probs, weights, var = ensemble_predict_proba(subs_s, Xns)
        y_pred = int(p_ens >= 0.5)
        label = "Nutrient Deficient" if y_pred == 1 else "Fertile"

        st.markdown(f"**Predicted Result:** {label}")
        conf = round(p_ens * 100, 2)
        st.progress(int(conf))
        st.write(f"🧠 Confidence: **{conf}%** · Baseline: {p_lr:.2f} · Var: {var:.4f}")

        fig, ax = plt.subplots()
        ax.bar([f"Sub{i+1}" for i in range(len(sub_probs))] + ["Final"], list(sub_probs) + [p_ens])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        expl = linear_contribs(soil_pipe, Xns, ["pH", "Nitrogen", "Phosphorus", "Potassium", "Moisture"])
        st.markdown("**Why this result? (Feature contributions)**")
        for f, c in expl:
            st.write(f"- {f}: {c:+.3f}")

        st.session_state["last_stable"] = {
            "mode": "Soil",
            "X": [float(v) for v in X0s.ravel()],
            "prob": float(p_ens),
            "confidence": float(conf),
            "noise_pct": int(noise_pct),
            "time": dt.datetime.now().isoformat(),
        }

        st.markdown("**Noise Robustness (probability vs noise):**")
        levels = [0, 25, 50, 75, 100]
        base_p, ens_p = robustness_curve(soil_pipe, subs_s, X0s, levels)
        fig2, ax2 = plt.subplots()
        ax2.plot(levels, base_p, marker="o", label="Baseline (LR)")
        ax2.plot(levels, ens_p, marker="o", label="Interference Ensemble")
        ax2.set_xlabel("Noise (%)")
        ax2.set_ylabel("Positive Probability")
        ax2.set_ylim(0, 1)
        ax2.legend()
        st.pyplot(fig2)

# ---- HEALTH TAB ----
with tab_health:
    st.subheader("🧬 Health Diagnostics (Offline)")
    col1, col2 = st.columns(2)
    with col1:
        hb = st.number_input("Hemoglobin (g/dL)", 0.0, 25.0, 12.5)
        wbc = st.number_input("WBC (cells/µL)", 0.0, 30000.0, 7000.0)
        pltlt = st.number_input("Platelets (cells/µL)", 0.0, 900000.0, 250000.0)
    with col2:
        temp = st.number_input("Body Temp (°C)", 30.0, 45.0, 36.8)
        pulse = st.number_input("Pulse Rate (bpm)", 30.0, 200.0, 80.0)
    X0 = np.array([[hb, wbc, pltlt, temp, pulse]])
    Xn = inject_noise(X0, noise_pct)

    if st.button("Run Health Analysis"):
        p_lr = health_pipe.predict_proba(Xn)[0,1]
        subs_h = make_submodels_from(health_pipe, eps=0.04)
        p_ens, sub_probs, weights, var = ensemble_predict_proba(subs_h, Xn)
        y_pred = int(p_ens >= 0.5)
        label = "Possible Condition" if y_pred==1 else "Healthy"
        st.markdown(f"**Predicted Result:** {label}")
        conf = round(p_ens * 100, 2)
        st.progress(int(conf))
        st.write(f"🧠 Confidence: **{conf}%** · Baseline: {p_lr:.2f} · Var: {var:.4f}")

# ---- WATER TAB ----
with tab_water:
    st.subheader("💧 Water Quality (Offline)")
    col1, col2 = st.columns(2)
    with col1:
        ph = st.number_input("pH", 0.0, 14.0, 7.2)
        turb = st.number_input("Turbidity (NTU)", 0.0, 500.0, 5.0)
        tds = st.number_input("TDS (ppm)", 0.0, 5000.0, 300.0)
    with col2:
        ec = st.number_input("EC (µS/cm)", 0.0, 10000.0, 600.0)
        wtemp = st.number_input("Water Temp (°C)", 0.0, 60.0, 25.0)
    X0w = np.array([[ph, turb, tds, ec, wtemp]])
    Xnw = inject_noise(X0w, noise_pct)

    if st.button("Run Water Analysis"):
        p_lr = water_pipe.predict_proba(Xnw)[0,1]
        subs_w = make_submodels_from(water_pipe, eps=0.04)
        p_ens, sub_probs, weights, var = ensemble_predict_proba(subs_w, Xnw)
        y_pred = int(p_ens >= 0.5)
        label = "Contaminated" if y_pred==1 else "Safe"
        st.markdown(f"**Predicted Result:** {label}")
        conf = round(p_ens * 100, 2)
        st.progress(int(conf))
        st.write(f"🧠 Confidence: **{conf}%** · Baseline: {p_lr:.2f} · Var: {var:.4f}")

# ---- QUANTUM VIEW ----
with tab_quantum:
    st.subheader("⚛ Quantum-Inspired View")
    st.write("We simulate **interference** by combining multiple submodels and smoothing disagreement.")
    st.write("- High disagreement → equalized weights (destructive interference cancels noise)")
    st.write("- Low disagreement → central estimates reinforced (constructive interference)")

# ---- REPORTS ----
with tab_report:
    st.subheader("🧾 Local Reports")
    st.write("Each analysis can be downloaded as CSV. Works fully offline.")

# ---- DASHBOARD ----
with tab_dashboard:
    st.subheader("📊 SDG Dashboard Summary")
    if "history" not in st.session_state:
        st.session_state["history"] = {"Soil": [], "Health": [], "Water": []}
    if "last_stable" not in st.session_state or not isinstance(st.session_state["last_stable"], dict):
        st.session_state["last_stable"] = {}

    last = st.session_state["last_stable"]
    mode = last.get("mode", "")
    conf_val = last.get("confidence", None)
    if mode in ["Soil", "Health", "Water"] and conf_val is not None:
        history = st.session_state["history"][mode]
        history.append(conf_val)
        if len(history) > 3:  
            history.pop(0)
        st.session_state["history"][mode] = history

    soil_conf = np.mean(st.session_state["history"]["Soil"]) if st.session_state["history"]["Soil"] else 0
    health_conf = np.mean(st.session_state["history"]["Health"]) if st.session_state["history"]["Health"] else 0
    water_conf = np.mean(st.session_state["history"]["Water"]) if st.session_state["history"]["Water"] else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("🌾 Soil Fertility Index (SDG 2)", f"{soil_conf:.1f}%")
    col2.metric("🧬 Health Reliability (SDG 3)", f"{health_conf:.1f}%")
    col3.metric("💧 Water Purity Confidence (SDG 6)", f"{water_conf:.1f}%")

    st.markdown("### 🔄 Confidence Trends (Last 3 Tests)")
    fig, ax = plt.subplots()
    ax.plot(st.session_state["history"]["Soil"], "o-", label="Soil")
    ax.plot(st.session_state["history"]["Health"], "o-", label="Health")
    ax.plot(st.session_state["history"]["Water"], "o-", label="Water")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Confidence (%)")
    ax.set_xlabel("Last 3 Tests")
    ax.legend()
    st.pyplot(fig)

    goal = 80
    st.markdown("### 🎯 Progress Toward UN SDG Goals")
    st.progress(int(min(soil_conf / goal, 1) * 100))
    st.caption(f"🌾 Soil Fertility Goal: {soil_conf:.1f}% of {goal}% target")

    st.progress(int(min(health_conf / goal, 1) * 100))
    st.caption(f"🧬 Health Reliability Goal: {health_conf:.1f}% of {goal}% target")

    st.progress(int(min(water_conf / goal, 1) * 100))
    st.caption(f"💧 Water Purity Goal: {water_conf:.1f}% of {goal}% target")

    overall = round((soil_conf + health_conf + water_conf) / 3, 1)
    st.markdown(f"### 🌍 Overall Sustainability Confidence: **{overall}%**")

    update_time = st.session_state.get("last_update_time", "No tests yet")
    st.markdown(f"🕒 **Last Update:** {update_time}")

    st.caption(
        "NoiseShield AI — Quantum-inspired, offline diagnostics supporting SDGs 2, 3, and 6 in low-resource settings."
    )
