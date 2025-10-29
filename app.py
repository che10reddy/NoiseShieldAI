import streamlit as st
import numpy as np, pandas as pd, datetime as dt
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# =========================
# App meta
# =========================
st.set_page_config(page_title="NoiseShield AI", page_icon="🔰", layout="centered")

# ---------- One-time session init ----------
if "results" not in st.session_state:
    # per-domain last result (persistent during session)
    st.session_state["results"] = {"Soil": None, "Health": None, "Water": None}

if "history" not in st.session_state:
    # rolling confidence history (for future trend plots if needed)
    st.session_state["history"] = {"Soil": [], "Health": [], "Water": []}

# =========================
# Theme (no external emojis/images)
# =========================
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"

theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"],
                                index=0 if st.session_state["theme_mode"] == "Light" else 1)
st.session_state["theme_mode"] = theme_choice

is_dark = st.session_state["theme_mode"] == "Dark"
bg_color   = "#0E1117" if is_dark else "#FFFFFF"
text_color = "#FAFAFA" if is_dark else "#111111"
accent     = "#00B4B4" if is_dark else "#0A84FF"
banner_fg  = accent

st.markdown(f"""
<style>
  body, .stApp {{
    background-color: {bg_color};
    color: {text_color};
  }}
  .stButton>button {{
    background-color: {accent};
    color: white;
    border-radius: 8px;
    font-weight: 600;
    border: 0;
  }}
  .stProgress > div > div {{
    background-color: {accent} !important;
  }}
  .block-container {{
    padding-top: 1.2rem; max-width: 980px;
  }}
  h2, h3, h4, h5, h6 {{ color: {banner_fg}; }}
  /* Make labels readable in light mode */
  label, .stTextInput label, .stNumberInput label {{
    color: {'#EDEDED' if is_dark else '#1F1F1F'} !important;
    font-weight: 600;
  }}
  /* SDG badges (pure CSS, offline safe) */
  .sdg-badges {{
    display:flex; gap:8px; justify-content:center; margin:8px 0 0 0;
  }}
  .sdg-badge {{
    padding:6px 10px; border-radius:8px; font-weight:700; font-size:12px;
    background: {accent}22; color: {banner_fg}; border: 1px solid {banner_fg}44;
  }}
  * {{ transition: background-color .25s ease, color .25s ease; }}
</style>
""", unsafe_allow_html=True)

# ---------- Banner (offline safe) ----------
st.markdown(f"""
<div style="text-align:center; margin-bottom:14px">
  <h2 style="font-weight:800; margin:0">NoiseShield AI · Quantum-Inspired Diagnostics</h2>
  <div class="sdg-badges">
    <div class="sdg-badge">SDG 2 · Zero Hunger</div>
    <div class="sdg-badge">SDG 3 · Good Health</div>
    <div class="sdg-badge">SDG 6 · Clean Water</div>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Core helpers
# =========================
def seed_rng(seed=42): return np.random.default_rng(seed)

def inject_noise(X, pct, rng=None):
    """Multiply inputs by 1 + Normal(0, pct/100)."""
    if pct <= 0: return X.copy()
    if rng is None: rng = seed_rng(123)
    scale = pct / 100.0
    return X * (1.0 + rng.normal(0, scale, size=X.shape))

def make_submodels_from(base_lr, eps=0.03, n=3):
    """Build n perturbed copies of a trained LR pipeline (quantum-inspired ensemble)."""
    scaler = base_lr.named_steps['standardscaler']
    lr = base_lr.named_steps['logisticregression']
    subs = []
    for k in range(n):
        lr_k = LogisticRegression()
        lr_k.classes_ = lr.classes_
        lr_k.coef_ = lr.coef_ * (1 + (k-1)*eps)       # -eps, 0, +eps
        lr_k.intercept_ = lr.intercept_ * (1 + (k-1)*eps)
        pipe_k = make_pipeline(StandardScaler(with_mean=scaler.with_mean, with_std=scaler.with_std))
        # Dummy fit to initialize scaler attrs
        pipe_k.fit(np.zeros((1, scaler.scale_.shape[0])), [0])
        pipe_k.named_steps['standardscaler'].mean_ = scaler.mean_.copy()
        pipe_k.named_steps['standardscaler'].scale_ = scaler.scale_.copy()
        pipe_k.steps.append(('logisticregression', lr_k))
        subs.append(pipe_k)
    return subs

def ensemble_predict_proba(submodels, X_row):
    """Combine submodels with adaptive weights -> interference-inspired smoothing."""
    probs = np.array([m.predict_proba(X_row)[0, 1] for m in submodels])  # shape (n,)
    var = float(np.var(probs))
    if var > 0.02:
        w = np.ones_like(probs) / len(probs)  # destructive interference -> equalize
    else:
        centered = np.exp(-(probs - probs.mean())**2 / (2*0.0025))
        w = centered / centered.sum()         # constructive -> reinforce central estimate
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

def result_to_row(domain, features, x_row, label, p_ens, conf, noise_pct):
    now = dt.datetime.now().isoformat()
    return {
        "Domain": domain,
        **{features[i]: float(x_row.ravel()[i]) for i in range(len(features))},
        "Prediction": label,
        "Probability": float(p_ens),
        "Confidence": float(conf),
        "NoisePct": int(noise_pct),
        "Timestamp": now
    }

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    sio = StringIO(); df.to_csv(sio, index=False); return sio.getvalue().encode()

# =========================
# Synthetic “training” for offline demo
# =========================
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
    water_pipe  = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    health_pipe.fit(Xh, yh)
    water_pipe.fit(Xw, yw)
    return health_pipe, water_pipe

health_pipe, water_pipe = train_baselines()

# =========================
# UI scaffolding
# =========================
st.markdown("### App Controls")
lang = st.sidebar.selectbox("Language", ["English (EN)"])
noise_pct = st.sidebar.slider("Simulated Sensor Noise (%)", 0, 100, 0, step=5)

tabs = st.tabs([
    "Soil (SDG 2)", "Health (SDG 3)", "Water (SDG 6)",
    "Quantum View", "Reports", "SDG Dashboard"
])

# =========================
# SOIL TAB (synthetic baseline model created inline)
# =========================
with tabs[0]:
    st.subheader("Soil Fertility Analysis (Offline)")
    c1, c2 = st.columns(2)
    with c1:
        spH  = st.number_input("Soil pH", 3.0, 10.0, 6.5)
        N    = st.number_input("Nitrogen (mg/kg)", 0.0, 200.0, 40.0)
        P    = st.number_input("Phosphorus (mg/kg)", 0.0, 200.0, 30.0)
    with c2:
        K    = st.number_input("Potassium (mg/kg)", 0.0, 300.0, 120.0)
        moist= st.number_input("Soil Moisture (%)", 0.0, 100.0, 25.0)

    X0s = np.array([[spH, N, P, K, moist]])
    Xns = inject_noise(X0s, noise_pct)

    @st.cache_resource
    def train_soil_baseline():
        rng = seed_rng(3); n = 400
        pH = rng.normal(6.5, 0.8, n).clip(3.5, 9.5)
        N  = rng.normal(50, 25, n).clip(0, 200)
        P  = rng.normal(40, 20, n).clip(0, 200)
        K  = rng.normal(150, 60, n).clip(0, 300)
        M  = rng.normal(30, 15, n).clip(0, 100)
        X  = np.column_stack([pH, N, P, K, M])
        y  = ((N < 30) | (P < 20) | (K < 80) | (pH < 5.5) | (pH > 8.5)).astype(int)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        pipe.fit(X, y)
        return pipe

    soil_pipe = train_soil_baseline()

    if st.button("Run Soil Analysis"):
        p_lr = soil_pipe.predict_proba(Xns)[0,1]
        subs_s = make_submodels_from(soil_pipe, eps=0.04)
        p_ens, sub_probs, weights, var = ensemble_predict_proba(subs_s, Xns)
        y_pred = int(p_ens >= 0.5)
        label  = "Nutrient Deficient" if y_pred==1 else "Fertile"
        conf   = round(p_ens * 100, 2)

        st.markdown(f"**Predicted Result:** {label}")
        st.progress(int(conf))
        st.write(f"Confidence: **{conf}%** · Baseline prob: {p_lr:.2f} · Disagreement var: {var:.4f}")

        # Interference viz
        fig, ax = plt.subplots()
        ax.bar([f"Sub{i+1}" for i in range(len(sub_probs))] + ["Final"], list(sub_probs)+[p_ens])
        ax.set_ylim(0,1); ax.set_ylabel("Probability")
        st.pyplot(fig)

        # Save last result (persistent)
        st.session_state["results"]["Soil"] = {
            "confidence": float(conf),
            "prob": float(p_ens),
            "label": label,
            "time": dt.datetime.now().isoformat()
        }
        # Optional: keep last 3 confidences for trend
        hist = st.session_state["history"]["Soil"]
        hist.append(conf); st.session_state["history"]["Soil"] = hist[-3:]

        # CSV (single row) for this run
        row = result_to_row(
            "Soil", ["pH","Nitrogen","Phosphorus","Potassium","Moisture"],
            X0s, label, p_ens, conf, noise_pct
        )
        soil_row_df = pd.DataFrame([row])
        st.download_button("Download Soil Result (CSV)",
                           data=df_to_csv_bytes(soil_row_df),
                           file_name="noiseshield_soil_result.csv",
                           mime="text/csv")

# =========================
# HEALTH TAB
# =========================
with tabs[1]:
    st.subheader("Health Diagnostics (Offline)")
    c1, c2 = st.columns(2)
    with c1:
        hb    = st.number_input("Hemoglobin (g/dL)", 0.0, 25.0, 12.5)
        wbc   = st.number_input("WBC (cells/µL)", 0.0, 30000.0, 7000.0)
        pltlt = st.number_input("Platelets (cells/µL)", 0.0, 900000.0, 250000.0)
    with c2:
        temp  = st.number_input("Body Temperature (°C)", 30.0, 45.0, 36.8)
        pulse = st.number_input("Pulse Rate (bpm)", 30.0, 200.0, 80.0)

    X0 = np.array([[hb, wbc, pltlt, temp, pulse]])
    Xn = inject_noise(X0, noise_pct)

    if st.button("Run Health Analysis"):
        p_lr = health_pipe.predict_proba(Xn)[0,1]
        subs_h = make_submodels_from(health_pipe, eps=0.04)
        p_ens, sub_probs, weights, var = ensemble_predict_proba(subs_h, Xn)
        y_pred = int(p_ens >= 0.5)
        label  = "Possible Condition" if y_pred==1 else "Healthy"
        conf   = round(p_ens * 100, 2)

        st.markdown(f"**Predicted Result:** {label}")
        st.progress(int(conf))
        st.write(f"Confidence: **{conf}%** · Baseline prob: {p_lr:.2f} · Disagreement var: {var:.4f}")

        # Interference viz
        fig, ax = plt.subplots()
        ax.bar([f"Sub{i+1}" for i in range(len(sub_probs))] + ["Final"], list(sub_probs)+[p_ens])
        ax.set_ylim(0,1); ax.set_ylabel("Probability")
        st.pyplot(fig)

        st.session_state["results"]["Health"] = {
            "confidence": float(conf),
            "prob": float(p_ens),
            "label": label,
            "time": dt.datetime.now().isoformat()
        }
        hist = st.session_state["history"]["Health"]
        hist.append(conf); st.session_state["history"]["Health"] = hist[-3:]

        row = result_to_row(
            "Health", ["Hemoglobin","WBC","Platelets","Temp","Pulse"],
            X0, label, p_ens, conf, noise_pct
        )
        health_row_df = pd.DataFrame([row])
        st.download_button("Download Health Result (CSV)",
                           data=df_to_csv_bytes(health_row_df),
                           file_name="noiseshield_health_result.csv",
                           mime="text/csv")

# =========================
# WATER TAB
# =========================
with tabs[2]:
    st.subheader("Water Quality (Offline)")
    c1, c2 = st.columns(2)
    with c1:
        ph   = st.number_input("pH", 0.0, 14.0, 7.2)
        turb = st.number_input("Turbidity (NTU)", 0.0, 500.0, 5.0)
        tds  = st.number_input("TDS (ppm)", 0.0, 5000.0, 300.0)
    with c2:
        ec   = st.number_input("EC (µS/cm)", 0.0, 10000.0, 600.0)
        wtemp= st.number_input("Water Temp (°C)", 0.0, 60.0, 25.0)

    X0w = np.array([[ph, turb, tds, ec, wtemp]])
    Xnw = inject_noise(X0w, noise_pct)

    if st.button("Run Water Analysis"):
        p_lr = water_pipe.predict_proba(Xnw)[0,1]
        subs_w = make_submodels_from(water_pipe, eps=0.04)
        p_ens, sub_probs, weights, var = ensemble_predict_proba(subs_w, Xnw)
        y_pred = int(p_ens >= 0.5)
        label  = "Contaminated" if y_pred==1 else "Safe"
        conf   = round(p_ens * 100, 2)

        st.markdown(f"**Predicted Result:** {label}")
        st.progress(int(conf))
        st.write(f"Confidence: **{conf}%** · Baseline prob: {p_lr:.2f} · Disagreement var: {var:.4f}")

        # Interference viz
        fig, ax = plt.subplots()
        ax.bar([f"Sub{i+1}" for i in range(len(sub_probs))] + ["Final"], list(sub_probs)+[p_ens])
        ax.set_ylim(0,1); ax.set_ylabel("Probability")
        st.pyplot(fig)

        st.session_state["results"]["Water"] = {
            "confidence": float(conf),
            "prob": float(p_ens),
            "label": label,
            "time": dt.datetime.now().isoformat()
        }
        hist = st.session_state["history"]["Water"]
        hist.append(conf); st.session_state["history"]["Water"] = hist[-3:]

        row = result_to_row(
            "Water", ["pH","Turbidity","TDS","EC","WaterTemp"],
            X0w, label, p_ens, conf, noise_pct
        )
        water_row_df = pd.DataFrame([row])
        st.download_button("Download Water Result (CSV)",
                           data=df_to_csv_bytes(water_row_df),
                           file_name="noiseshield_water_result.csv",
                           mime="text/csv")

# =========================
# QUANTUM VIEW
# =========================
with tabs[3]:
    st.subheader("Quantum-Inspired View")
    st.write("This app simulates **interference** by combining several lightweight submodels (perturbed logistic regressions).")
    st.write("- High disagreement → weights equalize (analogous to **destructive interference** cancelling noise)")
    st.write("- Low disagreement → central estimate reinforced (analogous to **constructive interference**)")

# =========================
# REPORTS (download all summary)
# =========================
with tabs[4]:
    st.subheader("Local Reports (Offline)")
    res = st.session_state["results"]
    rows = []
    for dom in ["Soil","Health","Water"]:
        r = res.get(dom)
        if r is not None:
            rows.append({"Domain": dom, "Prediction": r["label"],
                         "Probability": r["prob"], "Confidence": r["confidence"],
                         "Timestamp": r["time"]})
        else:
            rows.append({"Domain": dom, "Prediction": "—",
                         "Probability": "—", "Confidence": 0, "Timestamp": "—"})
    df_all = pd.DataFrame(rows)
    st.dataframe(df_all, use_container_width=True)
    st.download_button("⬇️ Download Summary (CSV)",
                       data=df_to_csv_bytes(df_all),
                       file_name="noiseshield_summary.csv",
                       mime="text/csv")

# =========================
# DASHBOARD (always visible & stable)
# =========================
with tabs[5]:
    st.subheader("SDG Dashboard Summary")
    r = st.session_state["results"]
    soil_conf   = (r["Soil"] or {}).get("confidence", 0.0)
    health_conf = (r["Health"] or {}).get("confidence", 0.0)
    water_conf  = (r["Water"] or {}).get("confidence", 0.0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Soil Fertility Index (SDG 2)", f"{float(soil_conf):.1f}%")
    col2.metric("Health Reliability (SDG 3)",   f"{float(health_conf):.1f}%")
    col3.metric("Water Purity Confidence (SDG 6)", f"{float(water_conf):.1f}%")

    overall = round((float(soil_conf) + float(health_conf) + float(water_conf)) / 3, 1)
    status = "🟢 Excellent" if overall >= 80 else ("🟡 Moderate" if overall >= 50 else "🔴 Needs Work")
    st.markdown(f"### Overall Sustainability Confidence: **{overall}%** · {status}")

    # Optional mini trend (uses last 3 confidences per domain)
    with st.expander("Show Confidence Trends (last 3 per domain)"):
        fig, ax = plt.subplots()
        ax.plot(st.session_state["history"]["Soil"],   "o-", label="Soil")
        ax.plot(st.session_state["history"]["Health"], "o-", label="Health")
        ax.plot(st.session_state["history"]["Water"],  "o-", label="Water")
        ax.set_ylim(0, 100); ax.set_ylabel("Confidence (%)"); ax.set_xlabel("Recent Tests")
        ax.legend(); st.pyplot(fig)

    st.caption("NoiseShield AI — a quantum-inspired, offline tool for soil, health, and water diagnostics in low-resource settings.")
