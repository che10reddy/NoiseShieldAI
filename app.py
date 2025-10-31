import streamlit as st
import numpy as np, pandas as pd, datetime as dt
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Optional PDF report (kept very lightweight & offline)
try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

# =========================
# App meta
# =========================
st.set_page_config(page_title="NoiseShield AI", page_icon="🔰", layout="centered")

# ---------- Session init ----------
if "results" not in st.session_state:
    # per-domain last result
    st.session_state["results"] = {"Soil": None, "Health": None, "Water": None}
if "history" not in st.session_state:
    # last 3 confidence values per domain (for trends)
    st.session_state["history"] = {"Soil": [], "Health": [], "Water": []}
if "last_stable" not in st.session_state:
    # last known safe reading per domain when confidence was decent
    st.session_state["last_stable"] = {"Soil": None, "Health": None, "Water": None}

# =========================
# Theme
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
panel_bg   = "#151922" if is_dark else "#F7F9FC"

st.markdown(f"""
<style>
  body, .stApp {{ background-color: {bg_color}; color: {text_color}; }}
  .stButton>button {{ background-color: {accent}; color: white; border-radius: 8px; font-weight: 600; border: 0; }}
  .stProgress > div > div {{ background-color: {accent} !important; }}
  .block-container {{ padding-top: 1.0rem; max-width: 980px; }}
  h2, h3, h4, h5, h6 {{ color: {banner_fg}; }}
  label, .stTextInput label, .stNumberInput label {{
    color: {'#EDEDED' if is_dark else '#1F1F1F'} !important; font-weight: 600;
  }}
  .sdg-badges {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:center; margin:6px 0 0 0; }}
  .sdg-badge {{
    padding:6px 10px; border-radius:8px; font-weight:700; font-size:12px;
    background: {accent}22; color: {banner_fg}; border: 1px solid {banner_fg}44;
  }}
  .panel {{
    background:{panel_bg}; padding:12px 14px; border-radius:10px; border:1px solid {banner_fg}30;
  }}
  * {{ transition: background-color .25s ease, color .25s ease; }}
</style>
""", unsafe_allow_html=True)

# =========================
# Languages (EN, Telugu, Hindi, Tamil, Bengali, Marathi)
# =========================
LANG = {
    "English": {
        "title": "NoiseShield AI · Quantum-Inspired Diagnostics",
        "sdg2": "SDG 2 · Zero Hunger",
        "sdg3": "SDG 3 · Good Health",
        "sdg6": "SDG 6 · Clean Water",
        "tabs": ["Soil (SDG 2)", "Health (SDG 3)", "Water (SDG 6)",
                 "Quantum View", "Reports", "SDG Dashboard", "Cross-Domain Stability"],
        "controls": "App Controls",
        "noise": "Simulated Sensor Noise (%)",
        "soil": "Soil Fertility Analysis (Offline)",
        "health": "Health Diagnostics (Offline)",
        "water": "Water Quality (Offline)",
        "quantum": "Quantum-Inspired View",
        "reports": "Local Reports (Offline)",
        "dashboard": "SDG Dashboard Summary",
        "stability": "Cross-Domain Stability (Noise vs Probability)",
        "predicted": "Predicted Result",
        "confidence": "Confidence",
        "baseline": "Baseline prob",
        "var": "Disagreement var",
        "download_soil": "Download Soil Result (CSV)",
        "download_health": "Download Health Result (CSV)",
        "download_water": "Download Water Result (CSV)",
        "download_summary": "Download Summary (CSV)",
        "download_pdf": "Download PDF Report",
        "pdf_missing": "Install 'fpdf' to enable PDF report (pip install fpdf).",
        "why": "Why this result?",
        "last_safe": "Last Known Safe Reading",
        "unstable": "Data unstable — showing last safe reading",
        "domain": "Domain",
        "prediction": "Prediction",
        "prob": "Probability",
        "conf": "Confidence",
        "time": "Timestamp",
        "overall": "Overall Sustainability Confidence",
        "excellent": "🟢 Excellent",
        "moderate": "🟡 Moderate",
        "needs": "🔴 Needs Work",
        "trend": "Confidence Trends (last 3 per domain)",
        "caption": "Quantum-inspired, offline tool for soil, health, and water diagnostics in low-resource settings.",
        "health_label_pos": "Possible Condition",
        "health_label_neg": "Healthy",
        "soil_label_pos": "Nutrient Deficient",
        "soil_label_neg": "Fertile",
        "water_label_pos": "Contaminated",
        "water_label_neg": "Safe",
        "inputs": "Inputs",
        "noise_level": "Noise Level",
        "mode": "Mode",
        "note_panel": "Designed for YCS competition & college admissions: explainable, robust, and offline."
    },
    "తెలుగు": {  # Telugu
        "title": "నాయిస్‌షీల్డ్ AI · క్వాంటమ్ ప్రేరిత నిర్ధారణలు",
        "sdg2": "SDG 2 · ఆకలి నిర్మూలన",
        "sdg3": "SDG 3 · ఆరోగ్యము",
        "sdg6": "SDG 6 · శుభ్రమైన నీరు",
        "tabs": ["మట్టి (SDG 2)", "ఆరోగ్యం (SDG 3)", "నీరు (SDG 6)",
                 "క్వాంటమ్ వీయూ", "రిపోర్ట్స్", "SDG డ్యాష్‌బోర్డ్", "క్రాస్-డొమైన్ స్థిరత్వం"],
        "controls": "యాప్ కంట్రోల్స్",
        "noise": "సెన్సార్ శబ్దం (%)",
        "soil": "మట్టి సారవంతత విశ్లేషణ (ఆఫ్‌లైన్)",
        "health": "ఆరోగ్య నిర్ధారణ (ఆఫ్‌లైన్)",
        "water": "నీటి నాణ్యత (ఆఫ్‌లైన్)",
        "quantum": "క్వాంటమ్ ప్రేరణ వీయూ",
        "reports": "లోకల్ రిపోర్ట్స్ (ఆఫ్‌లైన్)",
        "dashboard": "SDG డ్యాష్‌బోర్డ్ సమ్మరీ",
        "stability": "క్రాస్-డొమైన్ స్థిరత్వం (నాయిస్ vs Probability)",
        "predicted": "అంచనా ఫలితం",
        "confidence": "నమ్మకం",
        "baseline": "బేస్‌లైన్ అవకాశం",
        "var": "విభేదం variance",
        "download_soil": "మట్టి ఫలితం (CSV)",
        "download_health": "ఆరోగ్య ఫలితం (CSV)",
        "download_water": "నీటి ఫలితం (CSV)",
        "download_summary": "సమ్మరీ (CSV)",
        "download_pdf": "PDF రిపోర్ట్",
        "pdf_missing": "'fpdf' ఇన్‌స్టాల్ చేయండి (pip install fpdf).",
        "why": "ఈ ఫలితానికి కారణం?",
        "last_safe": "గత సురక్షిత రీడింగ్",
        "unstable": "డేటా స్థిరంగా లేదు — చివరి సురక్షిత రీడింగ్ చూపింపు",
        "domain": "డొమైన్",
        "prediction": "ఫలితం",
        "prob": "సంభావ్యత",
        "conf": "నమ్మకం",
        "time": "సమయం",
        "overall": "సమగ్ర సుస్థిరత నమ్మకం",
        "excellent": "🟢 అద్భుతం",
        "moderate": "🟡 సరాసరి",
        "needs": "🔴 మెరుగులు అవసరం",
        "trend": "నమ్మకం ట్రెండ్స్ (చివరి 3)",
        "caption": "క్వాంటమ్ ప్రేరణతో, ఆఫ్‌లైన్ టూల్ — తక్కువ వనరుల ప్రాంతాలకు.",
        "health_label_pos": "సాధ్యమైన పరిస్థితి",
        "health_label_neg": "ఆరోగ్యంగా ఉంది",
        "soil_label_pos": "పోషక లోపం",
        "soil_label_neg": "సారవంతమైనది",
        "water_label_pos": "కాలుష్యం",
        "water_label_neg": "సురక్షితం",
        "inputs": "ఇన్పుట్స్",
        "noise_level": "నాయిస్ స్థాయి",
        "mode": "మోడ్",
        "note_panel": "YCS & కాలేజ్ అడ్మిషన్స్ కోసం రూపొందించబడింది: explainable, robust, offline."
    },
    "हिंदी": {  # Hindi
        "title": "NoiseShield AI · क्वांटम-प्रेरित निदान",
        "sdg2": "SDG 2 · भुखमरी मुक्त",
        "sdg3": "SDG 3 · उत्तम स्वास्थ्य",
        "sdg6": "SDG 6 · स्वच्छ पानी",
        "tabs": ["मिट्टी (SDG 2)", "स्वास्थ्य (SDG 3)", "जल (SDG 6)",
                 "क्वांटम दृश्य", "रिपोर्ट्स", "SDG डैशबोर्ड", "क्रॉस-डोमेन स्थिरता"],
        "controls": "एप कंट्रोल्स",
        "noise": "सेंसर शोर (%)",
        "soil": "मिट्टी उर्वरता विश्लेषण (ऑफलाइन)",
        "health": "स्वास्थ्य निदान (ऑफलाइन)",
        "water": "जल गुणवत्ता (ऑफलाइन)",
        "quantum": "क्वांटम-प्रेरित दृश्य",
        "reports": "स्थानीय रिपोर्ट्स (ऑफलाइन)",
        "dashboard": "SDG डैशबोर्ड सार",
        "stability": "क्रॉस-डोमेन स्थिरता (Noise vs Probability)",
        "predicted": "अनुमानित परिणाम",
        "confidence": "विश्वास",
        "baseline": "बेसलाइन प्रायिकता",
        "var": "असहमति variance",
        "download_soil": "मिट्टी परिणाम (CSV)",
        "download_health": "स्वास्थ्य परिणाम (CSV)",
        "download_water": "जल परिणाम (CSV)",
        "download_summary": "सार (CSV)",
        "download_pdf": "PDF रिपोर्ट",
        "pdf_missing": "'fpdf' इंस्टॉल करें (pip install fpdf).",
        "why": "यह परिणाम क्यों?",
        "last_safe": "अंतिम सुरक्षित रीडिंग",
        "unstable": "डेटा अस्थिर — अंतिम सुरक्षित रीडिंग दिखा रहे हैं",
        "domain": "डोमेन",
        "prediction": "परिणाम",
        "prob": "प्रायिकता",
        "conf": "विश्वास",
        "time": "समय",
        "overall": "समग्र स्थिरता विश्वास",
        "excellent": "🟢 उत्कृष्ट",
        "moderate": "🟡 मध्यम",
        "needs": "🔴 सुधार आवश्यक",
        "trend": "विश्वास ट्रेंड (पिछले 3)",
        "caption": "क्वांटम-प्रेरित, ऑफलाइन टूल — कम संसाधन सेटिंग्स के लिए।",
        "health_label_pos": "संभावित स्थिति",
        "health_label_neg": "स्वस्थ",
        "soil_label_pos": "पोषक कमी",
        "soil_label_neg": "उर्वर",
        "water_label_pos": "दूषित",
        "water_label_neg": "सुरक्षित",
        "inputs": "इनपुट्स",
        "noise_level": "शोर स्तर",
        "mode": "मोड",
        "note_panel": "YCS और कॉलेज एडमिशन हेतु: explainable, robust, offline."
    },
    "தமிழ்": {  # Tamil
        "title": "NoiseShield AI · குவாண்டம் ஊக்கமூட்டிய நாடுகாணல்",
        "sdg2": "SDG 2 · பசி ஒழிப்பு",
        "sdg3": "SDG 3 · நல்ல ஆரோக்கியம்",
        "sdg6": "SDG 6 · தூய்மையான நீர்",
        "tabs": ["மண் (SDG 2)", "ஆரோக்கியம் (SDG 3)", "நீர் (SDG 6)",
                 "குவாண்டம் காட்சி", "அறிக்கைகள்", "SDG டாஷ்போர்டு", "குறுக்கு-விளைநிலை நிலைத்தன்மை"],
        "controls": "அப் கட்டுப்பாடுகள்",
        "noise": "சென்சார் சத்தம் (%)",
        "soil": "மண் வளம் பகுப்பாய்வு (ஆஃப்லைன்)",
        "health": "நோய் கண்டறிதல் (ஆஃப்லைன்)",
        "water": "நீர் தரம் (ஆஃப்லைன்)",
        "quantum": "குவாண்டம்-ஊக்க காட்சி",
        "reports": "உள்ளூர் அறிக்கைகள் (ஆஃப்லைன்)",
        "dashboard": "SDG டாஷ்போர்டு சுருக்கம்",
        "stability": "குறுக்கு-விளைநிலை நிலைத்தன்மை (சத்தம் vs Probability)",
        "predicted": "முடிவு",
        "confidence": "நம்பிக்கை",
        "baseline": "அடிப்படை சாத்தியம்",
        "var": "வேறுபாடு variance",
        "download_soil": "மண் முடிவு (CSV)",
        "download_health": "ஆரோக்கிய முடிவு (CSV)",
        "download_water": "நீர் முடிவு (CSV)",
        "download_summary": "சுருக்கம் (CSV)",
        "download_pdf": "PDF அறிக்கை",
        "pdf_missing": "'fpdf' நிறுவவும் (pip install fpdf).",
        "why": "ஏன் இந்த முடிவு?",
        "last_safe": "கடைசி பாதுகாப்பான ரீடிங்",
        "unstable": "தரவு நிலைகுலைவு — கடைசிப் பாதுகாப்பான ரீடிங் காட்டப்படுகிறது",
        "domain": "துறை",
        "prediction": "முடிவு",
        "prob": "சாத்தியம்",
        "conf": "நம்பிக்கை",
        "time": "நேரம்",
        "overall": "மொத்த நிலைத்தன்மை நம்பிக்கை",
        "excellent": "🟢 சிறப்பு",
        "moderate": "🟡 நடுத்தரம்",
        "needs": "🔴 மேம்பாடு தேவை",
        "trend": "நம்பிக்கை போக்கு (கடைசி 3)",
        "caption": "குவாண்டம் ஊக்கமூட்டிய, ஆஃப்லைன் கருவி — குறைந்த வள பகுதிகளுக்கு.",
        "health_label_pos": "சாத்தியமான நிலை",
        "health_label_neg": "ஆரோக்கியம்",
        "soil_label_pos": "ஊட்டச்சத்து பற்றாக்குறை",
        "soil_label_neg": "வளமானது",
        "water_label_pos": "கழிவு",
        "water_label_neg": "பாதுகாப்பானது",
        "inputs": "உள்ளீடுகள்",
        "noise_level": "சத்தம்",
        "mode": "முறை",
        "note_panel": "YCS & கல்லூரி சேர்க்கை நோக்கி: explainable, robust, offline."
    },
    "বাংলা": {  # Bengali
        "title": "NoiseShield AI · কোয়ান্টাম-প্রাণিত ডায়াগনস্টিক",
        "sdg2": "SDG 2 · ক্ষুধামুক্ত",
        "sdg3": "SDG 3 · সুস্বাস্থ্য",
        "sdg6": "SDG 6 · বিশুদ্ধ পানি",
        "tabs": ["মাটি (SDG 2)", "স্বাস্থ্য (SDG 3)", "পানীয় জল (SDG 6)",
                 "কোয়ান্টাম ভিউ", "রিপোর্ট", "SDG ড্যাশবোর্ড", "ক্রস-ডোমেইন স্থায়িত্ব"],
        "controls": "অ্যাপ কন্ট্রোল",
        "noise": "সেন্সর নয়েজ (%)",
        "soil": "মাটির উর্বরতা বিশ্লেষণ (অফলাইন)",
        "health": "স্বাস্থ্য নির্ণয় (অফলাইন)",
        "water": "জলের গুণমান (অফলাইন)",
        "quantum": "কোয়ান্টাম-প্রাণিত ভিউ",
        "reports": "লোকাল রিপোর্ট (অফলাইন)",
        "dashboard": "SDG ড্যাশবোর্ড সারসংক্ষেপ",
        "stability": "ক্রস-ডোমেইন স্থায়িত্ব (Noise vs Probability)",
        "predicted": "অনুমেয় ফলাফল",
        "confidence": "আস্থা",
        "baseline": "বেসলাইন সম্ভাব্যতা",
        "var": "বিভেদ variance",
        "download_soil": "মাটি ফলাফল (CSV)",
        "download_health": "স্বাস্থ্য ফলাফল (CSV)",
        "download_water": "জল ফলাফল (CSV)",
        "download_summary": "সারাংশ (CSV)",
        "download_pdf": "PDF রিপোর্ট",
        "pdf_missing": "'fpdf' ইনস্টল করুন (pip install fpdf).",
        "why": "এই ফলাফল কেন?",
        "last_safe": "সর্বশেষ নিরাপদ রিডিং",
        "unstable": "ডেটা অস্থির — সর্বশেষ নিরাপদ রিডিং দেখানো হচ্ছে",
        "domain": "ডোমেইন",
        "prediction": "ফলাফল",
        "prob": "সম্ভাব্যতা",
        "conf": "আস্থা",
        "time": "সময়",
        "overall": "সমগ্র স্থায়িত্বের আস্থা",
        "excellent": "🟢 চমৎকার",
        "moderate": "🟡 মাঝামাঝি",
        "needs": "🔴 উন্নতি প্রয়োজন",
        "trend": "আস্থা প্রবণতা (শেষ 3)",
        "caption": "কোয়ান্টাম-প্রাণিত, অফলাইন টুল — স্বল্প সম্পদ সেটিংসে।",
        "health_label_pos": "সম্ভাব্য অবস্থা",
        "health_label_neg": "সুস্থ",
        "soil_label_pos": "পুষ্টি ঘাটতি",
        "soil_label_neg": "উর্বর",
        "water_label_pos": "দূষিত",
        "water_label_neg": "নিরাপদ",
        "inputs": "ইনপুট",
        "noise_level": "নয়েজ স্তর",
        "mode": "মোড",
        "note_panel": "YCS ও কলেজ অ্যাডমিশনের জন্য: explainable, robust, offline."
    },
    "मराठी": {  # Marathi
        "title": "NoiseShield AI · क्वांटम-प्रेरित निदान",
        "sdg2": "SDG 2 · उपासमार निर्मूलन",
        "sdg3": "SDG 3 · चांगले आरोग्य",
        "sdg6": "SDG 6 · स्वच्छ पाणी",
        "tabs": ["माती (SDG 2)", "आरोग्य (SDG 3)", "पाणी (SDG 6)",
                 "क्वांटम दृश्य", "अहवाल", "SDG डॅशबोर्ड", "क्रॉस-डोमेन स्थैर्य"],
        "controls": "अ‍ॅप नियंत्रण",
        "noise": "सेन्सर नॉईज (%)",
        "soil": "माती सुपीकता विश्लेषण (ऑफलाइन)",
        "health": "आरोग्य निदान (ऑफलाइन)",
        "water": "पाणी गुणवत्ता (ऑफलाइन)",
        "quantum": "क्वांटम-प्रेरित दृश्य",
        "reports": "स्थानिक अहवाल (ऑफलाइन)",
        "dashboard": "SDG डॅशबोर्ड सारांश",
        "stability": "क्रॉस-डोमेन स्थैर्य (Noise vs Probability)",
        "predicted": "भाकीत परिणाम",
        "confidence": "विश्वास",
        "baseline": "बेसलाइन प्रॉबॅबिलिटी",
        "var": "मतभेद variance",
        "download_soil": "माती निकाल (CSV)",
        "download_health": "आरोग्य निकाल (CSV)",
        "download_water": "पाणी निकाल (CSV)",
        "download_summary": "सारांश (CSV)",
        "download_pdf": "PDF अहवाल",
        "pdf_missing": "'fpdf' इन्स्टॉल करा (pip install fpdf).",
        "why": "हा परिणाम का?",
        "last_safe": "शेवटचे सुरक्षित रीडिंग",
        "unstable": "डेटा अस्थिर — शेवटचे सुरक्षित रीडिंग दाखवले",
        "domain": "डोमेन",
        "prediction": "निकाल",
        "prob": "प्रॉबॅबिलिटी",
        "conf": "विश्वास",
        "time": "वेळ",
        "overall": "एकूण स्थैर्य विश्वास",
        "excellent": "🟢 उत्कृष्ट",
        "moderate": "🟡 मध्यम",
        "needs": "🔴 सुधारणा आवश्यक",
        "trend": "विश्वास ट्रेंड (शेवटचे 3)",
        "caption": "क्वांटम-प्रेरित, ऑफलाइन साधन — कमी संसाधन भागांसाठी.",
        "health_label_pos": "संभाव्य स्थिती",
        "health_label_neg": "निरोगी",
        "soil_label_pos": "पोषण कमी",
        "soil_label_neg": "सुपीक",
        "water_label_pos": "दूषित",
        "water_label_neg": "सुरक्षित",
        "inputs": "इनपुट्स",
        "noise_level": "नॉईज",
        "mode": "मोड",
        "note_panel": "YCS आणि कॉलेज अ‍ॅडमिशनसाठी: explainable, robust, offline."
    },
}

ui_lang = st.sidebar.selectbox("Language", list(LANG.keys()), index=0)
L = LANG[ui_lang]

# =========================
# Banner
# =========================
st.markdown(f"""
<div style="text-align:center; margin-bottom:10px">
  <h2 style="font-weight:800; margin:0">{L['title']}</h2>
  <div class="sdg-badges">
    <div class="sdg-badge">{L['sdg2']}</div>
    <div class="sdg-badge">{L['sdg3']}</div>
    <div class="sdg-badge">{L['sdg6']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def seed_rng(seed=42): return np.random.default_rng(seed)

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
        lr_k.coef_ = lr.coef_ * (1 + (k-1)*eps)         # -eps, 0, +eps
        lr_k.intercept_ = lr.intercept_ * (1 + (k-1)*eps)
        pipe_k = make_pipeline(StandardScaler(with_mean=scaler.with_mean, with_std=scaler.with_std))
        pipe_k.fit(np.zeros((1, scaler.scale_.shape[0])), [0])  # init scaler attrs
        pipe_k.named_steps['standardscaler'].mean_  = scaler.mean_.copy()
        pipe_k.named_steps['standardscaler'].scale_ = scaler.scale_.copy()
        pipe_k.steps.append(('logisticregression', lr_k))
        subs.append(pipe_k)
    return subs

def ensemble_predict_proba(submodels, X_row):
    probs = np.array([m.predict_proba(X_row)[0,1] for m in submodels])
    var = float(np.var(probs))
    if var > 0.02:
        w = np.ones_like(probs) / len(probs)       # destructive-like equalization
    else:
        centered = np.exp(-(probs - probs.mean())**2 / (2*0.0025))
        w = centered / centered.sum()              # constructive-like reinforcement
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

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    sio = StringIO()
    df.to_csv(sio, index=False)
    return sio.getvalue().encode()

def pdf_report_bytes(domain, inputs, label, conf, noise):
    if not HAS_FPDF: return None
    pdf = FPDF(); pdf.add_page()
    pdf.set_font("Arial", "B", 16); pdf.cell(0, 10, "NoiseShield Diagnostic Report", ln=1, align='C')
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.cell(0, 8, f"{L['mode']}: {domain}", ln=1)
    pdf.multi_cell(0, 8, f"{L['inputs']}: {inputs}")
    pdf.cell(0, 8, f"{L['prediction']}: {label}", ln=1)
    pdf.cell(0, 8, f"{L['confidence']}: {conf}%", ln=1)
    pdf.cell(0, 8, f"{L['noise_level']}: {noise}%", ln=1)
    return pdf.output(dest='S').encode('latin-1')

# =========================
# Synthetic training (offline baseline)
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
    health_pipe.fit(Xh, yh); water_pipe.fit(Xw, yw)
    return health_pipe, water_pipe

health_pipe, water_pipe = train_baselines()

# =========================
# Controls
# =========================
st.markdown(f"### {L['controls']}")
noise_pct = st.sidebar.slider(L["noise"], 0, 100, 0, step=5)

tabs = st.tabs(L["tabs"])

# =========================
# SOIL TAB
# =========================
with tabs[0]:
    st.subheader(L["soil"])
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
        pipe.fit(X, y); return pipe

    soil_pipe = train_soil_baseline()

    if st.button("Run Soil Analysis"):
        p_lr = soil_pipe.predict_proba(Xns)[0,1]
        subs_s = make_submodels_from(soil_pipe, eps=0.04)
        p_ens, sub_probs, weights, var = ensemble_predict_proba(subs_s, Xns)
        y_pred = int(p_ens >= 0.5)
        label  = L["soil_label_pos"] if y_pred==1 else L["soil_label_neg"]
        conf   = round(p_ens * 100, 2)

        st.markdown(f"**{L['predicted']}:** {label}")
        st.progress(int(conf))
        st.write(f"{L['confidence']}: **{conf}%** · {L['baseline']}: {p_lr:.2f} · {L['var']}: {var:.4f}")

        # Confidence-aware message
        if conf >= 80: st.success(f"🟢 {L['confidence']}: {conf}%")
        elif conf >= 50: st.warning(f"🟡 {L['confidence']}: {conf}%")
        else: st.error(f"🔴 {L['confidence']}: {conf}%")

        # Interference visualization
        with st.expander("Interference Visualization"):
            fig, ax = plt.subplots()
            ax.bar([f"Sub{i+1}" for i in range(len(sub_probs))], sub_probs, alpha=0.8)
            ax.axhline(p_ens, color='r', linestyle='--', label='Final (Interference)')
            ax.set_ylim(0,1); ax.set_ylabel("Probability"); ax.legend()
            st.pyplot(fig)

        # Explainability
        with st.expander(f"🧠 {L['why']}"):
            expl = linear_contribs(soil_pipe, Xns, ["pH","Nitrogen","Phosphorus","Potassium","Moisture"])
            df_expl = pd.DataFrame(expl, columns=["Feature","Contribution"]).set_index("Feature")
            st.bar_chart(df_expl)

        # Last-known-safe memory
        if conf >= 50:
            st.session_state["last_stable"]["Soil"] = {
                "inputs": X0s.tolist(), "label": label, "confidence": conf, "time": dt.datetime.now().isoformat()
            }
        else:
            last = st.session_state["last_stable"].get("Soil")
            if last:
                st.info(f"⚠️ {L['unstable']}: {last['time']}")
                st.json(last)

        # Persist results (dashboard + reports)
        st.session_state["results"]["Soil"] = {
            "confidence": float(conf), "prob": float(p_ens), "label": label,
            "time": dt.datetime.now().isoformat()
        }
        hist = st.session_state["history"]["Soil"]; hist.append(conf); st.session_state["history"]["Soil"] = hist[-3:]

        # Per-run CSV
        row = {
            L["domain"]: "Soil", "pH": float(spH), "Nitrogen": float(N), "Phosphorus": float(P),
            "Potassium": float(K), "Moisture": float(moist), L["prediction"]: label,
            L["prob"]: float(p_ens), L["conf"]: float(conf), L["time"]: dt.datetime.now().isoformat()
        }
        st.download_button(L["download_soil"], df_to_csv_bytes(pd.DataFrame([row])),
                           file_name="noiseshield_soil_result.csv", mime="text/csv")

        # Optional PDF
        if HAS_FPDF:
            pdf_bytes = pdf_report_bytes("Soil", row, label, conf, noise_pct)
            st.download_button(f"📄 {L['download_pdf']}", pdf_bytes, "noiseshield_soil_report.pdf",
                               mime="application/pdf")
        else:
            st.caption(L["pdf_missing"])

# =========================
# HEALTH TAB
# =========================
with tabs[1]:
    st.subheader(L["health"])
    c1, c2 = st.columns(2)
    with c1:
        hb    = st.number_input("Hemoglobin (g/dL)", 0.0, 25.0, 12.5)
        wbc   = st.number_input("WBC (cells/µL)", 0.0, 30000.0, 7000.0)
        pltlt = st.number_input("Platelets (cells/µL)", 0.0, 900000.0, 250000.0)
    with c2:
        temp  = st.number_input("Body Temp (°C)", 30.0, 45.0, 36.8)
        pulse = st.number_input("Pulse Rate (bpm)", 30.0, 200.0, 80.0)
    X0 = np.array([[hb, wbc, pltlt, temp, pulse]])
    Xn = inject_noise(X0, noise_pct)

    if st.button("Run Health Analysis"):
        p_lr = health_pipe.predict_proba(Xn)[0,1]
        subs_h = make_submodels_from(health_pipe, eps=0.04)
        p_ens, sub_probs, weights, var = ensemble_predict_proba(subs_h, Xn)
        y_pred = int(p_ens >= 0.5)
        label  = L["health_label_pos"] if y_pred==1 else L["health_label_neg"]
        conf   = round(p_ens * 100, 2)

        st.markdown(f"**{L['predicted']}:** {label}")
        st.progress(int(conf))
        st.write(f"{L['confidence']}: **{conf}%** · {L['baseline']}: {p_lr:.2f} · {L['var']}: {var:.4f}")

        if conf >= 80: st.success(f"🟢 {L['confidence']}: {conf}%")
        elif conf >= 50: st.warning(f"🟡 {L['confidence']}: {conf}%")
        else: st.error(f"🔴 {L['confidence']}: {conf}%")

        with st.expander("Interference Visualization"):
            fig, ax = plt.subplots()
            ax.bar([f"Sub{i+1}" for i in range(len(sub_probs))], sub_probs, alpha=0.8)
            ax.axhline(p_ens, color='r', linestyle='--', label='Final (Interference)')
            ax.set_ylim(0,1); ax.set_ylabel("Probability"); ax.legend()
            st.pyplot(fig)

        with st.expander(f"🧠 {L['why']}"):
            expl = linear_contribs(health_pipe, Xn, ["Hemoglobin","WBC","Platelets","Temp","Pulse"])
            df_expl = pd.DataFrame(expl, columns=["Feature","Contribution"]).set_index("Feature")
            st.bar_chart(df_expl)

        if conf >= 50:
            st.session_state["last_stable"]["Health"] = {
                "inputs": X0.tolist(), "label": label, "confidence": conf, "time": dt.datetime.now().isoformat()
            }
        else:
            last = st.session_state["last_stable"].get("Health")
            if last:
                st.info(f"⚠️ {L['unstable']}: {last['time']}")
                st.json(last)

        st.session_state["results"]["Health"] = {
            "confidence": float(conf), "prob": float(p_ens), "label": label,
            "time": dt.datetime.now().isoformat()
        }
        hist = st.session_state["history"]["Health"]; hist.append(conf); st.session_state["history"]["Health"] = hist[-3:]

        row = {
            L["domain"]: "Health", "Hemoglobin": float(hb), "WBC": float(wbc), "Platelets": float(pltlt),
            "Temp": float(temp), "Pulse": float(pulse), L["prediction"]: label, L["prob"]: float(p_ens),
            L["conf"]: float(conf), L["time"]: dt.datetime.now().isoformat()
        }
        st.download_button(L["download_health"], df_to_csv_bytes(pd.DataFrame([row])),
                           file_name="noiseshield_health_result.csv", mime="text/csv")
        if HAS_FPDF:
            pdf_bytes = pdf_report_bytes("Health", row, label, conf, noise_pct)
            st.download_button(f"📄 {L['download_pdf']}", pdf_bytes, "noiseshield_health_report.pdf",
                               mime="application/pdf")
        else:
            st.caption(L["pdf_missing"])

# =========================
# WATER TAB
# =========================
with tabs[2]:
    st.subheader(L["water"])
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
        label  = L["water_label_pos"] if y_pred==1 else L["water_label_neg"]
        conf   = round(p_ens * 100, 2)

        st.markdown(f"**{L['predicted']}:** {label}")
        st.progress(int(conf))
        st.write(f"{L['confidence']}: **{conf}%** · {L['baseline']}: {p_lr:.2f} · {L['var']}: {var:.4f}")

        if conf >= 80: st.success(f"🟢 {L['confidence']}: {conf}%")
        elif conf >= 50: st.warning(f"🟡 {L['confidence']}: {conf}%")
        else: st.error(f"🔴 {L['confidence']}: {conf}%")

        with st.expander("Interference Visualization"):
            fig, ax = plt.subplots()
            ax.bar([f"Sub{i+1}" for i in range(len(sub_probs))], sub_probs, alpha=0.8)
            ax.axhline(p_ens, color='r', linestyle='--', label='Final (Interference)')
            ax.set_ylim(0,1); ax.set_ylabel("Probability"); ax.legend()
            st.pyplot(fig)

        with st.expander(f"🧠 {L['why']}"):
            expl = linear_contribs(water_pipe, Xnw, ["pH","Turbidity","TDS","EC","Water Temp"])
            df_expl = pd.DataFrame(expl, columns=["Feature","Contribution"]).set_index("Feature")
            st.bar_chart(df_expl)

        if conf >= 50:
            st.session_state["last_stable"]["Water"] = {
                "inputs": X0w.tolist(), "label": label, "confidence": conf, "time": dt.datetime.now().isoformat()
            }
        else:
            last = st.session_state["last_stable"].get("Water")
            if last:
                st.info(f"⚠️ {L['unstable']}: {last['time']}")
                st.json(last)

        st.session_state["results"]["Water"] = {
            "confidence": float(conf), "prob": float(p_ens), "label": label,
            "time": dt.datetime.now().isoformat()
        }
        hist = st.session_state["history"]["Water"]; hist.append(conf); st.session_state["history"]["Water"] = hist[-3:]

        row = {
            L["domain"]: "Water", "pH": float(ph), "Turbidity": float(turb), "TDS": float(tds),
            "EC": float(ec), "WaterTemp": float(wtemp), L["prediction"]: label, L["prob"]: float(p_ens),
            L["conf"]: float(conf), L["time"]: dt.datetime.now().isoformat()
        }
        st.download_button(L["download_water"], df_to_csv_bytes(pd.DataFrame([row])),
                           file_name="noiseshield_water_result.csv", mime="text/csv")
        if HAS_FPDF:
            pdf_bytes = pdf_report_bytes("Water", row, label, conf, noise_pct)
            st.download_button(f"📄 {L['download_pdf']}", pdf_bytes, "noiseshield_water_report.pdf",
                               mime="application/pdf")
        else:
            st.caption(L["pdf_missing"])

# =========================
# QUANTUM VIEW
# =========================
with tabs[3]:
    st.subheader(L["quantum"])
    st.write("• Multiple perturbed submodels (logistic regressions) act like **amplitude paths**.")
    st.write("• When they disagree strongly, weights equalize → **noise cancellation** (destructive-like).")
    st.write("• When they agree, central estimate is reinforced → **stability** (constructive-like).")
    st.markdown(f"<div class='panel'>{L['note_panel']}</div>", unsafe_allow_html=True)

# =========================
# REPORTS (summary)
# =========================
with tabs[4]:
    st.subheader(L["reports"])
    res = st.session_state["results"]
    rows = []
    for dom in ["Soil","Health","Water"]:
        r = res.get(dom)
        if r is not None:
            rows.append({L["domain"]: dom, L["prediction"]: r["label"],
                         L["prob"]: r["prob"], L["conf"]: r["confidence"], L["time"]: r["time"]})
        else:
            rows.append({L["domain"]: dom, L["prediction"]: "—",
                         L["prob"]: "—", L["conf"]: 0, L["time"]: "—"})
    df_all = pd.DataFrame(rows)
    st.dataframe(df_all, use_container_width=True)
    st.download_button(f"⬇️ {L['download_summary']}",
                       data=df_to_csv_bytes(df_all),
                       file_name="noiseshield_summary.csv", mime="text/csv")

# =========================
# DASHBOARD
# =========================
with tabs[5]:
    st.subheader(L["dashboard"])
    r = st.session_state["results"]
    soil_conf   = float((r["Soil"]   or {}).get("confidence", 0.0))
    health_conf = float((r["Health"] or {}).get("confidence", 0.0))
    water_conf  = float((r["Water"]  or {}).get("confidence", 0.0))

    c1, c2, c3 = st.columns(3)
    c1.metric(L["sdg2"], f"{soil_conf:.1f}%")
    c2.metric(L["sdg3"], f"{health_conf:.1f}%")
    c3.metric(L["sdg6"], f"{water_conf:.1f}%")

    overall = round((soil_conf + health_conf + water_conf) / 3, 1)
    status = L["excellent"] if overall >= 80 else (L["moderate"] if overall >= 50 else L["needs"])
    st.markdown(f"### {L['overall']}: **{overall}%** · {status}")

    with st.expander(L["trend"]):
        fig, ax = plt.subplots()
        ax.plot(st.session_state["history"]["Soil"],   "o-", label="Soil")
        ax.plot(st.session_state["history"]["Health"], "o-", label="Health")
        ax.plot(st.session_state["history"]["Water"],  "o-", label="Water")
        ax.set_ylim(0, 100); ax.set_ylabel("Confidence (%)"); ax.set_xlabel("Recent Tests")
        ax.legend(); st.pyplot(fig)

    st.caption(L["caption"])

# =========================
# CROSS-DOMAIN STABILITY (Soil, Health, Water)
# =========================
with tabs[6]:
    st.subheader(L["stability"])
    st.caption("Noise (%) → Probability (baseline vs interference)")

    levels = [0, 20, 40, 60, 80, 100]

    # Soil probe point (use current inputs)
    soil_subs = make_submodels_from(train_soil_baseline(), eps=0.04)
    soil_base, soil_ens = robustness_curve(train_soil_baseline(), soil_subs, np.array([[6.5,40,30,120,25]]), levels)

    # Health probe (typical normal vitals)
    health_subs = make_submodels_from(health_pipe, eps=0.04)
    health_base, health_ens = robustness_curve(health_pipe, health_subs, np.array([[12.5,7000,250000,36.8,80]]), levels)

    # Water probe (typical potable water)
    water_subs = make_submodels_from(water_pipe, eps=0.04)
    water_base, water_ens = robustness_curve(water_pipe, water_subs, np.array([[7.2,5,300,600,25]]), levels)

    def plot_curve(title, base, ens):
        st.markdown(f"**{title}**")
        fig, ax = plt.subplots()
        ax.plot(levels, base, marker="o", label="Baseline (LR)")
        ax.plot(levels, ens, marker="o", label="Interference Ensemble")
        ax.set_xlabel("Noise (%)"); ax.set_ylabel("Positive Probability"); ax.set_ylim(0,1); ax.legend()
        st.pyplot(fig)

    plot_curve("Soil",   soil_base, soil_ens)
    plot_curve("Health", health_base, health_ens)
    plot_curve("Water",  water_base, water_ens)
