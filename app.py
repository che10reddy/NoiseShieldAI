# =========================
# NoiseShield AI · Quantum-Inspired Diagnostics
# =========================
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"   # Prevent inotify crash

import streamlit as st
import numpy as np, pandas as pd, datetime as dt
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False


# ---------- App meta ----------
st.set_page_config(page_title="NoiseShield AI", page_icon="🔰", layout="centered")

# ---------- Session init ----------
defaults = {
    "results": {"Soil": None, "Health": None, "Water": None},
    "history": {"Soil": [], "Health": [], "Water": []},
    "last_stable": {"Soil": None, "Health": None, "Water": None},
    "theme_mode": "Dark",
}
for k,v in defaults.items(): st.session_state.setdefault(k,v)


# ---------- Theme ----------
theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"],
    index=0 if st.session_state["theme_mode"]=="Light" else 1)
st.session_state["theme_mode"] = theme_choice
is_dark = st.session_state["theme_mode"]=="Dark"

bg   = "#0E1117" if is_dark else "#FFFFFF"
txt  = "#FAFAFA" if is_dark else "#111111"
acc  = "#00B4B4" if is_dark else "#0A84FF"
panel= "#151922" if is_dark else "#F7F9FC"

st.markdown(f"""
<style>
body,.stApp{{background:{bg};color:{txt};}}
.stButton>button{{background:{acc};color:white;border-radius:8px;font-weight:600;border:0;}}
.stProgress>div>div{{background:{acc}!important;}}
.panel{{background:{panel};padding:12px 14px;border-radius:10px;border:1px solid {acc}30;}}
.block-container{{max-width:980px;padding-top:1rem;}}
</style>
""", unsafe_allow_html=True)


# ---------- Language ----------
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
        "baseline": "Baseline Probability",
        "var": "Disagreement Variance",
        "download_soil": "Download Soil Result (CSV)",
        "download_health": "Download Health Result (CSV)",
        "download_water": "Download Water Result (CSV)",
        "download_summary": "Download Summary (CSV)",
        "download_pdf": "Download PDF Report",
        "pdf_missing": "Install 'fpdf' to enable PDF report.",
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
        "caption": "Quantum-inspired, offline tool for soil, health, and water diagnostics in low-resource settings."
    },
    "తెలుగు": {
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
        "stability": "క్రాస్-డొమైన్ స్థిరత్వం (Noise vs Probability)",
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
        "caption": "క్వాంటమ్ ప్రేరణతో, ఆఫ్‌లైన్ టూల్ — తక్కువ వనరుల ప్రాంతాలకు."
    },
    "हिंदी": {
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
        "caption": "क्वांटम-प्रेरित, ऑफलाइन टूल — कम संसाधन सेटिंग्स के लिए।"
    },
    "தமிழ்": {
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
        "stability": "குறுக்கு-விளைநிலை நிலைத்தன்மை (Noise vs Probability)",
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
        "caption": "குவாண்டம் ஊக்கமூட்டிய, ஆஃப்லைன் கருவி — குறைந்த வள பகுதிகளுக்கு."
    },
    "বাংলা": {
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
        "caption": "কোয়ান্টাম-প্রাণিত, অফলাইন টুল — স্বল্প সম্পদ সেটিংসে।"
    },
    "मराठी": {
        "title": "NoiseShield AI · क्वांटम-प्रेरित निदान",
        "sdg2": "SDG 2 · उपासमार निर्मूलन",
        "sdg3": "SDG 3 · चांगले आरोग्य",
        "sdg6": "SDG 6 · स्वच्छ पाणी",
        "tabs": ["माती (SDG 2)", "आरोग্য (SDG 3)", "पाणी (SDG 6)",
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
        "caption": "क्वांटम-प्रेरित, ऑफलाइन साधन — कमी संसाधन भागांसाठी."
    }
}

ui_lang = st.sidebar.selectbox("Language", list(LANG.keys()), index=0)
L = LANG[ui_lang]


# ---------- Banner ----------
st.markdown(f"<h2 style='text-align:center;font-weight:800;'>{L['title']}</h2>",unsafe_allow_html=True)


# ---------- Utilities ----------
def seed_rng(s=42): return np.random.default_rng(s)
def inject_noise(X,p,r=None):
    if p<=0:return X.copy()
    r=r or seed_rng(123);return X*(1+r.normal(0,p/100,size=X.shape))
def make_submodels_from(base,eps=0.03,n=3):
    sc=base.named_steps['standardscaler'];lr=base.named_steps['logisticregression'];subs=[]
    for k in range(n):
        lrk=LogisticRegression();lrk.classes_=lr.classes_
        lrk.coef_=lr.coef_*(1+(k-1)*eps);lrk.intercept_=lr.intercept_*(1+(k-1)*eps)
        p=make_pipeline(StandardScaler());p.fit(np.zeros((1,len(sc.scale_))),[0])
        p.named_steps['standardscaler'].mean_=sc.mean_.copy()
        p.named_steps['standardscaler'].scale_=sc.scale_.copy()
        p.steps.append(('logisticregression',lrk));subs.append(p)
    return subs
def ensemble_predict_proba(subs,X):
    probs=np.array([m.predict_proba(X)[0,1] for m in subs]);var=float(np.var(probs))
    w=np.ones_like(probs)/len(probs) if var>0.02 else np.exp(-(probs-probs.mean())**2/(2*0.0025))
    w/=w.sum();return float(w@probs),probs,w,var
def df_to_csv_bytes(df): s=StringIO();df.to_csv(s,index=False);return s.getvalue().encode()
def pdf_report_bytes(dom,row,label,conf,noise):
    if not HAS_FPDF:return None
    p=FPDF();p.add_page();p.set_font("Arial","B",16)
    p.cell(0,10,"NoiseShield Diagnostic Report",ln=1,align='C')
    p.set_font("Arial","",12)
    for k,v in [("Domain",dom),("Prediction",label),("Confidence",f"{conf}%"),("Noise",f"{noise}%")]:
        p.cell(0,8,f"{k}: {v}",ln=1)
    return p.output(dest='S').encode('latin-1')


# ---------- Synthetic Training ----------
def synth_health_data(n=400,r=None):
    r=r or seed_rng(1)
    Hb=r.normal(13,2.2,n).clip(6,20);WBC=r.normal(7000,2500,n).clip(2e3,3e4)
    PLT=r.normal(250000,80000,n).clip(7e4,8e5)
    Temp=r.normal(36.8,0.7,n).clip(34.5,41.5);Pulse=r.normal(80,15,n).clip(45,160)
    X=np.c_[Hb,WBC,PLT,Temp,Pulse];y=((Hb<11)|((Temp>37.8)&(WBC>10000))|(PLT<120000)).astype(int);return X,y
def synth_water_data(n=400,r=None):
    r=r or seed_rng(2);pH=r.normal(7.1,0.6,n).clip(4.5,9.5)
    turb=np.abs(r.normal(5,15,n)).clip(0,200);tds=np.abs(r.normal(300,250,n)).clip(50,2500)
    ec=np.abs(r.normal(600,400,n)).clip(50,4500);temp=r.normal(24,6,n).clip(5,45)
    X=np.c_[pH,turb,tds,ec,temp];y=((turb>10)|(tds>1000)|(ec>2000)|(pH<6)|(pH>8.5)).astype(int);return X,y
@st.cache_resource
def train_baselines():
    Xh,yh=synth_health_data();Xw,yw=synth_water_data()
    hp=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))
    wp=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))
    hp.fit(Xh,yh);wp.fit(Xw,yw)
    # Soil baseline
    r=seed_rng(3);n=400
    pH=r.normal(6.5,0.8,n).clip(3.5,9.5);N=r.normal(50,25,n).clip(0,200)
    P=r.normal(40,20,n).clip(0,200);K=r.normal(150,60,n).clip(0,300);M=r.normal(30,15,n).clip(0,100)
    X=np.c_[pH,N,P,K,M];y=((N<30)|(P<20)|(K<80)|(pH<5.5)|(pH>8.5)).astype(int)
    sp=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500));sp.fit(X,y)
    return sp,hp,wp
soil_pipe,health_pipe,water_pipe=train_baselines()


# ---------- Controls ----------
st.sidebar.header(L["controls"])
noise=st.sidebar.slider(L["noise"],0,100,0,5)
tabs=st.tabs(L["tabs"])


# ---------- Soil ----------
with tabs[0]:
    st.subheader("Soil Fertility Analysis")
    c1,c2=st.columns(2)
    with c1:
        spH=st.number_input("Soil pH",3.,10.,6.5)
        N=st.number_input("Nitrogen (mg/kg)",0.,200.,40.)
        P=st.number_input("Phosphorus (mg/kg)",0.,200.,30.)
    with c2:
        K=st.number_input("Potassium (mg/kg)",0.,300.,120.)
        M=st.number_input("Moisture (%)",0.,100.,25.)
    X=np.array([[spH,N,P,K,M]]);Xn=inject_noise(X,noise)
    if st.button("Run Soil Analysis"):
        subs=make_submodels_from(soil_pipe,0.04)
        p_ens,subs_p,_,var=ensemble_predict_proba(subs,Xn)
        label="Nutrient Deficient" if p_ens>=0.5 else "Fertile"
        conf=round(p_ens*100,2)
        st.write(f"**{L['predicted']}** {label}")
        st.progress(int(conf))
        st.write(f"{L['confidence']}: {conf}% · {L['var']}: {var:.4f}")
        fig,ax=plt.subplots();ax.bar([f"S{i+1}" for i in range(len(subs_p))],subs_p)
        ax.axhline(p_ens,color='r',ls='--',label='Final');ax.legend();ax.set_ylim(0,1);st.pyplot(fig)
        if HAS_FPDF:
            pdf=pdf_report_bytes("Soil",{"pH":spH},label,conf,noise)
            st.download_button("📄 PDF",pdf,"soil_report.pdf")


# ---------- Health ----------
with tabs[1]:
    st.subheader("Health Diagnostics")
    hb=st.number_input("Hemoglobin (g/dL)",0.,25.,12.5)
    wbc=st.number_input("WBC (cells/µL)",0.,3e4,7e3)
    pltlt=st.number_input("Platelets (cells/µL)",0.,9e5,2.5e5)
    temp=st.number_input("Body Temp (°C)",30.,45.,36.8)
    pulse=st.number_input("Pulse Rate (bpm)",30.,200.,80.)
    X=np.array([[hb,wbc,pltlt,temp,pulse]]);Xn=inject_noise(X,noise)
    if st.button("Run Health Analysis"):
        subs=make_submodels_from(health_pipe,0.04)
        p_ens,subs_p,_,var=ensemble_predict_proba(subs,Xn)
        label="Possible Condition" if p_ens>=0.5 else "Healthy"
        conf=round(p_ens*100,2)
        st.write(f"**{L['predicted']}** {label}")
        st.progress(int(conf))
        fig,ax=plt.subplots();ax.bar([f"S{i+1}" for i in range(len(subs_p))],subs_p)
        ax.axhline(p_ens,color='r',ls='--',label='Final');ax.legend();ax.set_ylim(0,1);st.pyplot(fig)


# ---------- Water ----------
with tabs[2]:
    st.subheader("Water Quality Analysis")
    ph=st.number_input("pH",0.,14.,7.2)
    turb=st.number_input("Turbidity (NTU)",0.,500.,5.)
    tds=st.number_input("TDS (ppm)",0.,5000.,300.)
    ec=st.number_input("EC (µS/cm)",0.,10000.,600.)
    wtemp=st.number_input("Water Temp (°C)",0.,60.,25.)
    X=np.array([[ph,turb,tds,ec,wtemp]]);Xn=inject_noise(X,noise)
    if st.button("Run Water Analysis"):
        subs=make_submodels_from(water_pipe,0.04)
        p_ens,subs_p,_,var=ensemble_predict_proba(subs,Xn)
        label="Contaminated" if p_ens>=0.5 else "Safe"
        conf=round(p_ens*100,2)
        st.write(f"**{L['predicted']}** {label}")
        st.progress(int(conf))
        fig,ax=plt.subplots();ax.bar([f"S{i+1}" for i in range(len(subs_p))],subs_p)
        ax.axhline(p_ens,color='r',ls='--',label='Final');ax.legend();ax.set_ylim(0,1);st.pyplot(fig)


# ---------- Quantum View ----------
with tabs[3]:
    st.subheader("Quantum-Inspired View")
    st.write("• Multiple submodels ≈ amplitude paths.")
    st.write("• High disagreement → equalized weights → destructive interference.")
    st.write("• Agreement → constructive reinforcement → stability.")
    st.markdown(f"<div class='panel'>{L['note_panel']}</div>",unsafe_allow_html=True)


# ---------- Reports ----------
with tabs[4]:
    st.subheader("Reports (Summary)")
    rows=[]
    for d in ["Soil","Health","Water"]:
        r=st.session_state["results"].get(d)
        if r: rows.append({L["domain"]:d,L["prediction"]:r.get("label","—"),
                           L["prob"]:r.get("prob",0),L["conf"]:r.get("confidence",0),
                           L["time"]:r.get("time","—")})
    df=pd.DataFrame(rows or [{L["domain"]:"—",L["prediction"]:"—"}])
    df.replace("—",np.nan,inplace=True)
    st.dataframe(df,use_container_width=True)
    st.download_button("⬇️ Download Summary CSV",df_to_csv_bytes(df),"noiseshield_summary.csv","text/csv")


# ---------- Dashboard ----------
with tabs[5]:
    st.subheader(L["overall"])
    res=st.session_state["results"]
    vals=[(res[d] or {}).get("confidence",0) for d in ["Soil","Health","Water"]]
    overall=round(np.mean(vals),1)
    status=L["excellent"] if overall>=80 else (L["moderate"] if overall>=50 else L["needs"])
    st.metric(L["overall"],f"{overall}%",status)
    st.caption(L["caption"])


# ---------- Cross-Domain Stability ----------
with tabs[6]:
    st.subheader("Cross-Domain Stability (Noise vs Probability)")
    lvls=[0,20,40,60,80,100]
    def rc(pipe,X):
        subs=make_submodels_from(pipe,0.04)
        b,e=[],[]
        for n in lvls:
            Xn=inject_noise(X,n)
            b.append(pipe.predict_proba(Xn)[0,1])
            e.append(ensemble_predict_proba(subs,Xn)[0])
        return np.array(b),np.array(e)
    soil_b,soil_e=rc(soil_pipe,np.array([[6.5,40,30,120,25]]))
    health_b,health_e=rc(health_pipe,np.array([[12.5,7000,250000,36.8,80]]))
    water_b,water_e=rc(water_pipe,np.array([[7.2,5,300,600,25]]))
    for title,base,ens in [("Soil",soil_b,soil_e),("Health",health_b,health_e),("Water",water_b,water_e)]:
        st.markdown(f"**{title}**")
        fig,ax=plt.subplots()
        ax.plot(lvls,base,"o-",label="Baseline (LR)")
        ax.plot(lvls,ens,"o-",label="Interference Ensemble")
        ax.set_xlabel("Noise (%)");ax.set_ylabel("Positive Probability");ax.set_ylim(0,1);ax.legend()
        st.pyplot(fig)
