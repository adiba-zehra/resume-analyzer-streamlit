# app.py
import streamlit as st
import joblib, re, os
from PyPDF2 import PdfReader
import docx2txt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
from pathlib import Path
from textwrap import shorten

# ========== NLTK stopwords (download only if missing) ==========
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    with st.spinner("Downloading NLTK data..."):
        nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))

# ========== Helpers & Patterns ==========
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s]{8,}\d)')
SECTION_HINTS = [
    "summary","objective","experience","work experience","professional experience",
    "education","projects","skills","certifications","awards","achievements","publications",
    "languages","interests","activities","leadership"
]

def clean_text(t):
    if not isinstance(t, str): return ""
    t = re.sub(r'http\S+|www\S+', ' ', t)
    t = re.sub(r'[^a-zA-Z]', ' ', t)
    t = t.lower()
    t = re.sub(r'\s+', ' ', t).strip()
    tokens = [w for w in t.split() if w not in stop_words]
    return " ".join(tokens)

def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)
    try:
        if name.lower().endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            txt = ""
            for p in reader.pages:
                txt += p.extract_text() or ""
            return txt
        elif name.lower().endswith(".docx"):
            return docx2txt.process(uploaded_file)
        else:
            return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error("Error extracting text from the file.")
        return ""

def jd_resume_match_score(resume_text: str, jd_text: str, vectorizer) -> float:
    A = vectorizer.transform([resume_text])
    B = vectorizer.transform([jd_text])
    return float(cosine_similarity(A, B)[0][0])

def estimate_pages(resume_text: str) -> int:
    words = len(resume_text.split())
    return max(1, int(np.ceil(words / 600)))

def ats_score(resume_text: str, jd_text: str = "", base_vectorizer=None) -> dict:
    raw = resume_text or ""
    text = raw.lower()
    score = 0
    breakdown = {}
    # Length/pages (20)
    pages = estimate_pages(raw)
    if 1 <= pages <= 2: part = 18
    elif pages == 3:    part = 14
    else:               part = 8
    score += part; breakdown["length_pages_(20)"] = part
    # Contact (20)
    email_ok = bool(EMAIL_RE.search(raw))
    phone_ok = bool(PHONE_RE.search(raw))
    part = (10 if email_ok else 0) + (10 if phone_ok else 0)
    score += part; breakdown["contact_info_(20)"] = part
    # Sections (20)
    found = sum(1 for s in SECTION_HINTS if s in text)
    part = min(20, found * 2)
    score += part; breakdown["sections_(20)"] = part
    # Keyword coverage (40)
    keywords = []
    if jd_text and jd_text.strip():
        vect = TfidfVectorizer(max_features=50, stop_words='english')
        mat = vect.fit_transform([jd_text])
        keywords = [t for t in vect.get_feature_names_out() if len(t) > 2]
    else:
        keywords = [
            "python","java","sql","machine learning","deep learning","nlp","excel","power bi","tableau",
            "pandas","numpy","scikit","tensorflow","pytorch","html","css","javascript","aws","azure","git",
            "docker","kubernetes","communication","leadership","teamwork","mysql","linux"
        ]
    hits, missing = 0, []
    for k in keywords:
        if k in text: hits += 1
        else: missing.append(k)
    coverage = hits / max(1, len(keywords))
    part = int(round(coverage * 40))
    score += part; breakdown["keyword_coverage_(40)"] = part
    breakdown["missing_keywords_suggestion"] = missing[:20]
    breakdown["total"] = min(100, score)
    breakdown["pages_estimated"] = pages
    breakdown["email_found"] = email_ok
    breakdown["phone_found"] = phone_ok
    return breakdown

# ========== Load models safely ==========
MODEL_DIR = Path("models")
clf = tfidf_vec = label_encoder = None
if MODEL_DIR.exists():
    try:
        clf = joblib.load(MODEL_DIR/"resume_classifier.joblib")
        tfidf_vec = joblib.load(MODEL_DIR/"tfidf_vectorizer.joblib")
        label_encoder = joblib.load(MODEL_DIR/"label_encoder.joblib")
    except Exception as e:
        st.error("Error loading model artifacts. Make sure /models contains joblib files.")
else:
    st.warning("No models/ folder found — classifier won't be available (upload the models folder).")

# ========== UI tweaks & CSS ==========
st.set_page_config(page_title="AI Resume Analyzer", page_icon="🧠", layout="wide")
st.markdown("""
<style>
body {background: linear-gradient(180deg,#0f172a,#0b1220);}
h1 {color: #9be7ff; text-align:center;}
.stApp { color: #e6eef6; }
div.block-container{padding:1rem 2rem;}
.stButton>button {background:linear-gradient(90deg,#7c3aed,#06b6d4); border:none; color:white;}
.stMetric {border-radius: 12px; padding: 8px;}
</style>
""", unsafe_allow_html=True)

# ========== Page layout ==========
st.title("🧠 AI Resume Analyzer — Clean, Fast, Smart")
st.write("Upload a resume (PDF/DOCX). Optionally paste a Job Description to get match & ATS feedback.")
left, right = st.columns((2,1))

with right:
    st.image("https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/_static/img/transformers-logo.png", width=120)  # optional - replace with your logo
    st.write("**Quick actions**")
    if st.button("Example Resume"):
        st.info("You can upload a resume file to test the app.")

with left:
    jd_text = st.text_area("Paste Job Description (optional)", height=140)
    uploaded = st.file_uploader("Upload Resume (.pdf or .docx)", type=["pdf","docx"])
    show_full = st.checkbox("Show full extracted text", value=False)

if uploaded:
    with st.spinner("Extracting and analyzing..."):
        resume_text_raw = extract_text_from_file(uploaded)
        if not resume_text_raw.strip():
            st.error("Could not extract text from the uploaded file.")
        else:
            resume_clean = clean_text(resume_text_raw)
            # Predicted category
            if clf is not None and tfidf_vec is not None and label_encoder is not None:
                try:
                    pred_idx = clf.predict(tfidf_vec.transform([resume_clean]))[0]
                    category = label_encoder.inverse_transform([pred_idx])[0]
                except Exception:
                    category = "Prediction failed"
            else:
                category = "Model not available"
            # Match score
            match = None
            if jd_text and tfidf_vec is not None:
                match = jd_resume_match_score(resume_clean, clean_text(jd_text), tfidf_vec)
            # ATS
            ats = ats_score(resume_text_raw, jd_text, base_vectorizer=tfidf_vec)
    # Display results
    col1, col2, col3 = st.columns(3)
    col1.metric("ATS Score", f"{ats['total']} / 100")
    col2.metric("Estimated Pages", ats['pages_estimated'])
    if match is not None:
        col3.metric("JD Match (0-1)", f"{match:.3f}")
    st.markdown("---")
    st.subheader("Predicted Domain / Category")
    st.success(category)
    st.subheader("ATS Breakdown")
    st.json({k:v for k,v in ats.items() if k.endswith(')')})
    st.subheader("Suggested keywords to add")
    st.write(", ".join(ats["missing_keywords_suggestion"]))
    st.markdown("---")
    st.subheader("Resume Preview")
    if show_full:
        st.text(resume_text_raw)
    else:
        st.text(shorten(resume_text_raw, width=1200, placeholder="..."))
    # Allow download of cleaned resume text
    st.download_button("Download cleaned resume text", data=resume_clean, file_name="resume_clean.txt")
else:
    st.info("Upload a resume to begin analysis.")
