import streamlit as st
import joblib, re
from PyPDF2 import PdfReader
import docx2txt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# ---------- Helpers ----------
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s]{8,}\d)')
SECTION_HINTS = [
    "summary","objective","experience","work experience","professional experience",
    "education","projects","skills","certifications","awards","achievements","publications",
    "languages","interests","activities","leadership"
]
stop_words = set(stopwords.words("english"))

def clean_text(t):
    import re
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
    except Exception:
        return ""

def jd_resume_match_score(resume_text: str, jd_text: str, vectorizer) -> float:
    A = vectorizer.transform([resume_text])
    B = vectorizer.transform([jd_text])
    return float(cosine_similarity(A, B)[0][0])

def estimate_pages(resume_text: str) -> int:
    words = len(resume_text.split())
    return max(1, int(np.ceil(words / 600)))

def ats_score(resume_text: str, jd_text: str = "", base_vectorizer=None) -> dict:
    raw = resume_text
    text = resume_text.lower()
    score = 0
    breakdown = {}

    # 1) Length/pages (20)
    pages = estimate_pages(raw)
    if 1 <= pages <= 2: part = 18
    elif pages == 3:    part = 14
    else:               part = 8
    score += part; breakdown["length_pages_(20)"] = part

    # 2) Contact info (20)
    email_ok = bool(EMAIL_RE.search(raw))
    phone_ok = bool(PHONE_RE.search(raw))
    part = (10 if email_ok else 0) + (10 if phone_ok else 0)
    score += part; breakdown["contact_info_(20)"] = part

    # 3) Sections (20)
    found = sum(1 for s in SECTION_HINTS if s in text)
    part = min(20, found * 2)
    score += part; breakdown["sections_(20)"] = part

    # 4) Keyword coverage (40)
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

# ---------- Load models ----------
clf = joblib.load("models/resume_classifier.joblib")
tfidf_vec = joblib.load("models/tfidf_vectorizer.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

# ---------- UI ----------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🧠",
)

st.title("🧠 AI Resume Analyzer")
st.caption("Upload your resume and (optionally) paste a Job Description to get a match score and ATS score.")

jd_text = st.text_area("Paste Job Description (optional for keyword coverage & match score)")

uploaded = st.file_uploader("Upload Resume (.pdf or .docx)", type=["pdf","docx"])

if uploaded:
    resume_text_raw = extract_text_from_file(uploaded)
    if not resume_text_raw.strip():
        st.error("Could not extract text from the uploaded file.")
    else:
        st.subheader("Resume Preview")
        st.write(resume_text_raw[:800] + ("..." if len(resume_text_raw) > 800 else ""))

        # Cleaned text for models
        resume_clean = clean_text(resume_text_raw)

        # Predict category
        pred_idx = clf.predict(tfidf_vec.transform([resume_clean]))[0]
        category = label_encoder.inverse_transform([pred_idx])[0]

        # JD–Resume match (if JD given)
        if jd_text.strip():
            match = jd_resume_match_score(resume_clean, clean_text(jd_text), tfidf_vec)
            st.metric("JD–Resume Match (0–1)", f"{match:.3f}")

        # ATS score
        ats = ats_score(resume_text_raw, jd_text, base_vectorizer=tfidf_vec)
        st.metric("ATS Score (0–100)", f"{ats['total']}")

        st.write("**Predicted Category:**", category)
        st.write("**Pages (estimated):**", ats["pages_estimated"])
        st.write("**Contact Info Found:**", f"Email: {ats['email_found']} • Phone: {ats['phone_found']}")
        st.write("**ATS Breakdown:**")
        st.json({k:v for k,v in ats.items() if k.endswith(')')})
        st.write("**Suggested keywords to add:**", ", ".join(ats["missing_keywords_suggestion"]))

