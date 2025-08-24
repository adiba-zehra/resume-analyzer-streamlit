# app.py (Gradio version)
import gradio as gr
import joblib, re, os, tempfile
from PyPDF2 import PdfReader
import docx2txt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
from pathlib import Path
from textwrap import shorten

# ---------- NLTK stopwords (download if necessary) ----------
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# ---------- Helpers & Patterns ----------
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

def extract_text_from_file(uploaded):
    """
    uploaded may be:
      - a local path (str) when Gradio provides it,
      - a (name, bytes) tuple in some contexts,
      - a file-like object with .name attribute.
    """
    path = None
    if uploaded is None:
        return ""
    # If Gradio gives a string path
    if isinstance(uploaded, str) and os.path.exists(uploaded):
        path = uploaded
    # If it's a tuple (name, fileobj) sometimes returned
    elif isinstance(uploaded, tuple) and len(uploaded) >= 1:
        cand = uploaded[0]
        if isinstance(cand, str) and os.path.exists(cand):
            path = cand
    # If it has .name attribute (file-like)
    elif hasattr(uploaded, "name") and os.path.exists(uploaded.name):
        path = uploaded.name

    # If we still don't have a path, try saving the bytes to temp
    if path is None:
        try:
            # uploaded might be a file-like object; read bytes
            data = uploaded.read()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
            tmp.write(data)
            tmp.close()
            path = tmp.name
        except Exception:
            return ""

    # Now extract using file path & extension
    name = str(path).lower()
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(path)
            txt = ""
            for p in reader.pages:
                txt += p.extract_text() or ""
            return txt
        elif name.endswith(".docx"):
            return docx2txt.process(path)
        else:
            # fallback: read as text
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        return ""

def jd_resume_match_score(resume_text: str, jd_text: str, vectorizer) -> float:
    if vectorizer is None:
        return None
    try:
        A = vectorizer.transform([resume_text])
        B = vectorizer.transform([jd_text])
        return float(cosine_similarity(A, B)[0][0])
    except Exception:
        return None

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

# ========== Load models (if present) ==========
MODEL_DIR = Path("models")
clf = tfidf_vec = label_encoder = None
if MODEL_DIR.exists():
    try:
        clf = joblib.load(MODEL_DIR/"resume_classifier.joblib")
        tfidf_vec = joblib.load(MODEL_DIR/"tfidf_vectorizer.joblib")
        label_encoder = joblib.load(MODEL_DIR/"label_encoder.joblib")
    except Exception as e:
        clf = tfidf_vec = label_encoder = None
else:
    clf = tfidf_vec = label_encoder = None

# ========== Analysis function ==========
def analyze_resume(uploaded_file, jd_text):
    resume_text_raw = extract_text_from_file(uploaded_file)
    if not resume_text_raw or not resume_text_raw.strip():
        return (
            "Could not extract text from the file.",
            "Model not available",
            None,
            None,
            {},
            "",
            None
        )

    resume_clean = clean_text(resume_text_raw)

    # category
    if clf is not None and tfidf_vec is not None and label_encoder is not None:
        try:
            pred_idx = clf.predict(tfidf_vec.transform([resume_clean]))[0]
            category = label_encoder.inverse_transform([pred_idx])[0]
        except Exception:
            category = "Prediction failed"
    else:
        category = "Model not available"

    # match
    match_score = None
    if jd_text and tfidf_vec is not None:
        match_score = jd_resume_match_score(resume_clean, clean_text(jd_text), tfidf_vec)

    # ats
    ats = ats_score(resume_text_raw, jd_text, base_vectorizer=tfidf_vec)

    # create a temp file for the cleaned resume text to allow download in Gradio
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmpf.write(resume_clean.encode("utf-8"))
    tmpf.flush()
    tmpf.close()
    download_path = tmpf.name

    preview = resume_text_raw[:4000] + ("..." if len(resume_text_raw) > 4000 else "")

    suggestions = ", ".join(ats.get("missing_keywords_suggestion", []))
    return (preview, category, match_score, ats.get("total", None), ats, suggestions, download_path)

# ========== Gradio UI ==========
css = """
body {background: linear-gradient(180deg,#0f172a,#071025);}
.gr-button { background: linear-gradient(90deg,#7c3aed,#06b6d4); color: white;}
"""

with gr.Blocks(title="AI Resume Analyzer", css=css) as demo:
    gr.Markdown("# 🧠 AI Resume Analyzer")
    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(label="Upload Resume (.pdf or .docx)")
            jd_input = gr.Textbox(label="Paste Job Description (optional)", lines=4)
            analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=1):
            preview_out = gr.Textbox(label="Resume Preview (first 4000 chars)", lines=12)
            category_out = gr.Textbox(label="Predicted Category")
            match_out = gr.Number(label="JD–Resume Match (0–1)")
            ats_out = gr.Number(label="ATS Score (0–100)")
            ats_json = gr.JSON(label="ATS Breakdown")
            suggestions_out = gr.Textbox(label="Suggested keywords to add")
            download_file = gr.File(label="Download cleaned resume text")

    analyze_btn.click(
        fn=analyze_resume,
        inputs=[file_input, jd_input],
        outputs=[preview_out, category_out, match_out, ats_out, ats_json, suggestions_out, download_file],
    )

if __name__ == "__main__":
    demo.launch()
