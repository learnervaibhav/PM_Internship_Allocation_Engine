import os, re
import pdfplumber
from docx import Document
from datetime import datetime
import joblib, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------
# Config
# -----------------------
ALLOWED_EXT = {".pdf", ".docx", ".doc", ".txt"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# load artifacts
art_path = os.path.join(MODEL_DIR, "artifacts.joblib")
art = joblib.load(art_path)

feature_cols = art["feature_cols"]

# embedder
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------
# Feature Engineering
# -----------------------
def emb_sim_from_text(app_skills_txt, int_req_txt):
    a = model.encode(app_skills_txt or "", convert_to_numpy=True, normalize_embeddings=True)
    b = model.encode(int_req_txt or "", convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(a, b))

def compute_features_for_pair(app_row, int_row):
    # skill similarity
    emb_sim = emb_sim_from_text(
        app_row.get("Skills", ""),
        int_row.get("Required_Skills", "")
    )

    # overlap
    app_skills = [s.strip().lower() for s in str(app_row.get("Skills", "")).split(",") if s.strip()]
    int_skills = [s.strip().lower() for s in str(int_row.get("Required_Skills", "")).split(",") if s.strip()]
    skill_overlap = len(set(app_skills) & set(int_skills)) / max(1, len(int_skills))

    # qualification score
    qual_map = {"10th": 1, "12th": 2, "Diploma": 3, "Graduation": 4, "Post-Graduation": 5}
    qualification_num = qual_map.get(str(app_row.get("Qualification", "")).strip(), 2)

    # fresher flag
    fresher_flag = 1 if int(app_row.get("Experience", 0)) == 0 else 0

    # low income
    low_income = 1 if float(app_row.get("Parent_Income", 1e9)) <= 800000 else 0

    row = {
        "emb_sim": emb_sim,
        "skill_overlap": skill_overlap,
        "qualification_num": qualification_num,
        "fresher_flag": fresher_flag,
        "low_income": low_income
    }
    return np.array([row[c] for c in feature_cols])

# -----------------------
# File Utilities
# -----------------------
def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT

def extract_text_from_pdf(path):
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
    return "\n".join(texts)

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in (".docx", ".doc"):
        return extract_text_from_docx(path)
    return extract_text_from_txt(path)

# -----------------------
# Skill Extraction
# -----------------------
COMMON_IT_SKILLS = [
    "python", "java", "c++", "javascript", "react", "angular", "node", "sql", "mysql", "postgres",
    "mongodb", "aws", "azure", "gcp", "docker", "kubernetes", "git", "html", "css", "flask", "django", "spring",
    "tensorflow", "pytorch", "keras", "nlp", "pandas", "numpy", "scikit-learn", "xgboost", "lightgbm"
]

def extract_skills_from_text(text):
    if not text:
        return []
    text_low = text.lower()
    found = set()
    for sk in COMMON_IT_SKILLS:
        if re.search(r'\b' + re.escape(sk) + r'\b', text_low):
            found.add(sk)
    return sorted(found)

# -----------------------
# Age from DOB
# -----------------------
def compute_age_from_dob(dob_str):
    try:
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(dob_str, fmt)
                break
            except:
                dt = None
        if dt is None:
            return None
        today = datetime.today()
        age = today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day))
        return age
    except:
        return None
