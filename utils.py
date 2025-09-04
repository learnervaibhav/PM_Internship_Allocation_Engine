import os, re
import pdfplumber
from docx import Document
from datetime import datetime

ALLOWED_EXT = {".pdf", ".docx", ".doc", ".txt"}

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

COMMON_IT_SKILLS = [
    "python","java","c++","javascript","react","angular","node","sql","mysql","postgres",
    "mongodb","aws","azure","gcp","docker","kubernetes","git","html","css","flask","django","spring",
    "tensorflow","pytorch","keras","nlp","pandas","numpy","scikit-learn","xgboost","lightgbm"
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

def compute_age_from_dob(dob_str):
    try:
        for fmt in ("%Y-%m-%d","%d-%m-%Y","%d/%m/%Y","%Y/%m/%d"):
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
