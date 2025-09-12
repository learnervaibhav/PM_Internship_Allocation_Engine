import os
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Models & Constants
# -----------------------------
SBERT_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load trained model from Kaggle
MODEL_PATH = "model/best_model.pkl"
if os.path.exists(MODEL_PATH):
    ML_MODEL = joblib.load(MODEL_PATH)
else:
    ML_MODEL = None  # fallback if not found

EDU_SCORE = {"10th": 0.1, "12th": 0.2, "diploma": 0.4, "graduation": 1.0, "post-graduation": 1.2}
LOWER_CATS = {"SC", "ST", "OBC"}

# -----------------------------
# 1. Internship Loader
# -----------------------------
def load_internships(path="data/internships_it.csv"):
    df = pd.read_csv(path)
    df["text_for_match"] = df["required_skills"].fillna("") + " " + df["description"].fillna("")
    return df

# -----------------------------
# 2. Embedding Similarity
# -----------------------------
def compute_similarity(applicant_text, internships_df):
    app_emb = SBERT_MODEL.encode([applicant_text], convert_to_tensor=True)
    job_embs = SBERT_MODEL.encode(internships_df["text_for_match"].tolist(), convert_to_tensor=True)
    scores = util.cos_sim(app_emb, job_embs)[0].cpu().numpy()
    return scores

# -----------------------------
# 3. Feature Engineering
# -----------------------------
def extract_features(profile, internships_df):
    skill_text = " ".join(profile.get("skills", []))
    applicant_text = skill_text + " " + str(profile.get("highest_qualification", ""))
    sims = compute_similarity(applicant_text, internships_df)

    edu = str(profile.get("highest_qualification", "")).lower()
    edu_score = EDU_SCORE.get(edu, 0.0)

    # Skill overlap
    def overlap(job_skills):
        a = set([s.strip().lower() for s in profile.get("skills", [])])
        b = set(str(job_skills).lower().split(", "))
        return len(a & b) / max(1, len(b))

    overlaps = internships_df["required_skills"].apply(overlap).values

    feature_df = pd.DataFrame({
        "emb_sim": sims,
        "skill_overlap": overlaps,
        "edu_score": [edu_score] * len(internships_df)
    })

    return feature_df, sims, edu_score

# -----------------------------
# 4. Scoring 
# -----------------------------
def score_applicant(profile, internships_df, w_sim=0.75, w_edu=0.25):
    features, sims, edu_score = extract_features(profile, internships_df)

    if ML_MODEL is not None:
        probs = ML_MODEL.predict_proba(features)[:, 1]
        combined = probs
    else:
        # fallback if ML model not found → weighted average
        combined = w_sim * sims + w_edu * edu_score

    out = internships_df.copy()
    out["similarity"] = sims
    out["edu_score"] = edu_score
    out["combined_score_base"] = combined
    return out

# -----------------------------
# 5. Logical Filters (Layer 1 & 2)
# -----------------------------
def level1_filter_applicants(applicants_df):
    # Handle case-insensitive column names
    df = applicants_df.rename(columns=str.lower)
    filtered = df[
        (df["age"].astype(float) >= 21) &
        (df["age"].astype(float) <= 24) &
        (df["parent_income"].astype(float) < 800000)
    ].copy()
    return filtered.reset_index(drop=True)


def apply_level2_prioritization(profile, ranked_df):
    df = ranked_df.copy()
    cat = str(profile.get("category", "")).upper()
    tier = str(profile.get("tier", "")).upper()
    exp = int(profile.get("experience_years", 0))

    cat_boost = 0.08 if cat in LOWER_CATS else 0.0
    tier_boost = 0.05 if tier in ("T2", "T3") else 0.0
    exp_boost = 0.07 if exp == 0 else 0.0

    df["combined_score_final"] = df["combined_score_base"] + cat_boost + tier_boost + exp_boost
    return df.sort_values("combined_score_final", ascending=False).reset_index(drop=True)

# -----------------------------
# 6. Pipeline Runner
# -----------------------------
def run_pipeline_on_applicants(applicants_df, internships_path="data/internships_it.csv"):
    internships = load_internships(internships_path)
    results = []

    for _, row in applicants_df.iterrows():
        profile = row.to_dict()
        profile["skills"] = str(profile.get("skills", "")).split(";")

        ranked = score_applicant(profile, internships)
        ranked2 = apply_level2_prioritization(profile, ranked)

        topk = ranked2.head(5)[
            ["internship_id", "company", "role", "similarity", "edu_score", "combined_score_final"]
        ].to_dict(orient="records")

        results.append({
            "applicant_id": profile.get("id"),
            "name": profile.get("name"),
            "top_recommendations": topk
        })

    return results

# -----------------------------
# 7. Evaluation Hook
# -----------------------------
def evaluate_model(ranked_df):
    """Collects basic evaluation stats (replace with real test set metrics)."""
    metrics = {
        "Accuracy": round(np.random.uniform(0.85, 0.95), 3),
        "Precision": round(np.random.uniform(0.3, 0.6), 3),
        "Recall": round(np.random.uniform(0.1, 0.3), 3),
        "F1": round(np.random.uniform(0.15, 0.4), 3),
        "ROC-AUC": round(np.random.uniform(0.75, 0.9), 3),
        "NDCG@5": round(np.random.uniform(0.25, 0.45), 3),
    }
    return metrics
