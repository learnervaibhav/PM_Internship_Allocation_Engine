import pandas as pd
from sentence_transformers import SentenceTransformer, util

MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

EDU_SCORE = {"10th": 0.1, "12th": 0.2, "diploma": 0.4, "graduation": 1.0}
LOWER_CATS = {"SC","ST","OBC"}

def load_internships(path="data/internships_it.csv"):
    df = pd.read_csv(path)
    df["text_for_match"] = df["required_skills"].fillna("") + " " + df["description"].fillna("")
    return df

def compute_similarity(applicant_text, internships_df):
    app_emb = MODEL.encode([applicant_text], convert_to_tensor=True)
    job_embs = MODEL.encode(internships_df["text_for_match"].tolist(), convert_to_tensor=True)
    scores = util.cos_sim(app_emb, job_embs)[0].cpu().numpy()
    return scores

def score_applicant(profile, internships_df, w_sim=0.75, w_edu=0.25):
    skill_text = " ".join(profile.get("skills", []))
    applicant_text = skill_text + " " + str(profile.get("highest_qualification",""))
    sims = compute_similarity(applicant_text, internships_df)
    edu = profile.get("highest_qualification","").lower()
    edu_score = EDU_SCORE.get(edu, 0.0)
    combined = w_sim * sims + w_edu * edu_score
    out = internships_df.copy()
    out["similarity"] = sims
    out["edu_score"] = edu_score
    out["combined_score_base"] = combined
    return out

def level1_filter_applicants(applicants_df):
    filtered = applicants_df[
        (applicants_df["age"] >= 21) &
        (applicants_df["age"] <= 24) &
        (applicants_df["parent_income"] < 800000)
    ].copy()
    return filtered.reset_index(drop=True)

def apply_level2_prioritization(profile, ranked_df):
    df = ranked_df.copy()
    cat = str(profile.get("category","")).upper()
    tier = str(profile.get("tier","")).upper()
    exp = int(profile.get("experience_years",0))
    cat_boost = 0.08 if cat in LOWER_CATS else 0.0
    tier_boost = 0.05 if tier in ("T2","T3") else 0.0
    exp_boost = 0.07 if exp == 0 else 0.0
    df["combined_score_final"] = df["combined_score_base"] + cat_boost + tier_boost + exp_boost
    return df.sort_values("combined_score_final", ascending=False).reset_index(drop=True)

def run_pipeline_on_applicants(applicants_df, internships_path="data/internships_it.csv"):
    internships = load_internships(internships_path)
    results = []
    for _, row in applicants_df.iterrows():
        profile = row.to_dict()
        profile["skills"] = str(profile.get("skills","")).split(";")
        ranked = score_applicant(profile, internships)
        ranked2 = apply_level2_prioritization(profile, ranked)
        topk = ranked2.head(5)[["internship_id","company","role","similarity","edu_score","combined_score_final"]].to_dict(orient="records")
        results.append({"applicant_id": profile.get("id"), "name": profile.get("name"), "top_recommendations": topk})
    return results
