import os
import io
import csv
import json
import joblib
import traceback
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly
import lightgbm as lgb

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename

# local modules
import utils
import engine

# -----------------------
# Paths & folders
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

APPLICANTS_CSV = DATA_DIR / "Applicants.csv"
INTERNSHIPS_CSV = DATA_DIR / "Internships.csv"
MATCHES_CSV = DATA_DIR / "Matches.csv"

# Candidate model artifacts (prefer LTR text model, else sklearn joblib)
LGB_LTR_PATH = MODELS_DIR / "ltr_model.txt"
SKL_MODEL_PATH = MODELS_DIR / "best_model.pkl"
ART_PATH = MODELS_DIR / "artifacts.joblib"

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.secret_key = "vaibhav"

# -----------------------
# Load artifacts / models (safe)
# -----------------------
lgb_ltr = None
skl_model = None
artifacts = None

# try to load artifacts.joblib (contains feature_cols or precomputed embeddings)
if ART_PATH.exists():
    try:
        artifacts = joblib.load(str(ART_PATH))
    except Exception as e:
        app.logger.warning("Could not load artifacts.joblib: %s", e)

# try ltr (LightGBM saved text)
if LGB_LTR_PATH.exists():
    try:
        lgb_ltr = lgb.Booster(model_file=str(LGB_LTR_PATH))
        app.logger.info("Loaded LTR LightGBM model from %s", LGB_LTR_PATH)
    except Exception as e:
        app.logger.warning("Failed loading ltr_model.txt: %s", e)

# try sklearn-like model
if SKL_MODEL_PATH.exists():
    try:
        skl_model = joblib.load(str(SKL_MODEL_PATH))
        app.logger.info("Loaded sklearn model from %s", SKL_MODEL_PATH)
    except Exception as e:
        app.logger.warning("Failed loading best_model.pkl: %s", e)

# feature cols - prefer utils.feature_cols (already in your utils.py)
FEATURE_COLS = getattr(utils, "feature_cols", None)
if FEATURE_COLS is None and artifacts and "feature_cols" in artifacts:
    FEATURE_COLS = artifacts["feature_cols"]

if FEATURE_COLS is None:
    # fallback default (must match training order)
    FEATURE_COLS = ["emb_sim", "skill_overlap", "qualification_num", "fresher_flag", "low_income"]
    app.logger.warning("feature_cols not found in utils/artifacts - using fallback FEATURE_COLS")

# -----------------------
# Helper functions for CSVs
# -----------------------
def load_applicants_df():
    if not APPLICANTS_CSV.exists():
        return pd.DataFrame(columns=[
            "ApplicantID","Name","DOB","Age","Gender","Parent_Income","Category",
            "Address","PWD","Language","Qualification","Skills","Experience","Tier","ResumeText"
        ])
    return pd.read_csv(APPLICANTS_CSV, on_bad_lines="skip")

def write_applicant(profile: dict):
    file_exists = APPLICANTS_CSV.exists()
    with open(APPLICANTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(profile.keys()), quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()
        writer.writerow(profile)

def load_internships_df():
    if not INTERNSHIPS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(INTERNSHIPS_CSV)

# -----------------------
# Routes
# -----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/apply", methods=["GET","POST"])
def apply():
    if request.method == "POST":
        try:
            form = request.form
            name = form.get("name", "Unknown")
            dob = form.get("dob", "")
            age = utils.compute_age_from_dob(dob) or form.get("age", "")
            # resume upload
            skills_found = []
            resume_text = ""
            if "resume" in request.files:
                f = request.files["resume"]
                if f and utils.allowed_file(f.filename):
                    filename = secure_filename(f.filename)
                    path = UPLOADS_DIR / filename
                    f.save(str(path))
                    resume_text = utils.extract_text_from_file(str(path))
                    skills_found = utils.extract_skills_from_text(resume_text)

            profile = {
                "ApplicantID": form.get("applicant_id") or str(len(load_applicants_df())+1),
                "Name": name,
                "DOB": dob,
                "Age": age,
                "Gender": form.get("gender",""),
                "Parent_Income": form.get("parent_income","0"),
                "Category": form.get("category","General"),
                "Address": form.get("address",""),
                "PWD": form.get("pwd","No"),
                "Language": form.get("language",""),
                "Qualification": form.get("qualification",""),
                "Skills": ";".join(skills_found) if skills_found else form.get("skills",""),
                "Experience": form.get("experience","0"),
                "Tier": form.get("tier","T1"),
                "ResumeText": resume_text
            }
            write_applicant(profile)
            flash("Application submitted.", "success")
            return redirect(url_for("index"))
        except Exception as e:
            app.logger.error("Error in /apply: %s\n%s", e, traceback.format_exc())
            flash("Failed to submit application. Check server logs.", "danger")
            return redirect(url_for("apply"))
    return render_template("apply.html")

@app.route("/ranks")
def ranks_index():
    # show internships (simple listing)
    ints = load_internships_df()
    return render_template("internships_list.html", internships=ints.to_dict(orient="records"))

@app.route("/ranks/<internship_id>")
def ranks_for_internship(internship_id):
    interns_df = load_internships_df()
    apps_df = load_applicants_df()
    if interns_df.empty:
        flash("No internships data available.", "warning")
        return redirect(url_for("index"))
    if apps_df.empty:
        flash("No applicants available.", "warning")
        return redirect(url_for("apply"))

    # find internship row
    try:
        row = interns_df.loc[interns_df["InternshipID"] == internship_id].iloc[0]
    except Exception:
        flash("Internship ID not found.", "warning")
        return redirect(url_for("ranks_index"))

    # filter by sector if column exists
    candidates = apps_df.copy()
    if "Sector" in interns_df.columns and "Sector" in apps_df.columns:
        candidates = candidates[candidates["Sector"] == row.get("Sector", candidates["Sector"].iloc[0])]

    # Compute features (vectorized via utils)
    try:
        # utils.compute_features_for_pair returns a numpy array - we use apply to get rows
        feats_df = candidates.apply(lambda r: utils.compute_features_for_pair(r, row), axis=1, result_type="expand")
        feats_df.columns = FEATURE_COLS
    except Exception:
        # fallback: use engine scoring if features utility fails
        app.logger.warning("Feature computation failed; falling back to engine scoring.")
        results = engine.run_pipeline_on_applicants(candidates)
        return render_template("ranks_fallback.html", internship=row.to_dict(), results=results)

    # Score using model (prefer sklearn-like, else ltr BoosteR or fallback to engine)
    scores = None
    if skl_model is not None:
        try:
            scores = skl_model.predict_proba(feats_df.values)[:,1]
        except Exception as e:
            app.logger.warning("skl_model predict_proba failed: %s", e)
            scores = None
    if scores is None and lgb_ltr is not None:
        try:
            # LightGBM Booster expects dataset values
            scores = lgb_ltr.predict(feats_df.values)
        except Exception as e:
            app.logger.warning("lgb_ltr predict failed: %s", e)
            scores = None
    if scores is None:
        # fallback scoring: normalized emb_sim if present
        if "emb_sim" in feats_df.columns:
            s = feats_df["emb_sim"].values.astype(float)
            scores = (s - s.min()) / max(1e-12, (s.max() - s.min()))
        else:
            # final fallback: engine scoring per applicant (slow)
            results = engine.run_pipeline_on_applicants(candidates, internships_path=str(INTERNSHIPS_CSV))
            return render_template("ranks_fallback.html", internship=row.to_dict(), results=results)

    # attach scores and sort
    candidates = candidates.assign(score=scores)
    ranked = candidates.sort_values("score", ascending=False).head(200)
    return render_template("ranks.html", internship=row.to_dict(), candidates=ranked.to_dict(orient="records"))

@app.route("/download_ranks/<internship_id>")
def download_ranks(internship_id):
    interns_df = load_internships_df()
    apps_df = load_applicants_df()
    try:
        row = interns_df.loc[interns_df["InternshipID"] == internship_id].iloc[0]
    except Exception:
        flash("Internship ID not found.", "warning")
        return redirect(url_for("ranks_index"))

    candidates = apps_df.copy()
    if "Sector" in interns_df.columns and "Sector" in apps_df.columns:
        candidates = candidates[candidates["Sector"] == row.get("Sector", candidates["Sector"].iloc[0])]

    feats_df = candidates.apply(lambda r: utils.compute_features_for_pair(r, row), axis=1, result_type="expand")
    feats_df.columns = FEATURE_COLS

    # score
    scores = None
    if skl_model is not None:
        try:
            scores = skl_model.predict_proba(feats_df.values)[:,1]
        except Exception:
            scores = None
    if scores is None and lgb_ltr is not None:
        try:
            scores = lgb_ltr.predict(feats_df.values)
        except Exception:
            scores = None
    if scores is None and "emb_sim" in feats_df.columns:
        s = feats_df["emb_sim"].values.astype(float)
        scores = (s - s.min()) / max(1e-12, (s.max() - s.min()))
    if scores is None:
        flash("Unable to compute scores at this time.", "danger")
        return redirect(url_for("ranks_index"))

    candidates = candidates.assign(score=scores)
    ranked = candidates.sort_values("score", ascending=False)
    buf = io.StringIO()
    ranked.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()), as_attachment=True, download_name=f"ranks_{internship_id}.csv", mimetype="text/csv")

@app.route("/dashboard")
def dashboard():
    df = load_applicants_df()
    if df.empty:
        flash("No applicants to show.", "warning")
        return redirect(url_for("apply"))

    total = len(df)
    eligible = engine.level1_filter_applicants(df) if hasattr(engine, "level1_filter_applicants") else df
    filtered_out = total - len(eligible)

    fig_cat = px.histogram(df, x="Category", title="Applicants by Category")
    catJSON = json.dumps(fig_cat, cls=plotly.utils.PlotlyJSONEncoder)

    fig_tier = px.histogram(df, x="Tier", title="Applicants by Tier")
    tierJSON = json.dumps(fig_tier, cls=plotly.utils.PlotlyJSONEncoder)

    fig_elig = px.pie(names=["Eligible","Filtered"], values=[len(eligible), filtered_out], title="Eligibility Filtering")
    eligJSON = json.dumps(fig_elig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("dashboard.html", total=total, eligible=len(eligible), filtered_out=filtered_out,
                           catJSON=catJSON, tierJSON=tierJSON, eligJSON=eligJSON)

@app.route("/about")
def about():
    # you can read your PDF and populate content here if needed
    return render_template("about.html")

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
