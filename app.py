import os, csv, json
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
import plotly.express as px
import plotly
import io

from utils import allowed_file, extract_text_from_file, extract_skills_from_text, compute_age_from_dob
from engine import run_pipeline_on_applicants, level1_filter_applicants

# -----------------------
# Config
# -----------------------
app = Flask(__name__)
app.secret_key = "supersecret"

UPLOAD_FOLDER = "uploads"
DATA_FOLDER = "data"
APPLICANTS_CSV = os.path.join(DATA_FOLDER, "applicants.csv")
INTERNSHIPS_CSV = os.path.join(DATA_FOLDER, "internships_it.csv")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

FIELDS = [
    "id", "name", "dob", "age", "gender", "parent_income",
    "category", "address", "pwd", "language",
    "highest_qualification", "skills", "experience_years", "tier"
]

# -----------------------
# Helpers
# -----------------------
def write_applicant_to_csv(profile):
    file_exists = os.path.exists(APPLICANTS_CSV)
    with open(APPLICANTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(profile)

def load_applicants():
    if not os.path.exists(APPLICANTS_CSV):
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(APPLICANTS_CSV)

# -----------------------
# Routes
# -----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        name = request.form["name"]
        dob = request.form["dob"]
        gender = request.form["gender"]
        parent_income = int(request.form["parent_income"])
        category = request.form["category"]
        address = request.form["address"]
        pwd = request.form.get("pwd", "no")
        language = request.form["language"]
        highest_qualification = request.form["highest_qualification"]
        experience_years = int(request.form["experience_years"])
        tier = request.form["tier"]

        # Resume upload
        skills = []
        if "resume" in request.files:
            file = request.files["resume"]
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(path)
                text = extract_text_from_file(path)
                skills = extract_skills_from_text(text)

        age = compute_age_from_dob(dob) or 0
        df = load_applicants()
        new_id = len(df) + 1

        profile = {
            "id": new_id,
            "name": name,
            "dob": dob,
            "age": age,
            "gender": gender,
            "parent_income": parent_income,
            "category": category,
            "address": address,
            "pwd": pwd,
            "language": language,
            "highest_qualification": highest_qualification,
            "skills": ";".join(skills),
            "experience_years": experience_years,
            "tier": tier
        }

        write_applicant_to_csv(profile)
        flash("Application submitted successfully!", "success")
        return redirect(url_for("ranks"))
    return render_template("form.html")

@app.route("/ranks")
def ranks():
    df = load_applicants()
    if df.empty:
        flash("No applicants yet.", "warning")
        return redirect(url_for("form"))

    eligible = level1_filter_applicants(df)
    if eligible.empty:
        flash("No eligible applicants found.", "danger")
        return redirect(url_for("form"))

    results = run_pipeline_on_applicants(eligible, internships_path=INTERNSHIPS_CSV)
    return render_template("ranks.html", results=results)

@app.route("/download_ranks")
def download_ranks():
    df = load_applicants()
    if df.empty:
        flash("No applicants yet.", "warning")
        return redirect(url_for("ranks"))

    eligible = level1_filter_applicants(df)
    if eligible.empty:
        flash("No eligible applicants.", "warning")
        return redirect(url_for("ranks"))

    results = run_pipeline_on_applicants(eligible, internships_path=INTERNSHIPS_CSV)
    rows = []
    for r in results:
        for i, rec in enumerate(r["top_recommendations"], start=1):
            rows.append({
                "applicant_id": r["applicant_id"],
                "name": r["name"],
                "rank": i,
                "internship_id": rec["internship_id"],
                "company": rec["company"],
                "role": rec["role"],
                "similarity": rec["similarity"],
                "edu_score": rec["edu_score"],
                "final_score": rec["combined_score_final"]
            })
    df_out = pd.DataFrame(rows)

    buf = io.BytesIO()
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="latest_ranks.csv", mimetype="text/csv")

@app.route("/performance")
def performance():
    df = load_applicants()
    if df.empty:
        flash("No applicants to evaluate.", "warning")
        return redirect(url_for("form"))

    eligible = level1_filter_applicants(df)
    results = run_pipeline_on_applicants(eligible, internships_path=INTERNSHIPS_CSV)

    all_scores = []
    for r in results:
        for rec in r["top_recommendations"]:
            all_scores.append(rec["similarity"])

    if not all_scores:
        flash("No scores available for evaluation.", "warning")
        return redirect(url_for("form"))

    fig = px.histogram(x=all_scores, nbins=20, title="Distribution of Similarity Scores")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("performance.html", graphJSON=graphJSON)

@app.route("/dashboard")
def dashboard():
    df = load_applicants()
    if df.empty:
        flash("No applicants yet.", "warning")
        return redirect(url_for("form"))

    total = len(df)
    eligible = level1_filter_applicants(df)
    filtered_out = total - len(eligible)

    fig_cat = px.histogram(df, x="category", title="Applicants by Category")
    catJSON = json.dumps(fig_cat, cls=plotly.utils.PlotlyJSONEncoder)

    fig_tier = px.histogram(df, x="tier", title="Applicants by Tier")
    tierJSON = json.dumps(fig_tier, cls=plotly.utils.PlotlyJSONEncoder)

    fig_elig = px.pie(names=["Eligible", "Filtered"], values=[len(eligible), filtered_out], title="Eligibility Filtering")
    eligJSON = json.dumps(fig_elig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "dashboard.html",
        total=total, eligible=len(eligible), filtered_out=filtered_out,
        catJSON=catJSON, tierJSON=tierJSON, eligJSON=eligJSON
    )

@app.route("/about")
def about():
    return render_template("about.html")

# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
