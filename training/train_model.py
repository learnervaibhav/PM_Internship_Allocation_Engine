# training/train_model.py
import os, joblib, json, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, train_test_split
import optuna
from sklearn.metrics import ndcg_score

DATA_DIR = r"E:\Vaibhav_Baranwal\Vaibhav_Projects\pm_internship_ai\data"
MODEL_DIR = r"E:\Vaibhav_Baranwal\Vaibhav_Projects\pm_internship_ai\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1) load
A = pd.read_csv(os.path.join(DATA_DIR,"Applicants.csv"))
I = pd.read_csv(os.path.join(DATA_DIR,"Internships.csv"))
M = pd.read_csv(os.path.join(DATA_DIR,"Matches.csv"))

# 2) prepare lookup maps
app_map = A.set_index("ApplicantID").to_dict(orient="index")
int_map = I.set_index("InternshipID").to_dict(orient="index")

# 3) feature engineering helper
def normalize_skills(s): return [] if pd.isna(s) else [x.strip().lower() for x in str(s).split(",")]
A["skills_list"] = A["Skills"].apply(normalize_skills)
I["req_skills_list"] = I["Required_Skills"].apply(normalize_skills)

# 4) embeddings (all applicants & internships once)
model = SentenceTransformer("all-MiniLM-L6-v2")
# applicant text concat
unique_apps = A["ApplicantID"].tolist()
app_texts = ["; ".join(A.loc[A["ApplicantID"]==aid,"skills_list"].values[0]) for aid in unique_apps]
app_embs = model.encode(app_texts, convert_to_numpy=True, normalize_embeddings=True)
app_emb_map = dict(zip(unique_apps, app_embs.tolist()))

unique_ints = I["InternshipID"].tolist()
int_texts = ["; ".join(I.loc[I["InternshipID"]==iid,"req_skills_list"].values[0]) for iid in unique_ints]
int_embs = model.encode(int_texts, convert_to_numpy=True, normalize_embeddings=True)
int_emb_map = dict(zip(unique_ints, int_embs.tolist()))

# 5) build features for matches 
def emb_sim(aid, iid):
    a = np.array(app_emb_map.get(aid))
    b = np.array(int_emb_map.get(iid))
    return float(np.dot(a,b))

def overlap(aid, iid):
    app_skills_str = app_map[aid].get("Skills", "")
    int_skills_str = int_map[iid].get("Required_Skills", "")
    if pd.isna(app_skills_str) or pd.isna(int_skills_str):
        return 0.0
    a = set(str(app_skills_str).lower().split(", "))
    b = set(str(int_skills_str).lower().split(", "))
    return len(a & b)/max(1,len(b))

M["emb_sim"] = M.apply(lambda r: emb_sim(r["ApplicantID"], r["InternshipID"]), axis=1)
M["skill_overlap"] = M.apply(lambda r: overlap(r["ApplicantID"], r["InternshipID"]), axis=1)
# additional features
qual_map = {"10th":1,"12th":2,"Diploma":3,"Graduation":4,"Post-Graduation":5}
A_q = A.set_index("ApplicantID")["Qualification"].map(qual_map).to_dict()
M["qualification_num"] = M["ApplicantID"].map(lambda x: A_q.get(x,2))
M["fresher_flag"] = (M["Experience"]==0).astype(int)
M["low_income"] = M["ApplicantID"].map(lambda x: 1 if app_map[x]["Parent_Income"]<=800000 else 0)

feature_cols = ["emb_sim","skill_overlap","qualification_num","fresher_flag","low_income"]

# 6) Two-stage retrieval: keep top-K per internship by emb_sim
TOP_K = 150
retrieved = M.groupby("InternshipID", group_keys=False).apply(lambda g: g.nlargest(TOP_K, "emb_sim")).reset_index(drop=True)

# 7) Optuna objective (GroupKFold by InternshipID)
def objective(trial):
    params = {
        "objective":"lambdarank",
        "metric":"ndcg",
        "ndcg_eval_at":[5],
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 128),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200),
        "feature_fraction": trial.suggest_float("feature_fraction",0.6,1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction",0.6,1.0),
        "bagging_freq": trial.suggest_int("bagging_freq",1,7),
        "lambda_l1": trial.suggest_float("lambda_l1",1e-8,10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2",1e-8,10.0, log=True),
        "verbosity": -1,
        "seed": 42
    }
    df = retrieved.copy()
    gkf = GroupKFold(n_splits=3)
    ndcgs=[]
    for train_idx, val_idx in gkf.split(df, df["Selected"], df["InternshipID"]):
        tr = df.iloc[train_idx]; va = df.iloc[val_idx]
        dtr = lgb.Dataset(tr[feature_cols].values, label=tr["Selected"].values,
                          group=tr.groupby("InternshipID").size().to_list())
        dva = lgb.Dataset(va[feature_cols].values, label=va["Selected"].values,
                          group=va.groupby("InternshipID").size().to_list(), reference=dtr)
        bst = lgb.train(params, dtr, valid_sets=[dva], num_boost_round=2000, early_stopping_rounds=50, verbose_eval=False)
        preds = bst.predict(va[feature_cols].values, num_iteration=bst.best_iteration)
        # compute mean NDCG@5 per internship in val set
        for iid, grp in va.groupby("InternshipID"):
            if len(grp)<2: continue
            y_true = grp["Selected"].values.reshape(1,-1)
            y_score = preds[grp.index].reshape(1,-1)
            ndcgs.append(ndcg_score(y_true, y_score, k=5))
    return np.mean(ndcgs) if len(ndcgs)>0 else 0.0

# 8) run optuna (small number of trials here; increase when you run locally)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, show_progress_bar=True)

best_params = study.best_params
print("Best params:", best_params)

# 9) Train final model on all retrieved with best params
final_params = best_params.copy()
final_params.update({"objective":"lambdarank","metric":"ndcg","ndcg_eval_at":[5],"verbosity":-1,"seed":42})
group_sizes = retrieved.groupby("InternshipID").size().to_list()
dall = lgb.Dataset(retrieved[feature_cols].values, label=retrieved["Selected"].values, group=group_sizes)
final_model = lgb.train(final_params, dall, num_boost_round=500)

# 10) save model + artifacts
final_model.save_model(os.path.join(MODEL_DIR,"ltr_model.txt"))
joblib.dump({"app_emb_map":app_emb_map, "int_emb_map":int_emb_map, "feature_cols":feature_cols}, os.path.join(MODEL_DIR,"artifacts.joblib"))
print("Saved model and artifacts")
