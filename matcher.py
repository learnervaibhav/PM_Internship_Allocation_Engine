# matcher.py
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load pretrained SBERT model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts):
    """Convert list of texts to embeddings"""
    return model.encode(texts, convert_to_tensor=True)

def recommend(resume_text: str, internships_df: pd.DataFrame, top_k: int = 5):
    """
    Recommend internships for a given resume text.
    """
    # Encode applicant resume
    resume_emb = embed_texts([resume_text])

    # Encode internship descriptions
    job_texts = internships_df["description"].fillna("").tolist()
    job_embs = embed_texts(job_texts)

    # Compute cosine similarity
    scores = util.pytorch_cos_sim(resume_emb, job_embs)[0].cpu().numpy()

    # Attach scores
    internships_df = internships_df.copy()
    internships_df["score"] = scores

    # Sort by score and return top_k
    return internships_df.sort_values("score", ascending=False).head(top_k)
