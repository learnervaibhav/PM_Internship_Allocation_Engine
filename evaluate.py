import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(y_true, y_pred, threshold=0.5):
    preds = (np.array(y_pred) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
    }

def ndcg_at_k(y_true, y_scores, k=5):
    order = np.argsort(y_scores)[::-1]
    y_true = np.take(y_true, order[:k])
    dcg = np.sum((2**y_true - 1) / np.log2(np.arange(2, len(y_true)+2)))
    idcg = np.sum((2**sorted(y_true, reverse=True) - 1) / np.log2(np.arange(2, len(y_true)+2)))
    return dcg/idcg if idcg > 0 else 0.0
