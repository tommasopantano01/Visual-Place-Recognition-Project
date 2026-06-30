"""
_calibrate_common.py — Motore condiviso dagli script di calibrazione
(quelli che generano threshold.csv / model.json), separato da _common.py
che invece e' il motore di INFERENZA.

Mai eseguito a deployment: questi script girano una volta sola, offline,
su train/val CSV etichettati, e scrivono i file che poi _common.py legge.
"""

import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONVERSIONE CSV CANDIDATE-LEVEL -> QUERY-LEVEL
#
# Formato candidate-level (una riga per coppia query-candidato):
#   query_id, candidate_path, l2_distance, retrieval_rank,
#   num_inliers, rerank_rank_topK, is_positive, K
# ============================================================

REQUIRED_CANDIDATE_COLS = [
    "query_id", "retrieval_rank", "num_inliers",
    "rerank_rank_topK", "is_positive", "l2_distance",
]


def compute_su(l2_distances, k=10, alpha=0.5, eps=1e-12):
    """RS/SD/SU dalle distanze L2 dei top-k candidati (gia' ordinati per retrieval_rank)."""
    d = np.asarray(l2_distances, dtype=float)[:k]
    s = 1.0 / (1.0 + d)
    if len(s) < 2:
        return 0.0, 0.0, 0.0
    rs = float(np.mean(s[1:] / (s[0] + eps)))
    sd = float(np.median(s) / (np.max(s) + eps))
    return rs, sd, alpha * rs + (1 - alpha) * sd


def _winner_at_budget(group, budget):
    """
    Candidato vincente se il rerank fosse limitato ai primi `budget`
    candidati per retrieval_rank (serve per sequential/, dove servono
    c_5 e c_10 oltre a c_0/c_20).
    """
    sub = group[group["retrieval_rank"] <= budget]
    if len(sub) == 0:
        return None
    return sub.sort_values(["num_inliers", "retrieval_rank"], ascending=[False, True]).iloc[0]


def load_query_level(csv_path, k_su=10, alpha=0.5, budgets=(5, 10)):
    """
    Converte un CSV candidate-level in query-level, una riga per query, con:
      num_inliers_top1, correct_0, correct_20, helps_20, hurts_20, RS, SD, SU
    Per ogni budget in `budgets` aggiunge anche:
      correct_<budget>, max_inliers_top<budget>
    (servono solo a sequential_calibrate.py, ignorabili per gli altri metodi).
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_CANDIDATE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne candidate-level mancanti: {missing}")

    df["query_id"]         = df["query_id"].astype(str)
    df["retrieval_rank"]   = df["retrieval_rank"].astype(int)
    df["num_inliers"]      = df["num_inliers"].astype(float)
    df["rerank_rank_topK"] = df["rerank_rank_topK"].astype(int)
    df["is_positive"]      = df["is_positive"].astype(int)
    df["l2_distance"]      = df["l2_distance"].astype(float)

    rows = []
    for query_id, group in df.groupby("query_id"):
        top1 = group[group["retrieval_rank"] == 1]
        if len(top1) == 0:
            continue
        top1 = top1.iloc[0]

        rerank_winner = group[group["rerank_rank_topK"] == 1]
        if len(rerank_winner) == 0:
            rerank_winner = group.sort_values(
                ["num_inliers", "retrieval_rank"], ascending=[False, True]
            ).head(1)
        rerank_winner = rerank_winner.iloc[0]

        correct_0  = int(top1["is_positive"])
        correct_20 = int(rerank_winner["is_positive"])

        ordered = group.sort_values("retrieval_rank")
        rs, sd, su = compute_su(ordered["l2_distance"].values, k=k_su, alpha=alpha)

        row = {
            "query_id":         query_id,
            "num_inliers_top1": float(top1["num_inliers"]),
            "correct_0":        correct_0,
            "correct_20":       correct_20,
            "helps_20":         int(correct_0 == 0 and correct_20 == 1),
            "hurts_20":         int(correct_0 == 1 and correct_20 == 0),
            "RS": rs, "SD": sd, "SU": su,
        }

        for b in budgets:
            winner_b = _winner_at_budget(group, b)
            if winner_b is not None:
                row[f"correct_{b}"]         = int(winner_b["is_positive"])
                row[f"max_inliers_top{b}"]  = float(
                    group[group["retrieval_rank"] <= b]["num_inliers"].max()
                )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# METRICHE DI UNA POLICY (su num_inliers o su probability, stessa formula)
# ============================================================

def compute_metrics(rerank_mask, correct_0, correct_20, K=20):
    adaptive_correct = np.where(rerank_mask, correct_20, correct_0)
    rerank_rate = rerank_mask.mean()
    avg_matches = 1 + (K - 1) * rerank_rate
    return {
        "adaptive_R@1": adaptive_correct.mean(),
        "avg_matches":  avg_matches,
        "savings_%":    100 * (1 - avg_matches / K),
    }


# ============================================================
# REGRESSIONE LOGISTICA (training)
# ============================================================

def fit_logistic(X, y):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)),
    ])
    clf.fit(X, y)
    return clf


def regressor_to_dict(clf, feat_cols):
    """Estrae i coefficienti in un dict JSON-able, stesso formato usato da _common.apply_sigmoid."""
    scaler = clf.named_steps["scaler"]
    logreg = clf.named_steps["logreg"]
    return {
        "feat_cols":    list(feat_cols),
        "scaler_mean":  scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef":         logreg.coef_.tolist(),
        "intercept":    logreg.intercept_.tolist(),
    }


# ============================================================
# SALVATAGGIO threshold.csv / model.json
# ============================================================

def save_threshold_csv(path, **fields):
    """Scrive threshold.csv con un'intestazione e una riga di valori."""
    path = Path(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields.keys())
        writer.writerow(fields.values())
    print(f"Salvato: {path}  ({fields})")


def save_model_json(path, model_dict):
    """Scrive model.json. model_dict puo' essere un singolo regressore o
    un dict di piu' regressori (es. {"help": {...}, "hurt": {...}})."""
    path = Path(path)
    with open(path, "w") as f:
        json.dump(model_dict, f, indent=2)
    print(f"Salvato: {path}")
