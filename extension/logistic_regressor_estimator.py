"""
Stima la threshold num_inliers per metodo4 — logistic regression P(hard).

Uso:
    python logistic_regressor_estimator.py \\
        --train-csv path/to/train_query_level.csv \\
        --val-csv   path/to/val_query_level.csv \\
        --model-out models/logreg.joblib

Output: il valore intero da inserire in THRESHOLDS di match_queries_preds.py.
        Il modello .joblib viene salvato in --model-out.

Colonne richieste nel train CSV: num_inliers_top1, correct_0
Colonne richieste nel val CSV:   num_inliers_top1, correct_0, correct_20
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib
from scipy.special import logit

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# METRICHE DI UNA POLICY
# ============================================================

def compute_metrics(p_hard, correct_0, correct_20, tau, K=20):
    rerank = p_hard > tau
    adaptive_correct = np.where(rerank, correct_20, correct_0)
    rerank_rate = rerank.mean()
    avg_matches = 1 + (K - 1) * rerank_rate
    return {
        "adaptive_R@1": adaptive_correct.mean(),
        "avg_matches":  avg_matches,
        "savings_%":    100 * (1 - avg_matches / K),
        "rerank_%":     100 * rerank_rate,
    }


# ============================================================
# TROVA N* = num_inliers dove P(hard) = best_tau
# Risolve analiticamente invertendo la sigmoide.
# ============================================================

def find_inlier_threshold(clf, best_tau):
    """
    Dalla sigmoid: logit(best_tau) = w * (N* - mean) / std + b
    → N* = (logit(best_tau) - b) / w * std + mean
    """
    logreg = clf.named_steps["logreg"]
    scaler = clf.named_steps["scaler"]

    w    = logreg.coef_[0][0]
    b    = logreg.intercept_[0]
    mean = scaler.mean_[0]
    std  = scaler.scale_[0]

    N_star = (logit(best_tau) - b) / w * std + mean
    return int(round(N_star))


# ============================================================
# ARGOMENTI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stima threshold num_inliers per metodo4 (logistic P(hard))"
    )
    parser.add_argument("--train-csv",   required=True, help="train query-level CSV")
    parser.add_argument("--val-csv",     required=True, help="validation query-level CSV")
    parser.add_argument("--model-out",   default="models/logreg.joblib",
                        help="dove salvare il modello addestrato")
    parser.add_argument("--tau-step",    type=float, default=0.01,
                        help="step per la grid search su tau")
    parser.add_argument("--max-drop-pp", type=float, default=0.1,
                        help="max calo R@1 tollerato rispetto al best (in punti percentuali)")
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main(args):
    # --- Caricamento ---
    train_df = pd.read_csv(args.train_csv).dropna(subset=["num_inliers_top1", "correct_0"])
    val_df   = pd.read_csv(args.val_csv).dropna(subset=["num_inliers_top1", "correct_0", "correct_20"])

    for df in [train_df, val_df]:
        df["num_inliers_top1"] = df["num_inliers_top1"].astype(float)
        df["correct_0"]        = df["correct_0"].astype(int)
    val_df["correct_20"] = val_df["correct_20"].astype(int)

    # hard = retrieval top-1 sbagliato
    train_df["hard"] = 1 - train_df["correct_0"]

    X_train    = train_df[["num_inliers_top1"]].values
    y_train    = train_df["hard"].values
    X_val      = val_df[["num_inliers_top1"]].values
    correct_0  = val_df["correct_0"].values
    correct_20 = val_df["correct_20"].values

    print(f"Train: {len(train_df)} query | Val: {len(val_df)} query")
    print(f"Retrieval R@1 (val):    {correct_0.mean():.4f}")
    print(f"Full rerank R@1 (val):  {correct_20.mean():.4f}")
    print(f"Hard rate (train):      {train_df['hard'].mean():.4f}")

    # --- Training ---
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)),
    ])
    clf.fit(X_train, y_train)

    coef = clf.named_steps["logreg"].coef_[0][0]
    if coef < 0:
        print("OK: P(hard) diminuisce al crescere di num_inliers.")
    else:
        print("ATTENZIONE: coefficiente positivo — controlla dati/label.")

    # Salva modello
    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    joblib.dump(clf, args.model_out)
    print(f"Modello salvato in: {args.model_out}")

    # --- Grid search su tau ---
    p_hard_val = clf.predict_proba(X_val)[:, 1]
    tau_values = np.round(np.arange(0.0, 1.0 + args.tau_step, args.tau_step), 4)

    best_r1     = -1.0
    best_tau_r1 = None

    for tau in tau_values:
        m = compute_metrics(p_hard_val, correct_0, correct_20, tau)
        if m["adaptive_R@1"] > best_r1:
            best_r1     = m["adaptive_R@1"]
            best_tau_r1 = tau

    # --- Policy più efficiente entro max_drop_pp ---
    target_r1    = best_r1 - args.max_drop_pp / 100.0
    best_tau_eff = None
    best_avg     = float("inf")

    for tau in tau_values:
        m = compute_metrics(p_hard_val, correct_0, correct_20, tau)
        if m["adaptive_R@1"] >= target_r1 and m["avg_matches"] < best_avg:
            best_avg     = m["avg_matches"]
            best_tau_eff = tau

    BEST_TAU = best_tau_eff if best_tau_eff is not None else best_tau_r1
    m_final  = compute_metrics(p_hard_val, correct_0, correct_20, BEST_TAU)

    # --- Converti in num_inliers threshold ---
    N_star = find_inlier_threshold(clf, BEST_TAU)

    print("\n" + "=" * 60)
    print("RISULTATO — metodo4 (logistic P(hard))")
    print(f"  BEST_TAU        = {BEST_TAU}")
    print(f"  Adaptive R@1    = {m_final['adaptive_R@1']:.4f}")
    print(f"  Savings         = {m_final['savings_%']:.1f}%")
    print(f"  → num_inliers threshold (N*) = {N_star}")
    print()
    print("Inserisci in match_queries_preds.py → THRESHOLDS:")
    print(f'  ("metodo4", "<vpr_method>", "<matcher>"): {N_star}')
    print("=" * 60)


if __name__ == "__main__":
    main(parse_args())
