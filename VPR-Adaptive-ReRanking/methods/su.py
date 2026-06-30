"""
su/su_calibrate.py — Allena P(help | SU) e calibra tau su validation.
Scrive threshold.csv + model.json.

SCELTA DA CONFERMARE: il notebook originale di Rocco esplorava 3 target
(hard, help, hurts) per ogni feature set. Qui si calibra solo su "help",
per coerenza con logistic_help/ e con quanto discusso ("stimare P_help è
più informativo anche se più difficile da stimare"). Se Rocco intende un
target diverso come riferimento per SU, vanno aggiornati X_train/y_train.

Uso:
    python VPR-adaptive-re-ranking/su/su_calibrate.py \
        --train-csv train.csv --val-csv val.csv
"""
import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _calibrate_common import (
    load_query_level, compute_metrics, fit_logistic, regressor_to_dict,
    save_threshold_csv, save_model_json,
)

_THRESHOLD_CSV = Path(__file__).parent / "threshold.csv"
_MODEL_JSON    = Path(__file__).parent / "model.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Calibra logistic P(help|SU) — su (Rocco)")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv",   required=True)
    parser.add_argument("--su-k",      type=int, default=10)
    parser.add_argument("--su-alpha",  type=float, default=0.5)
    parser.add_argument("--tau-step",  type=float, default=0.01)
    parser.add_argument("--max-drop-pp", type=float, default=0.1)
    return parser.parse_args()


def main(args):
    train_df = load_query_level(args.train_csv, k_su=args.su_k, alpha=args.su_alpha)
    val_df   = load_query_level(args.val_csv,   k_su=args.su_k, alpha=args.su_alpha)

    X_train = train_df[["SU"]].values
    y_train = train_df["helps_20"].values
    X_val   = val_df[["SU"]].values
    correct_0, correct_20 = val_df["correct_0"].values, val_df["correct_20"].values

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | helps rate train: {train_df['helps_20'].mean():.3f}")

    clf = fit_logistic(X_train, y_train)
    p_val = clf.predict_proba(X_val)[:, 1]

    tau_values = np.round(np.arange(0.0, 1.0 + args.tau_step, args.tau_step), 4)
    best_r1, best_tau = -1.0, None
    for tau in tau_values:
        m = compute_metrics(p_val > tau, correct_0, correct_20)
        if m["adaptive_R@1"] > best_r1:
            best_r1, best_tau = m["adaptive_R@1"], tau

    target_r1, best_tau_eff, best_avg = best_r1 - args.max_drop_pp / 100.0, None, float("inf")
    for tau in tau_values:
        m = compute_metrics(p_val > tau, correct_0, correct_20)
        if m["adaptive_R@1"] >= target_r1 and m["avg_matches"] < best_avg:
            best_avg, best_tau_eff = m["avg_matches"], tau

    tau_final = best_tau_eff if best_tau_eff is not None else best_tau
    print(f"tau = {tau_final}")

    save_model_json(_MODEL_JSON, regressor_to_dict(clf, ["SU"]))
    save_threshold_csv(_THRESHOLD_CSV, tau=tau_final)


if __name__ == "__main__":
    main(parse_args())
