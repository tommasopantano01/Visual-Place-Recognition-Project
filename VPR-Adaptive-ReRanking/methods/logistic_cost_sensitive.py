"""
logistic_cost_sensitive/logistic_cost_sensitive_calibrate.py — Allena
P(help|inliers) e P(hurt|inliers) separatamente, poi calibra lambda e tau
su validation per S = P(help) - lambda*P(hurt). Scrive threshold.csv
(lambda,tau) + model.json (entrambi i regressori).

Uso:
    python VPR-adaptive-re-ranking/logistic_cost_sensitive/logistic_cost_sensitive_calibrate.py \
        --train-csv train.csv --val-csv val.csv
"""
import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _calibrate_common import (
    load_query_level, compute_metrics, fit_logistic, regressor_to_dict,
    save_threshold_csv, save_model_json,
)

_THRESHOLD_CSV = Path(__file__).parent / "threshold.csv"
_MODEL_JSON    = Path(__file__).parent / "model.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Calibra cost-sensitive (P_help - lambda*P_hurt) — metodo7")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv",   required=True)
    parser.add_argument("--lambda-max",  type=float, default=3.0)
    parser.add_argument("--lambda-step", type=float, default=0.1)
    parser.add_argument("--tau-min", type=float, default=-1.0)
    parser.add_argument("--tau-max", type=float, default=1.0)
    parser.add_argument("--tau-step", type=float, default=0.05)
    return parser.parse_args()


def main(args):
    train_df = load_query_level(args.train_csv)
    val_df   = load_query_level(args.val_csv)

    X_train = train_df[["num_inliers_top1"]].values
    y_help  = train_df["helps_20"].values
    y_hurt  = train_df["hurts_20"].values
    X_val   = val_df[["num_inliers_top1"]].values
    correct_0, correct_20 = val_df["correct_0"].values, val_df["correct_20"].values

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    clf_help = fit_logistic(X_train, y_help)
    clf_hurt = fit_logistic(X_train, y_hurt)
    p_help_val = clf_help.predict_proba(X_val)[:, 1]
    p_hurt_val = clf_hurt.predict_proba(X_val)[:, 1]

    lambda_values = np.round(np.arange(0.0, args.lambda_max + args.lambda_step, args.lambda_step), 3)
    tau_values    = np.round(np.arange(args.tau_min, args.tau_max + args.tau_step, args.tau_step), 3)

    best_r1, best = -1.0, None
    for lam in tqdm(lambda_values, desc="grid search lambda"):
        S_val = p_help_val - lam * p_hurt_val
        for tau in tau_values:
            m = compute_metrics(S_val > tau, correct_0, correct_20)
            better = m["adaptive_R@1"] > best_r1
            tie_more_saving = (best is not None and m["adaptive_R@1"] == best_r1
                                and m["avg_matches"] < best["m"]["avg_matches"])
            if better or tie_more_saving:
                best_r1 = m["adaptive_R@1"]
                best = {"lam": lam, "tau": tau, "m": m}

    print(f"lambda = {best['lam']}   tau = {best['tau']}   R@1 = {best['m']['adaptive_R@1']:.4f}")

    save_model_json(_MODEL_JSON, {
        "help": regressor_to_dict(clf_help, ["inliers"]),
        "hurt": regressor_to_dict(clf_hurt, ["inliers"]),
    })
    save_threshold_csv(_THRESHOLD_CSV, **{"lambda": best["lam"], "tau": best["tau"]})


if __name__ == "__main__":
    main(parse_args())
