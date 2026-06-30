"""
youden/youden_calibrate.py — Calibra T_B massimizzando lo Youden index
(TPR_hard - FPR_easy) su un validation set. Scrive threshold.csv.

Si esegue UNA volta, offline. youden.py (l'inferenza) legge solo il
risultato, non rialllena mai nulla.

Uso:
    python VPR-adaptive-re-ranking/youden/youden_calibrate.py --val-csv val.csv
"""
import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _calibrate_common import load_query_level, save_threshold_csv

_THRESHOLD_CSV = Path(__file__).parent / "threshold.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Calibra T_B (Youden) — metodo1")
    parser.add_argument("--val-csv", required=True, help="validation CSV candidate-level")
    return parser.parse_args()


def main(args):
    val_df = load_query_level(args.val_csv)
    x         = val_df["num_inliers_top1"].values
    correct_0 = val_df["correct_0"].values
    hard      = (correct_0 == 0)

    print(f"Val: {len(val_df)} query | hard: {hard.sum()} | easy: {(~hard).sum()}")

    best_J, best_T = -np.inf, int(x.min())
    for T in range(int(x.min()), int(x.max()) + 2):
        rerank = x < T
        tpr_hard  = rerank[hard].mean()  if hard.any()  else 0.0
        fpr_easy  = rerank[~hard].mean() if (~hard).any() else 0.0
        J = tpr_hard - fpr_easy
        if J > best_J:
            best_J, best_T = J, T

    print(f"T_B = {best_T}  (Youden J = {best_J:.4f})")
    save_threshold_csv(_THRESHOLD_CSV, threshold=best_T)


if __name__ == "__main__":
    main(parse_args())
