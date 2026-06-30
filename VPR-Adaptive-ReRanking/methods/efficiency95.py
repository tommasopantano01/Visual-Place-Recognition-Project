"""
efficiency/efficiency_calibrate.py — Calibra T_95, la soglia piu'
economica che conserva almeno il 95% del guadagno di R@1 best rispetto
al retrieval-only. Scrive threshold.csv.

Uso:
    python VPR-adaptive-re-ranking/efficiency/efficiency_calibrate.py \
        --val-csv val.csv [--retention 0.95]
"""
import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _calibrate_common import load_query_level, compute_metrics, save_threshold_csv

_THRESHOLD_CSV = Path(__file__).parent / "threshold.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Calibra T_95 — metodo3")
    parser.add_argument("--val-csv",    required=True, help="validation CSV candidate-level")
    parser.add_argument("--retention",  type=float, default=0.95)
    return parser.parse_args()


def main(args):
    val_df = load_query_level(args.val_csv)
    x          = val_df["num_inliers_top1"].values
    correct_0  = val_df["correct_0"].values
    correct_20 = val_df["correct_20"].values

    pre_r1 = correct_0.mean()
    print(f"Val: {len(val_df)} query | Retrieval R@1 = {pre_r1:.4f}")

    thresholds = range(int(x.min()), int(x.max()) + 2)
    metrics_by_T = {T: compute_metrics(x < T, correct_0, correct_20) for T in thresholds}

    best_r1 = max(m["adaptive_R@1"] for m in metrics_by_T.values())
    delta_r = best_r1 - pre_r1
    target_r1 = pre_r1 if delta_r <= 0 else pre_r1 + args.retention * delta_r

    candidates = [(T, m) for T, m in metrics_by_T.items() if m["adaptive_R@1"] >= target_r1]
    # massimo saving, a parita' massima R@1, a parita' soglia minima
    candidates.sort(key=lambda tm: (-tm[1]["savings_%"], -tm[1]["adaptive_R@1"], tm[0]))
    best_T, best_m = candidates[0]

    print(f"R@1_best = {best_r1:.4f}  target (retention {args.retention}) = {target_r1:.4f}")
    print(f"T_95 = {best_T}  (R@1 adattivo = {best_m['adaptive_R@1']:.4f}, savings = {best_m['savings_%']:.1f}%)")
    save_threshold_csv(_THRESHOLD_CSV, threshold=best_T)


if __name__ == "__main__":
    main(parse_args())
