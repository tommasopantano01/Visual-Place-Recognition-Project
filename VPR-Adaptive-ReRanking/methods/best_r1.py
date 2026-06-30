"""
best_r1/best_r1_calibrate.py — Calibra T_best, la soglia che massimizza
direttamente la R@1 adattiva su validation. Scrive threshold.csv.

NOTA: non e' raccomandato come policy a se' stante (e' soprattutto il
passo interno usato per calcolare il delta di T_95), mantenuto separato
finche' non viene confermato il contrario.

Uso:
    python VPR-adaptive-re-ranking/best_r1/best_r1_calibrate.py --val-csv val.csv
"""
import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _calibrate_common import load_query_level, compute_metrics, save_threshold_csv

_THRESHOLD_CSV = Path(__file__).parent / "threshold.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Calibra T_best — metodo2")
    parser.add_argument("--val-csv", required=True, help="validation CSV candidate-level")
    return parser.parse_args()


def main(args):
    val_df = load_query_level(args.val_csv)
    x          = val_df["num_inliers_top1"].values
    correct_0  = val_df["correct_0"].values
    correct_20 = val_df["correct_20"].values

    print(f"Val: {len(val_df)} query")

    best_r1, best_T, best_m = -1.0, int(x.min()), None
    for T in range(int(x.min()), int(x.max()) + 2):
        m = compute_metrics(x < T, correct_0, correct_20)
        if m["adaptive_R@1"] > best_r1:
            best_r1, best_T, best_m = m["adaptive_R@1"], T, m

    print(f"T_best = {best_T}  (R@1 adattivo = {best_r1:.4f}, savings = {best_m['savings_%']:.1f}%)")
    save_threshold_csv(_THRESHOLD_CSV, threshold=best_T)


if __name__ == "__main__":
    main(parse_args())
