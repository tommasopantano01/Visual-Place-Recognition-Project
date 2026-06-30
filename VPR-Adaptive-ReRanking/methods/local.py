"""
local/local_calibrate.py — Calibra N* tramite stima non parametrica di
P(helps | num_inliers): finestra adattiva centrata su ogni valore di
inliers, espansa finche' non contiene almeno K_min campioni di train.
Scrive threshold.csv.

Uso:
    python VPR-adaptive-re-ranking/local/local_calibrate.py \
        --train-csv train.csv --val-csv val.csv
"""
import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _calibrate_common import load_query_level, compute_metrics, save_threshold_csv

_THRESHOLD_CSV = Path(__file__).parent / "threshold.csv"


def estimate_p_help(values, train_x, train_y, initial_window=1, min_k=30):
    max_x = train_x.max()
    out = []
    for N in tqdm(values, desc=f"P(helps) locale (min_k={min_k})", leave=False):
        N = float(N)
        window = float(initial_window)
        while True:
            mask = (train_x >= N - window) & (train_x <= N + window)
            local_y = train_y[mask]
            if len(local_y) >= min_k or window >= max_x:
                break
            window += 1.0
        out.append(0.0 if len(local_y) == 0 else float(local_y.mean()))
    return np.array(out)


def parse_args():
    parser = argparse.ArgumentParser(description="Calibra N* (local P(helps)) — metodo4")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv",   required=True)
    parser.add_argument("--min-k-values", nargs="+", type=int, default=[10, 20, 30, 50, 75, 100])
    parser.add_argument("--initial-window", type=int, default=1)
    parser.add_argument("--tau-step", type=float, default=0.05)
    parser.add_argument("--max-drop-pp", type=float, default=0.1)
    return parser.parse_args()


def main(args):
    train_df = load_query_level(args.train_csv)
    val_df   = load_query_level(args.val_csv)
    train_x, train_y = train_df["num_inliers_top1"].values, train_df["helps_20"].values
    val_x      = val_df["num_inliers_top1"].values
    correct_0  = val_df["correct_0"].values
    correct_20 = val_df["correct_20"].values
    tau_values = np.round(np.arange(0.05, 1.0, args.tau_step), 3)

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    best_r1, best = -1.0, None
    for min_k in tqdm(args.min_k_values, desc="grid search min_k"):
        p_help = estimate_p_help(val_x, train_x, train_y, args.initial_window, min_k)
        for tau in tau_values:
            m = compute_metrics(p_help > tau, correct_0, correct_20)
            if m["adaptive_R@1"] > best_r1:
                best_r1, best = m["adaptive_R@1"], {"min_k": min_k, "tau": tau}

    target_r1, best_eff, best_avg = best_r1 - args.max_drop_pp / 100.0, None, float("inf")
    for min_k in args.min_k_values:
        p_help = estimate_p_help(val_x, train_x, train_y, args.initial_window, min_k)
        for tau in tau_values:
            m = compute_metrics(p_help > tau, correct_0, correct_20)
            if m["adaptive_R@1"] >= target_r1 and m["avg_matches"] < best_avg:
                best_avg, best_eff = m["avg_matches"], {"min_k": min_k, "tau": tau}

    chosen = best_eff if best_eff else best

    x_range = np.arange(int(train_x.min()), int(train_x.max()) + 1)
    p_help_range = estimate_p_help(x_range.astype(float), train_x, train_y,
                                    args.initial_window, chosen["min_k"])
    above = x_range[p_help_range > chosen["tau"]]
    N_star = int(above.max()) if len(above) > 0 else int(train_x.min())

    print(f"min_k={chosen['min_k']}  tau={chosen['tau']}  -> N* = {N_star}")
    save_threshold_csv(_THRESHOLD_CSV, threshold=N_star)


if __name__ == "__main__":
    main(parse_args())
