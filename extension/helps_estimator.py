"""
Stima la threshold num_inliers per metodo3 — local P(helps).

Uso:
    python estimate_threshold_local.py \\
        --train-csv path/to/train_query_level.csv \\
        --val-csv   path/to/val_query_level.csv

Output: il valore intero da inserire in THRESHOLDS di match_queries_preds.py.

Colonne richieste nel train CSV: num_inliers_top1, helps_20
Colonne richieste nel val CSV:   num_inliers_top1, correct_0, correct_20
"""

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# STIMA LOCALE DI P(helps | num_inliers_top1)
# ============================================================

def estimate_local_p_help(values, train_x, train_y, initial_window=1, min_k=30):
    """
    Per ogni valore in `values`, stima P(helps) come media locale sul train.
    Espande la finestra finché non trova almeno min_k campioni.
    """
    max_x = train_x.max()
    p_help_list = []

    for N in tqdm(values, desc=f"Stima P(helps) locale (min_k={min_k})", leave=False):
        N = float(N)
        window = float(initial_window)

        while True:
            mask = (train_x >= N - window) & (train_x <= N + window)
            local_y = train_y[mask]
            if len(local_y) >= min_k or window >= max_x:
                break
            window += 1.0

        p_help_list.append(0.0 if len(local_y) == 0 else float(local_y.mean()))

    return np.array(p_help_list)


# ============================================================
# METRICHE DI UNA POLICY
# ============================================================

def compute_metrics(p_help, correct_0, correct_20, tau, K=20):
    rerank = p_help > tau
    adaptive_correct = np.where(rerank, correct_20, correct_0)
    rerank_rate = rerank.mean()
    avg_matches = 1 + (K - 1) * rerank_rate
    return {
        "adaptive_R@1":    adaptive_correct.mean(),
        "avg_matches":     avg_matches,
        "savings_%":       100 * (1 - avg_matches / K),
        "rerank_%":        100 * rerank_rate,
    }


# ============================================================
# TROVA N* = crossover num_inliers dove P(helps) ≈ best_tau
# ============================================================

def find_inlier_threshold(train_x, train_y, best_min_k, best_tau, initial_window=1):
    """
    Scansiona tutti i valori interi da min a max num_inliers nel train.
    Restituisce l'ultimo N intero tale che P(helps|N) > best_tau.
    Questo è il threshold da inserire in THRESHOLDS.
    """
    x_range = np.arange(int(train_x.min()), int(train_x.max()) + 1)
    p_help = estimate_local_p_help(
        x_range.astype(float), train_x, train_y, initial_window, best_min_k
    )

    above = x_range[p_help > best_tau]

    if len(above) == 0:
        return int(train_x.min())
    return int(above.max())


# ============================================================
# ARGOMENTI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stima threshold num_inliers per metodo3 (local P(helps))"
    )
    parser.add_argument("--train-csv",      required=True, help="train query-level CSV")
    parser.add_argument("--val-csv",        required=True, help="validation query-level CSV")
    parser.add_argument("--min-k-values",   nargs="+", type=int,
                        default=[10, 20, 30, 50, 75, 100],
                        help="valori di min_k da esplorare")
    parser.add_argument("--tau-step",       type=float, default=0.05,
                        help="step per la grid search su tau")
    parser.add_argument("--initial-window", type=int, default=1)
    parser.add_argument("--max-drop-pp",    type=float, default=0.1,
                        help="max calo R@1 tollerato rispetto al best (in punti percentuali)")
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main(args):
    # --- Caricamento ---
    train_df = pd.read_csv(args.train_csv).dropna(subset=["num_inliers_top1", "helps_20"])
    val_df   = pd.read_csv(args.val_csv).dropna(subset=["num_inliers_top1", "correct_0", "correct_20"])

    train_df["num_inliers_top1"] = train_df["num_inliers_top1"].astype(float)
    train_df["helps_20"]         = train_df["helps_20"].astype(int)
    val_df["num_inliers_top1"]   = val_df["num_inliers_top1"].astype(float)
    val_df["correct_0"]          = val_df["correct_0"].astype(int)
    val_df["correct_20"]         = val_df["correct_20"].astype(int)

    train_x    = train_df["num_inliers_top1"].values
    train_y    = train_df["helps_20"].values
    val_x      = val_df["num_inliers_top1"].values
    correct_0  = val_df["correct_0"].values
    correct_20 = val_df["correct_20"].values

    tau_values = np.round(np.arange(0.05, 1.0, args.tau_step), 3)

    print(f"Train: {len(train_df)} query | Val: {len(val_df)} query")
    print(f"Retrieval R@1 (val):    {correct_0.mean():.4f}")
    print(f"Full rerank R@1 (val):  {correct_20.mean():.4f}")

    # --- Grid search su (min_k, tau) ---
    best_r1     = -1.0
    best_record = None

    for min_k in tqdm(args.min_k_values, desc="Grid search min_k"):
        p_help = estimate_local_p_help(val_x, train_x, train_y, args.initial_window, min_k)
        for tau in tau_values:
            m = compute_metrics(p_help, correct_0, correct_20, tau)
            if m["adaptive_R@1"] > best_r1:
                best_r1     = m["adaptive_R@1"]
                best_record = {"min_k": min_k, "tau": tau, "metrics": m}

    # --- Policy più efficiente entro max_drop_pp dal best ---
    target_r1       = best_r1 - args.max_drop_pp / 100.0
    best_efficient  = None
    best_avg        = float("inf")

    for min_k in args.min_k_values:
        p_help = estimate_local_p_help(val_x, train_x, train_y, args.initial_window, min_k)
        for tau in tau_values:
            m = compute_metrics(p_help, correct_0, correct_20, tau)
            if m["adaptive_R@1"] >= target_r1 and m["avg_matches"] < best_avg:
                best_avg       = m["avg_matches"]
                best_efficient = {"min_k": min_k, "tau": tau, "metrics": m}

    chosen = best_efficient if best_efficient else best_record

    BEST_MIN_K     = chosen["min_k"]
    BEST_THRESHOLD = chosen["tau"]
    m              = chosen["metrics"]

    # --- Converti in num_inliers threshold ---
    print("\nCalcolo N* (crossover num_inliers)...")
    N_star = find_inlier_threshold(train_x, train_y, BEST_MIN_K, BEST_THRESHOLD, args.initial_window)

    print("\n" + "=" * 60)
    print("RISULTATO — metodo3 (local P(helps))")
    print(f"  BEST_MIN_K      = {BEST_MIN_K}")
    print(f"  BEST_THRESHOLD  = {BEST_THRESHOLD}")
    print(f"  Adaptive R@1    = {m['adaptive_R@1']:.4f}")
    print(f"  Savings         = {m['savings_%']:.1f}%")
    print(f"  → num_inliers threshold (N*) = {N_star}")
    print()
    print("Inserisci in match_queries_preds.py → THRESHOLDS:")
    print(f'  ("metodo3", "<vpr_method>", "<matcher>"): {N_star}')
    print("=" * 60)


if __name__ == "__main__":
    main(parse_args())
