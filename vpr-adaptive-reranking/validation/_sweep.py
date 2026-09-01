"""
_sweep.py — Shared engine of the hardthreshold methods (youden, best_r1,
efficiency). From a validation candidate level CSV it builds the full sweep of
thresholds on num_inliers_top1 and contains the three selectors.

Decision rule: rerank(q) if inliers_top1(q) < T.

Used as a module by validation/thresholds.py or,
as a script: builds and saves the sweep from a candidate-level CSV (--help).
"""
import argparse
import os
import warnings
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"query_id", "retrieval_rank", "num_inliers",
                    "rerank_rank_topK", "is_positive", "K"}


# ── CANDIDATE-LEVEL -> QUERY-LEVEL -> SWEEP ──────────────────────────

def validate_candidate_level(df):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing candidate-level columns: {sorted(missing)}")
    df = df.copy()
    for c in ("query_id", "retrieval_rank", "num_inliers", "rerank_rank_topK", "is_positive", "K"):
        df[c] = df[c].astype(int)
    if (df["num_inliers"] < 0).any():
        raise ValueError("num_inliers must be non negative.")
    if not set(df["is_positive"].unique()).issubset({0, 1}):
        raise ValueError("is_positive must contain only 0/1.")
    return df


def candidate_level_to_query_level(df, strict_k=False):
    """For every query: inliers_top1 (retrieval_rank==1), pre_r1_correct
    (is_positive of retrieval_rank==1), post_r1_correct (is_positive of
    rerank_rank_topK==1)."""
    df = validate_candidate_level(df)

    candidates_per_query = df.groupby("query_id").size()
    expected_k = df.groupby("query_id")["K"].first()
    inconsistent_k = candidates_per_query[candidates_per_query != expected_k]
    if len(inconsistent_k) > 0:
        msg = f"Some queries have a number of rows != K. Examples: {inconsistent_k.head(10).to_dict()}"
        if strict_k:
            raise ValueError(msg)
        warnings.warn(msg)

    top1_retr = df[df["retrieval_rank"] == 1]
    if not (top1_retr.groupby("query_id").size() == 1).all():
        bad = top1_retr.groupby("query_id").size()
        bad = bad[bad != 1].head(10).to_dict()
        raise ValueError(f"Every query must have exactly one retrieval_rank==1. Examples: {bad}")

    top1_rerank = df[df["rerank_rank_topK"] == 1]
    if not (top1_rerank.groupby("query_id").size() == 1).all():
        bad = top1_rerank.groupby("query_id").size()
        bad = bad[bad != 1].head(10).to_dict()
        raise ValueError(f"Every query must have exactly one rerank_rank_topK==1. Examples: {bad}")

    q_pre = top1_retr[["query_id", "num_inliers", "is_positive"]].rename(
        columns={"num_inliers": "inliers_top1", "is_positive": "pre_r1_correct"})
    q_post = top1_rerank[["query_id", "is_positive"]].rename(
        columns={"is_positive": "post_r1_correct"})
    query_level = q_pre.merge(q_post, on="query_id", how="inner")

    if len(query_level) != df["query_id"].nunique():
        raise ValueError("Queries lost during the candidate->query conversion.")
    return query_level.sort_values("query_id").reset_index(drop=True)


def evaluate_threshold(query_level, threshold, top_k=20):
    """Evaluates a threshold T. Decision: rerank if inliers_top1 < T."""
    inliers      = query_level["inliers_top1"].to_numpy(dtype=int)
    pre_correct  = query_level["pre_r1_correct"].to_numpy(dtype=int)
    post_correct = query_level["post_r1_correct"].to_numpy(dtype=int)

    rerank_mask = inliers < threshold
    adaptive_correct = np.where(rerank_mask, post_correct, pre_correct)

    true_hard = pre_correct == 0
    true_easy = pre_correct == 1
    TP_hard = int(np.sum(rerank_mask & true_hard))
    FP_easy = int(np.sum(rerank_mask & true_easy))
    FN_hard = int(np.sum((~rerank_mask) & true_hard))
    TN_easy = int(np.sum((~rerank_mask) & true_easy))

    tpr_hard = TP_hard / (TP_hard + FN_hard) if (TP_hard + FN_hard) > 0 else np.nan
    fpr_easy = FP_easy / (FP_easy + TN_easy) if (FP_easy + TN_easy) > 0 else np.nan

    reranked_fraction = float(np.mean(rerank_mask))
    matches_per_query = 1.0 + (top_k - 1) * reranked_fraction
    saving = 1.0 - matches_per_query / top_k

    pre_r1      = float(np.mean(pre_correct))
    full_r1     = float(np.mean(post_correct))
    adaptive_r1 = float(np.mean(adaptive_correct))
    youden = tpr_hard - fpr_easy if np.isfinite(tpr_hard) and np.isfinite(fpr_easy) else np.nan

    return {
        "threshold": int(threshold),
        "n_queries": int(len(query_level)),
        "pre_r1_pct": 100.0 * pre_r1,
        "full_rerank_r1_pct": 100.0 * full_r1,
        "adaptive_r1_pct": 100.0 * adaptive_r1,
        "adaptive_delta_vs_pre_pp": 100.0 * (adaptive_r1 - pre_r1),
        "adaptive_delta_vs_full_pp": 100.0 * (adaptive_r1 - full_r1),
        "reranked_pct": 100.0 * reranked_fraction,
        "matches_per_query": matches_per_query,
        "saving_pct": 100.0 * saving,
        "tpr_hard_pct": 100.0 * tpr_hard if np.isfinite(tpr_hard) else np.nan,
        "fpr_easy_pct": 100.0 * fpr_easy if np.isfinite(fpr_easy) else np.nan,
        "youden": youden,
        "TP_hard": TP_hard, "FP_easy": FP_easy, "FN_hard": FN_hard, "TN_easy": TN_easy,
    }


def build_threshold_sweep(query_level, top_k=20):
    """Full sweep. T=0 -> no rerank (inliers >= 0); T=max+1 -> all of them."""
    max_threshold = int(query_level["inliers_top1"].max()) + 1
    rows = [evaluate_threshold(query_level, T, top_k=top_k)
            for T in range(0, max_threshold + 1)]
    return pd.DataFrame(rows)


def load_candidate_level(val_csv):
    """Single file or directory of candidate-level CSVs -> one DataFrame."""
    if os.path.isdir(val_csv):
        files = sorted(glob(os.path.join(val_csv, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV in {val_csv}")
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return pd.read_csv(val_csv)


def sweep_from_candidate(val_csv, top_k=20, strict_k=False):
    cl = load_candidate_level(val_csv)
    ql = candidate_level_to_query_level(cl, strict_k=strict_k)
    return build_threshold_sweep(ql, top_k=top_k)


# ── SELECTORS ────────────────────────────────────────────────────────

def select_best_r1_threshold(sweep):
    """Threshold that maximises adaptive R@1; tie: saving desc, threshold asc."""
    df = sweep.copy()
    best_r1 = df["adaptive_r1_pct"].max()
    candidates = df[df["adaptive_r1_pct"] == best_r1]
    row = candidates.sort_values(["saving_pct", "threshold"], ascending=[False, True]).iloc[0]
    return pd.DataFrame([{"method": "best_r1", "threshold": int(row["threshold"]),
                          "r1_adaptive_pct": row["adaptive_r1_pct"], "saving_pct": row["saving_pct"]}])


def select_youden_threshold(sweep):
    """Threshold that maximises Youden (TPR_hard - FPR_easy); tie: saving desc,
    adaptive_r1 desc, threshold asc."""
    df = sweep.copy()
    if "youden" not in df.columns:
        df["youden"] = df["tpr_hard_pct"] - df["fpr_easy_pct"]
    row = df.sort_values(["youden", "saving_pct", "adaptive_r1_pct", "threshold"],
                         ascending=[False, False, False, True]).iloc[0]
    return pd.DataFrame([{"method": "youden", "threshold": int(row["threshold"]),
                          "r1_adaptive_pct": row["adaptive_r1_pct"], "saving_pct": row["saving_pct"]}])


def select_eff95_threshold(sweep, retention=0.95):
    """Most efficient threshold that retains >= retention of the R@1 gain
    with respect to the best adaptive R@1; tie: saving desc, adaptive_r1 desc, threshold asc."""
    df = sweep.copy()
    pre_r1  = df["pre_r1_pct"].iloc[0]
    best_r1 = df["adaptive_r1_pct"].max()
    delta_r = best_r1 - pre_r1
    target_r1 = pre_r1 if delta_r <= 0 else pre_r1 + retention * delta_r
    candidates = df[df["adaptive_r1_pct"] >= target_r1]
    row = candidates.sort_values(["saving_pct", "adaptive_r1_pct", "threshold"],
                                 ascending=[False, False, True]).iloc[0]
    return pd.DataFrame([{"method": f"eff{int(retention * 100)}", "threshold": int(row["threshold"]),
                          "r1_adaptive_pct": row["adaptive_r1_pct"], "saving_pct": row["saving_pct"]}])


# ── CLI: builds and saves the sweep ──────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Builds the threshold sweep from a candidate-level CSV.")
    p.add_argument("--input", required=True, type=Path, help="candidate-level CSV (file or dir)")
    p.add_argument("--output", required=True, type=Path, help="output sweep CSV")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--query-level-output", type=Path, default=None)
    p.add_argument("--strict-k", action="store_true")
    args = p.parse_args()

    cl = load_candidate_level(str(args.input))
    ql = candidate_level_to_query_level(cl, strict_k=args.strict_k)
    sweep = build_threshold_sweep(ql, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output, index=False)
    if args.query_level_output is not None:
        args.query_level_output.parent.mkdir(parents=True, exist_ok=True)
        ql.to_csv(args.query_level_output, index=False)
    print(f"Sweep saved to: {args.output}  |  queries: {len(ql)}  |  thresholds: {len(sweep)}")


if __name__ == "__main__":
    main()
