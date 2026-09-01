"""
Loads the THREE trained regressors, chosen by retrieval model and matcher,
and grid-searches on the validation dataset the thresholds (tau1, tau5, tau10)
that maximise the adaptive R@1.

INPUT MODELS (--models-dir, default validation/sequential/): three files, one per gate:
    seq_model_continue_1_phelps_<model>_svox_train_<matcher>.json   (gate1)
    seq_model_continue_5_<model>_svox_train_<matcher>.json          (gate5)
    seq_model_continue_10_<model>_svox_train_<matcher>.json         (gate10)

Each JSON has the regressor_to_dict format (_common): anonymous feat_cols
("feature_0"...), scaler_mean/scale, coef, intercept, classes. Features are
passed POSITIONALLY in the exact order below.

FEATURES:
  gate1  (1):  [num_inliers_top1]
  gate5  (6):  [num_inliers_top1, max_inliers_top5, second_max_inliers_top5,
                gap_inliers_top5, best_retrieval_rank_top5, top1_is_best_top5]
  gate10 (10): [num_inliers_top1, max_inliers_top5, gap_inliers_top5,
                best_retrieval_rank_top5, top1_is_best_top5,
                max_inliers_top10, second_max_inliers_top10, gap_inliers_top10,
                best_retrieval_rank_top10, top1_is_best_top10]

TARGETS:
  gate1:  helps_20    = (correct_0==0) & (correct_20==1)
  gate5:  continue_5  = (correct_5==0) & (max(correct_10,correct_20)==1)
  gate10: continue_10 = (correct_10==0) & (correct_20==1)

POLICY:
  stop at retrieval if p1<=tau1; stop at top5 if p1>tau1 & p5<=tau5;
  stop at top10 if p5>tau5 & p10<=tau10; full top20 if p10>tau10.
  Cost: budget 0 costs 1 (the top-1 IM is needed). Tie-break: max R@1, then min avg_matches.

INPUT --val-csv.

Writes in validation/sequential/:
  threshold_<model>_<matcher>.csv   tau1, tau5, tau10 + metrics (read by deploy)
  selection_<model>_<matcher>.csv   val_csv, metrics, params, stop distribution
and updates validation/summary.csv.
"""
import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
sys.path.append(str(_HERE.parent))
from _common import regressor_from_dict, predict_proba_pos
from _outputs import (canon_model, canon_matcher, val_tag, write_threshold_csv,
                      write_selection_csv, upsert_summary, print_written)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **k):
        return x

FAMILY = "sequential"
SUBDIR = "sequential"

# ── EXACT FEATURE ORDER (positional, must match the trained JSONs) ──
FEATURES_CONTINUE_1 = ["num_inliers_top1"]
FEATURES_CONTINUE_5 = [
    "num_inliers_top1", "max_inliers_top5", "second_max_inliers_top5",
    "gap_inliers_top5", "best_retrieval_rank_top5", "top1_is_best_top5",
]
FEATURES_CONTINUE_10 = [
    "num_inliers_top1", "max_inliers_top5", "gap_inliers_top5",
    "best_retrieval_rank_top5", "top1_is_best_top5",
    "max_inliers_top10", "second_max_inliers_top10", "gap_inliers_top10",
    "best_retrieval_rank_top10", "top1_is_best_top10",
]
REQUIRED_CAND_COLS = ["query_id", "candidate_path", "retrieval_rank",
                      "num_inliers", "is_positive"]
_PROG_BUDGETS = (5, 10, 20)


# ── candidate-level ────────

def rerank_correct_with_budget(group, budget):
    sub = group[group["retrieval_rank"] <= budget]
    if len(sub) == 0:
        return np.nan
    winner = sub.sort_values(["num_inliers", "retrieval_rank"],
                             ascending=[False, True]).iloc[0]
    return int(winner["is_positive"])


def progressive_features_for_group(group, b):
    sub = group[group["retrieval_rank"] <= b]
    if len(sub) == 0:
        return {f"max_inliers_top{b}": np.nan, f"second_max_inliers_top{b}": np.nan,
                f"gap_inliers_top{b}": np.nan, f"best_retrieval_rank_top{b}": np.nan,
                f"top1_is_best_top{b}": np.nan}
    ss = sub.sort_values(["num_inliers", "retrieval_rank"], ascending=[False, True])
    best = ss.iloc[0]
    max_inl = float(best["num_inliers"])
    best_rank = int(best["retrieval_rank"])
    second_max = float(ss.iloc[1]["num_inliers"]) if len(ss) >= 2 else 0.0
    return {f"max_inliers_top{b}": max_inl,
            f"second_max_inliers_top{b}": second_max,
            f"gap_inliers_top{b}": max_inl - second_max,
            f"best_retrieval_rank_top{b}": best_rank,
            f"top1_is_best_top{b}": int(best_rank == 1)}


def candidate_to_query_seq_df(cdf):
    missing = [c for c in REQUIRED_CAND_COLS if c not in cdf.columns]
    if missing:
        raise ValueError(f"Missing candidate-level columns: {missing}")
    cdf = cdf.copy()
    cdf["query_id"] = cdf["query_id"].astype(str)
    cdf["retrieval_rank"] = cdf["retrieval_rank"].astype(int)
    cdf["num_inliers"] = cdf["num_inliers"].astype(float)
    cdf["is_positive"] = cdf["is_positive"].astype(int)

    rows = []
    for qid, g in cdf.groupby("query_id", sort=False):
        top1 = g[g["retrieval_rank"] == 1]
        if len(top1) == 0:
            continue
        top1 = top1.iloc[0]
        row = {"query_id": str(qid),
               "num_inliers_top1": float(top1["num_inliers"]),
               "correct_0": int(top1["is_positive"]),
               "correct_5": rerank_correct_with_budget(g, 5),
               "correct_10": rerank_correct_with_budget(g, 10),
               "correct_20": rerank_correct_with_budget(g, 20)}
        for b in _PROG_BUDGETS:
            row.update(progressive_features_for_group(g, b))
        rows.append(row)
    return pd.DataFrame(rows)


def add_sequential_labels(df):
    df = df.copy()
    for c in ("correct_0", "correct_5", "correct_10", "correct_20"):
        df[c] = df[c].astype(int)
    df["helps_20"]    = ((df["correct_0"] == 0) & (df["correct_20"] == 1)).astype(int)
    df["continue_5"]  = ((df["correct_5"] == 0) &
                         (df[["correct_10", "correct_20"]].max(axis=1) == 1)).astype(int)
    df["continue_10"] = ((df["correct_10"] == 0) & (df["correct_20"] == 1)).astype(int)
    return df


def load_query_seq(val_csv):
    if os.path.isdir(val_csv):
        files = sorted(glob(os.path.join(val_csv, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV in {val_csv}")
        raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        raw = pd.read_csv(val_csv)

    seq_needed = set(["query_id", "correct_0", "correct_5", "correct_10",
                      "correct_20"] + FEATURES_CONTINUE_10)
    cand_cols = set(REQUIRED_CAND_COLS)
    cols = set(raw.columns)
    if seq_needed.issubset(cols):
        df = raw.copy()
    elif cand_cols.issubset(cols):
        df = candidate_to_query_seq_df(raw)
    else:
        raise ValueError(f"Unrecognised CSV format. Columns found: {list(raw.columns)}")

    df = add_sequential_labels(df)
    all_feats = sorted(set(FEATURES_CONTINUE_5) | set(FEATURES_CONTINUE_10))
    keep = ["query_id", "correct_0", "correct_5", "correct_10", "correct_20"] + all_feats
    df = df.dropna(subset=[c for c in keep if c in df.columns]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid query after dropna (>=10 candidates/query needed).")
    for c in ("correct_0", "correct_5", "correct_10", "correct_20"):
        df[c] = df[c].astype(int)
    return df


# ── loading of the 3 gate JSONs from the models dir ────────────────

# tokens accepted in the JSON file names for each canonical matcher name
MATCHER_FILE_TOKENS = {"superpoint-lg": ("superpoint-lg", "sp-lg"), "loftr": ("loftr",)}


def _find_gate_json(models_dir, gate_num, model, matcher):
    """Find the gate JSON in models_dir, tolerant to naming."""
    pats = []
    for tok in MATCHER_FILE_TOKENS.get(matcher, (matcher,)):
        pats += [os.path.join(models_dir, f"*continue_{gate_num}_*{model}*{tok}*.json"),
                 os.path.join(models_dir, f"*continue_{gate_num}*{model}*{tok}*.json")]
    for pat in pats:
        hits = sorted(glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No JSON for gate{gate_num} in {models_dir} "
        f"(model='{model}', matcher='{matcher}'). Patterns tried: {pats}\n"
        "  -> run validation/download_models.py or pass --models-dir")


def load_gate_models(models_dir, model, matcher):
    g1 = _find_gate_json(models_dir, 1, model, matcher)
    g5 = _find_gate_json(models_dir, 5, model, matcher)
    g10 = _find_gate_json(models_dir, 10, model, matcher)
    print(f"  gate1  <- {os.path.basename(g1)}")
    print(f"  gate5  <- {os.path.basename(g5)}")
    print(f"  gate10 <- {os.path.basename(g10)}")
    with open(g1) as f:  m1 = json.load(f)
    with open(g5) as f:  m5 = json.load(f)
    with open(g10) as f: m10 = json.load(f)
    # expected number of features (faithful to the notebook)
    for name, m, exp in (("gate1", m1, 1), ("gate5", m5, 6), ("gate10", m10, 10)):
        n = len(m["feat_cols"])
        if n != exp:
            raise ValueError(f"{name}: expected {exp} features, the JSON has {n} "
                             f"(feat_cols={m['feat_cols']}).")
    return m1, m5, m10


def gate_proba(df, gate_model, feature_names):
    """Build X POSITIONALLY in the order feature_names (JSON names are
    anonymous feature_0..N) and apply the regressor."""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Features required but absent from the CSV: {missing}")
    X = df[feature_names].to_numpy(dtype=float)
    if X.shape[1] != len(gate_model["feat_cols"]):
        raise ValueError(f"Feature count mismatch: built {X.shape[1]}, "
                         f"the model wants {len(gate_model['feat_cols'])}.")
    return predict_proba_pos(regressor_from_dict(gate_model), X)


# ── grid-search ──────────────────────────────

def _policy_stats(p1, p5, p10, t1, t5, t10, c0, c5, c10, c20):
    go5  = p1 > t1
    go10 = go5 & (p5 > t5)
    go20 = go10 & (p10 > t10)
    budget = np.select([~go5, go5 & ~go10, go10 & ~go20, go20], [0, 5, 10, 20])
    cost   = np.select([budget == 0, budget == 5, budget == 10, budget == 20], [1, 5, 10, 20])
    corr   = np.select([budget == 0, budget == 5, budget == 10, budget == 20], [c0, c5, c10, c20])
    return float(corr.mean()), float(cost.mean()), budget


def grid_search(p1, p5, p10, c0, c5, c10, c20, taus):
    best = {"r1": -np.inf, "avg": np.inf, "tau1": float(taus[0]),
            "tau5": float(taus[0]), "tau10": float(taus[0])}
    for t1 in tqdm(taus, desc="grid tau1"):
        go5 = p1 > t1
        for t5 in taus:
            go10 = go5 & (p5 > t5)
            for t10 in taus:
                go20 = go10 & (p10 > t10)
                budget = np.select([~go5, go5 & ~go10, go10 & ~go20, go20], [0, 5, 10, 20])
                cost   = np.select([budget == 0, budget == 5, budget == 10, budget == 20], [1, 5, 10, 20])
                corr   = np.select([budget == 0, budget == 5, budget == 10, budget == 20], [c0, c5, c10, c20])
                r1  = float(corr.mean())
                avg = float(cost.mean())
                if r1 > best["r1"] or (r1 == best["r1"] and avg < best["avg"]):
                    best.update(r1=r1, avg=avg, tau1=float(t1), tau5=float(t5), tau10=float(t10))
    return best


def run(val_csv, model, matcher, models_dir=None, tau_step=0.02, k_full=20, out_dir=None):
    model, matcher = canon_model(model), canon_matcher(matcher)
    models_dir = Path(models_dir) if models_dir else _HERE / SUBDIR
    out_dir = Path(out_dir) if out_dir else _HERE / SUBDIR

    print(f"[{FAMILY}] {model}/{matcher}  <- {val_csv}")
    print(f"  models dir: {models_dir}")
    m1, m5, m10 = load_gate_models(str(models_dir), model, matcher)

    df = load_query_seq(val_csv)
    c0  = df["correct_0"].to_numpy(int)
    c5  = df["correct_5"].to_numpy(int)
    c10 = df["correct_10"].to_numpy(int)
    c20 = df["correct_20"].to_numpy(int)
    print(f"  N queries: {len(df)}")
    print(f"  baseline R@1={c0.mean()*100:.2f}%  top5={c5.mean()*100:.2f}%  "
          f"top10={c10.mean()*100:.2f}%  top20(full)={c20.mean()*100:.2f}%")

    p1  = gate_proba(df, m1,  FEATURES_CONTINUE_1)
    p5  = gate_proba(df, m5,  FEATURES_CONTINUE_5)
    p10 = gate_proba(df, m10, FEATURES_CONTINUE_10)

    taus = np.round(np.arange(0.0, 1.0 + tau_step / 2, tau_step), 4)
    print(f"  grid tau in [0,1] step {tau_step}  ({len(taus)}^3 = {len(taus)**3} combinations)")
    best = grid_search(p1, p5, p10, c0, c5, c10, c20, taus)

    r1, avg, budget = _policy_stats(p1, p5, p10, best["tau1"], best["tau5"], best["tau10"],
                                    c0, c5, c10, c20)
    dist = {b: float(np.mean(budget == b) * 100) for b in (0, 5, 10, 20)}
    saving = 100.0 * (1.0 - avg / k_full)
    print(f"\n  CHOSEN: tau1={best['tau1']:.2f}  tau5={best['tau5']:.2f}  tau10={best['tau10']:.2f}")
    print(f"          adaptive R@1={r1*100:.2f}%  matches/query={avg:.2f}  saving vs full={saving:.1f}%")
    print(f"          stop  top1={dist[0]:.1f}%  top5={dist[5]:.1f}%  "
          f"top10={dist[10]:.1f}%  top20={dist[20]:.1f}%")

    thr = {"tau1": float(best["tau1"]), "tau5": float(best["tau5"]), "tau10": float(best["tau10"]),
           "r1_adaptive_pct": r1 * 100, "matches_per_query": avg, "saving_pct": saving,
           "stop_top1_pct": dist[0], "stop_top5_pct": dist[5],
           "stop_top10_pct": dist[10], "stop_top20_pct": dist[20]}
    sel = {"family": FAMILY, "method": "sequential", "model": model, "matcher": matcher,
           "val_csv": val_tag(val_csv), "n_queries": int(len(df)),
           "base_r1_pct": float(c0.mean() * 100), "full_rerank_r1_pct": float(c20.mean() * 100),
           "adaptive_r1_pct": r1 * 100, "reranked_pct": 100.0 - dist[0],
           "matches_per_query": avg, "saving_pct": saving,
           "params": (f"tau1={best['tau1']:.2f};tau5={best['tau5']:.2f};tau10={best['tau10']:.2f};"
                      f"tau_step={tau_step};stop_top1={dist[0]:.1f}%;stop_top5={dist[5]:.1f}%;"
                      f"stop_top10={dist[10]:.1f}%;stop_top20={dist[20]:.1f}%")}
    paths = [
        write_threshold_csv(out_dir / f"threshold_{model}_{matcher}.csv", thr),
        write_selection_csv(out_dir / f"selection_{model}_{matcher}.csv", sel),
        upsert_summary(sel),
    ]
    print_written(paths)
    return sel


def parse_args():
    p = argparse.ArgumentParser(description="Validation — sequential (multi-feature cascade 1->5->10->20)")
    p.add_argument("--val-csv",    required=True, help="validation candidate-level CSV chosen by the user")
    p.add_argument("--model",      required=True, help="cosplace or megaloc")
    p.add_argument("--matcher",    required=True, help="superpoint-lg or loftr")
    p.add_argument("--models-dir", default=None, help="folder with the 3 gate JSONs (default: validation/sequential/)")
    p.add_argument("--tau-step",   type=float, default=0.02, help="tau grid step (default 0.02)")
    p.add_argument("--k-full",     type=int, default=20, help="max budget (default 20)")
    p.add_argument("--out-dir",    default=None, help="default: validation/sequential/")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.val_csv, a.model, a.matcher, models_dir=a.models_dir, tau_step=a.tau_step,
        k_full=a.k_full, out_dir=a.out_dir)
