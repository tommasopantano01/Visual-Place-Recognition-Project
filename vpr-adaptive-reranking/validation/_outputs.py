"""
_outputs.py — Naming conventions and output writers shared by every
validation script (thresholds.py, logistic.py, su.py, sequential.py).

Every validation run writes, inside validation/<subdir>/:
  threshold_<model>_<matcher>.csv   ONE row, NUMERIC columns only.
                                    This is the file read by the deploy
                                    scripts in methods/ (via _common.load_threshold_csv).
  selection_<model>_<matcher>.csv   ONE row, human readable: method, validation
                                    CSV used, metrics and chosen parameters.
  sweep_<model>_<matcher>*.csv      the full grid explored (for plots/tables).

and upserts one row per (method, model, matcher) into validation/summary.csv,
so that all validated thresholds and parameters are visible in a single file.

Model / matcher names are normalised (aliases below) so that the output file
names are always canonical, e.g. "sp-lg" -> "superpoint-lg".
"""
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = VALIDATION_DIR / "summary.csv"

# canonical names used in every file name
MODEL_ALIASES = {
    "cosplace": "cosplace",
    "megaloc":  "megaloc",
}
MATCHER_ALIASES = {
    "superpoint-lg": "superpoint-lg",
    "superpoint.lg": "superpoint-lg",
    "superpoint_lg": "superpoint-lg",
    "sp-lg":         "superpoint-lg",
    "splg":          "superpoint-lg",
    "loftr":         "loftr",
    "loft":          "loftr",
}

SUMMARY_COLUMNS = [
    "family", "method", "model", "matcher", "val_csv", "n_queries",
    "base_r1_pct", "full_rerank_r1_pct", "adaptive_r1_pct",
    "reranked_pct", "matches_per_query", "saving_pct", "params", "updated_at",
]


def canon_model(name):
    key = str(name).strip().lower()
    return MODEL_ALIASES.get(key, key)


def canon_matcher(name):
    key = str(name).strip().lower()
    return MATCHER_ALIASES.get(key, key)


def val_tag(val_csv):
    """Short identifier of the validation dataset (file name without extension)."""
    p = Path(str(val_csv))
    return p.stem if p.suffix else p.name


def cost_stats(reranked_pct, top_k=20, top1_cost=1.0):
    """matches_per_query and saving_pct given the % of reranked queries.
    top1_cost = 1 when the decision needs the top-1 image matching (every
    method on num_inliers), 0 when it does not (pure SU: decision from
    retrieval only)."""
    frac = float(reranked_pct) / 100.0
    matches_per_query = top1_cost * (1.0 - frac) + top_k * frac
    saving_pct = 100.0 * (1.0 - matches_per_query / top_k)
    return matches_per_query, saving_pct


def write_threshold_csv(path, values):
    """ONE row, numeric columns only (deploy reads every column as float)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(list(values.keys()))
        w.writerow([_fmt(v) for v in values.values()])
    return path


def write_selection_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([{k: _fmt(v) for k, v in row.items()}]).to_csv(path, index=False)
    return path


def upsert_summary(row, summary_csv=SUMMARY_CSV):
    """Insert or replace the row with the same (method, model, matcher)."""
    row = {c: row.get(c, "") for c in SUMMARY_COLUMNS}
    row["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if os.path.exists(summary_csv) and os.path.getsize(summary_csv) > 0:
        df = pd.read_csv(summary_csv, dtype=str, keep_default_na=False)
        for c in SUMMARY_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        df = df[SUMMARY_COLUMNS]
        keep = ~((df["method"] == row["method"]) & (df["model"] == row["model"])
                 & (df["matcher"] == row["matcher"]))
        df = df[keep]
    else:
        df = pd.DataFrame(columns=SUMMARY_COLUMNS)
    new = pd.DataFrame([{k: _fmt(v) for k, v in row.items()}], columns=SUMMARY_COLUMNS)
    df = pd.concat([df, new], ignore_index=True)
    df = df.sort_values(["family", "method", "model", "matcher"]).reset_index(drop=True)
    df.to_csv(summary_csv, index=False)
    return summary_csv


def print_written(paths):
    for p in paths:
        print(f"  -> {p}")


def _fmt(v):
    if isinstance(v, float):
        if v != v:            # NaN
            return "nan"
        return f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) >= 1e-4 or v == 0 else f"{v:.6g}"
    return str(v)
