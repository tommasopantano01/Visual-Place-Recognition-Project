"""
Loads the trained regressors JSON (hard/help/hurts) and
grid-searches on the validation CSV the parameters of ALL three criteria:
  P(hard) > tau                     ->  hard_tau
  P(help) > tau                     ->  help_tau
  P(help) - alpha*P(hurts) > tau    ->  cs_alpha, cs_tau
Grids: tau in [-1,1] step 0.01, alpha in [0,5] step 0.1.

Features: su -> ['SU']; su_inliers -> ['SU', 'num_inliers'] where the second
feature is the NEGATED top-1 inlier count (same convention as methods/su.py
and as the trained regressors).

The validation CSV must contain l2_distance (candidate-level built with --z_data_path).

Writes in validation/<subdir>/:
  threshold_<model>_<matcher>.csv   hard_tau, help_tau, cs_alpha, cs_tau + metrics per criterion  (read by deploy)
  selection_<model>_<matcher>.csv   3 rows (one per criterion): val_csv, metrics, params
  sweep_<model>_<matcher>_<crit>.csv  full grid per criterion (crit: hard | help | cs)
and updates validation/summary.csv (one row per criterion).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
from _su_validation import run_validation, TAUS_GRID, ALPHAS_GRID
from _outputs import (canon_model, canon_matcher, val_tag, cost_stats, write_threshold_csv,
                      upsert_summary, print_written, _fmt)

FAMILY = "su"

# features -> (subdir, feature_set key in the JSON, model JSON template, cost of the top-1 IM)
FEATURES = {
    "su":         ("su",         "SU",         "model_su_{model}_{matcher}.json",             0.0),
    "su_inliers": ("su_inliers", "SU+inliers", "model_su_num_inliers_{model}_{matcher}.json", 1.0),
}
CRITERIA = ("P(hard)", "P(help)", "P(help)-aP(hurts)")
PREFIX = {"P(hard)": "hard", "P(help)": "help", "P(help)-aP(hurts)": "cs"}


def run(features, val_csv, model, matcher, model_json=None, top_k=20, out_dir=None):
    if features not in FEATURES:
        raise ValueError(f"unknown features '{features}' (choose from {sorted(FEATURES)})")
    model, matcher = canon_model(model), canon_matcher(matcher)
    subdir, feature_set, tmpl, top1_cost = FEATURES[features]
    out_dir = Path(out_dir) if out_dir else _HERE / subdir
    model_json = Path(model_json) if model_json else _HERE / subdir / tmpl.format(model=model, matcher=matcher)
    if not model_json.exists():
        raise FileNotFoundError(
            f"model JSON not found: {model_json}\n"
    print(f"[{FAMILY}/{features}] {model}/{matcher}  <- {val_csv}")
    res = run_validation(model_json, val_csv, criteria=CRITERIA, taus=TAUS_GRID,
                         alphas=ALPHAS_GRID, feature_set=feature_set)
    if not res["criteria"]:
        raise RuntimeError(f"no criterion calibrated (regressors missing in {model_json})")

    thr, sel_rows, paths = {}, [], []
    for crit in CRITERIA:
        pfx = PREFIX[crit]
        if crit not in res["criteria"]:
            continue
        best = res["criteria"][crit]
        mpq, saving = cost_stats(best["pct_val"], top_k=top_k, top1_cost=top1_cost)
        if "alpha" in best:
            thr[f"{pfx}_alpha"] = float(best["alpha"])
        thr[f"{pfx}_tau"] = float(best["tau"])
        thr[f"{pfx}_r1_adaptive_pct"] = float(best["r1_val"])
        thr[f"{pfx}_reranked_pct"] = float(best["pct_val"])
        thr[f"{pfx}_saving_pct"] = saving
        params = (f"alpha={best['alpha']:.2f};" if "alpha" in best else "") + f"tau={best['tau']:.2f}"
        sel_rows.append({
            "family": FAMILY, "method": f"{features}[{crit}]", "model": model, "matcher": matcher,
            "val_csv": val_tag(val_csv), "n_queries": res["n_queries"],
            "base_r1_pct": res["base_r1_pct"], "full_rerank_r1_pct": res["full_rerank_r1_pct"],
            "adaptive_r1_pct": float(best["r1_val"]), "reranked_pct": float(best["pct_val"]),
            "matches_per_query": mpq, "saving_pct": saving,
            "params": f"criterion={crit};{params};su_k={res['su_k']};su_alpha={res['su_alpha']};model_json={model_json.name}",
        })
        sweep_path = out_dir / f"sweep_{model}_{matcher}_{pfx}.csv"
        res["sweeps"][crit].to_csv(sweep_path, index=False)
        paths.append(sweep_path)

    paths.insert(0, write_threshold_csv(out_dir / f"threshold_{model}_{matcher}.csv", thr))
    sel_path = out_dir / f"selection_{model}_{matcher}.csv"
    pd.DataFrame([{k: _fmt(v) for k, v in r.items()} for r in sel_rows]).to_csv(sel_path, index=False)
    paths.insert(1, sel_path)
    for row in sel_rows:
        summary = upsert_summary(row)
    paths.append(summary)
    print_written(paths)
    return sel_rows


def parse_args():
    p = argparse.ArgumentParser(description="Validation — SU | SU+inliers (all three criteria)")
    p.add_argument("--features",   required=True, choices=sorted(FEATURES.keys()))
    p.add_argument("--val-csv",    required=True, help="validation candidate-level CSV WITH l2_distance")
    p.add_argument("--model",      required=True, help="cosplace or megaloc")
    p.add_argument("--matcher",    required=True, help="superpoint-lg or loftr")
    p.add_argument("--model-json", default=None, help="default: validation/<subdir>/<template>_<model>_<matcher>.json")
    p.add_argument("--top-k",      type=int, default=20)
    p.add_argument("--out-dir",    default=None, help="default: validation/<subdir of the features>")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.features, a.val_csv, a.model, a.matcher, model_json=a.model_json, top_k=a.top_k, out_dir=a.out_dir)
