"""
From the validation candidate-level CSV it builds the sweep of
thresholds T on num_inliers_top1 (rerank if inliers_top1 < T) and selects T
with the chosen criterion.

Writes in validation/<subdir>/ (subdir: youden | bestR1 | efficiency ):
  threshold_<model>_<matcher>.csv   threshold, r1_adaptive_pct, saving_pct   (read by deploy)
  selection_<model>_<matcher>.csv   method, val_csv, metrics, params
  sweep_<model>_<matcher>.csv       full sweep (one row per T)
and updates validation/summary.csv.
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
from _sweep import (sweep_from_candidate, select_best_r1_threshold,
                    select_youden_threshold, select_eff95_threshold)
from _outputs import (canon_model, canon_matcher, val_tag, write_threshold_csv,
                      write_selection_csv, upsert_summary, print_written)

FAMILY = "thresholds"

METHODS = {
    "youden":     ("youden",     lambda sw, r: select_youden_threshold(sw)),
    "best_r1":    ("bestR1",     lambda sw, r: select_best_r1_threshold(sw)),
    "efficiency": ("efficiency", lambda sw, r: select_eff95_threshold(sw, retention=r)),
}


def run(method, val_csv, model, matcher, top_k=20, retention=0.95, out_dir=None):
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}' (choose from {sorted(METHODS)})")
    model, matcher = canon_model(model), canon_matcher(matcher)
    subdir, selector = METHODS[method]
    out_dir = Path(out_dir) if out_dir else _HERE / subdir
    print(f"[{FAMILY}/{method}] {model}/{matcher}  <- {val_csv}")


    sweep = sweep_from_candidate(val_csv, top_k=top_k)
    selection = selector(sweep, retention)
    r = selection.iloc[0]
    T = int(r["threshold"])
    row_sw = sweep[sweep["threshold"] == T].iloc[0]
    thr = {"threshold": T, "r1_adaptive_pct": float(r["r1_adaptive_pct"]),
           "saving_pct": float(r["saving_pct"])}
    sel = {"family": FAMILY, "method": method, "model": model, "matcher": matcher,
           "val_csv": val_tag(val_csv), "n_queries": int(row_sw["n_queries"]),
           "base_r1_pct": float(row_sw["pre_r1_pct"]),
           "full_rerank_r1_pct": float(row_sw["full_rerank_r1_pct"]),
           "adaptive_r1_pct": float(r["r1_adaptive_pct"]),
           "reranked_pct": float(row_sw["reranked_pct"]),
           "matches_per_query": float(row_sw["matches_per_query"]),
           "saving_pct": float(r["saving_pct"]), "params": f"T={T}"}
    if method == "efficiency":
        sel["params"] += f";retention={retention}"

    paths = [
        write_threshold_csv(out_dir / f"threshold_{model}_{matcher}.csv", thr),
        write_selection_csv(out_dir / f"selection_{model}_{matcher}.csv", sel),
    ]
    sweep.to_csv(out_dir / f"sweep_{model}_{matcher}.csv", index=False)
    paths.append(out_dir / f"sweep_{model}_{matcher}.csv")
    paths.append(upsert_summary(sel))

    print(f"  {method}: T={T}  R@1={thr['r1_adaptive_pct']:.2f}%  "
          f"(base {sel['base_r1_pct']:.2f}%, full {sel['full_rerank_r1_pct']:.2f}%)  "
          f"reranked={sel['reranked_pct']:.1f}%  saving={thr['saving_pct']:.2f}%")
    print_written(paths)
    return sel


def parse_args():
    p = argparse.ArgumentParser(description="Validation — hard-threshold (youden | best_r1 | efficiency | local)")
    p.add_argument("--method",    required=True, choices=sorted(METHODS.keys()))
    p.add_argument("--val-csv",   required=True, help="validation candidate-level CSV (file; a dir is allowed only if query_ids are unique)")
    p.add_argument("--model",     required=True, help="cosplace or megaloc")
    p.add_argument("--matcher",   required=True, help="superpoint-lg or loftr")
    p.add_argument("--top-k",     type=int, default=20)
    p.add_argument("--retention", type=float, default=0.95, help="efficiency only: fraction of the R@1 gain to keep")
    p.add_argument("--out-dir",   default=None, help="default: validation/<subdir of the method>")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.method, a.val_csv, a.model, a.matcher, top_k=a.top_k, retention=a.retention, out_dir=a.out_dir)
