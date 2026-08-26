"""
validation/logistic.py — logistic family on num_inliers_top1 (hard | help | cost_sensitive).
Mirror of methods/logistic.py.

No training. Loads the trained regressor JSON (downloaded from Google Drive by
download_models.py into validation/<subdir>/) and grid-searches on the
validation CSV the parameters that maximise the adaptive R@1:
  hard            rerank if P(hard) > tau                 tau in [0,1] step 0.01
  help            rerank if P(help) > tau                 tau in [0,1] step 0.01
  cost_sensitive  rerank if P(help) - alpha*P(hurts) > tau  alpha in [0,5] step 0.1, tau in [-1,1] step 0.01

Writes in validation/<subdir>/:
  threshold_<model>_<matcher>.csv   tau[,alpha], r1_adaptive_pct, reranked_pct, saving_pct  (read by deploy)
  selection_<model>_<matcher>.csv   method, val_csv, metrics, params
  sweep_<model>_<matcher>.csv       full grid explored
and updates validation/summary.csv.

Usage (Colab cell):
    !python VPR-Adaptive-ReRanking/validation/logistic.py --method help \\
        --val-csv /content/drive/MyDrive/VPR/candidate_level/val_cosplace_superpoint-lg.csv \\
        --model cosplace --matcher superpoint-lg
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
from _su_validation import run_validation, TAUS_GRID, TAUS_GRID_PROB, ALPHAS_GRID
from _outputs import (canon_model, canon_matcher, val_tag, cost_stats, write_threshold_csv,
                      write_selection_csv, upsert_summary, print_written)

FAMILY = "logistic"

# method -> (subdir, model JSON template, criterion, taus grid, alphas grid)
METHODS = {
    "hard":           ("logistic_hard",           "model_{model}_{matcher}.json",
                       "P(hard)",           TAUS_GRID_PROB, None),
    "help":           ("logistic_help",           "model_{model}_{matcher}.json",
                       "P(help)",           TAUS_GRID_PROB, None),
    "cost_sensitive": ("logistic_cost_sensitive", "model_logistic_cost_sensitive_{model}_{matcher}.json",
                       "P(help)-aP(hurts)", TAUS_GRID,      ALPHAS_GRID),
}


def run(method, val_csv, model, matcher, model_json=None, top_k=20, out_dir=None):
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}' (choose from {sorted(METHODS)})")
    model, matcher = canon_model(model), canon_matcher(matcher)
    subdir, tmpl, criterion, taus, alphas = METHODS[method]
    out_dir = Path(out_dir) if out_dir else _HERE / subdir
    model_json = Path(model_json) if model_json else _HERE / subdir / tmpl.format(model=model, matcher=matcher)
    if not model_json.exists():
        raise FileNotFoundError(
            f"model JSON not found: {model_json}\n"
            "  -> run validation/download_models.py (downloads the trained regressors from Google Drive)"
            " or pass --model-json")

    print(f"[{FAMILY}/{method}] {model}/{matcher}  <- {val_csv}")
    res = run_validation(model_json, val_csv, criteria=(criterion,), taus=taus,
                         alphas=alphas if alphas is not None else ALPHAS_GRID)
    if criterion not in res["criteria"]:
        raise RuntimeError(f"criterion {criterion} not calibrated (regressor missing in {model_json})")
    best = res["criteria"][criterion]
    mpq, saving = cost_stats(best["pct_val"], top_k=top_k, top1_cost=1.0)

    thr = {}
    if "alpha" in best:
        thr["alpha"] = float(best["alpha"])
    thr["tau"] = float(best["tau"])
    thr.update({"r1_adaptive_pct": float(best["r1_val"]), "reranked_pct": float(best["pct_val"]),
                "saving_pct": saving})
    params = (f"alpha={best['alpha']:.2f};" if "alpha" in best else "") + f"tau={best['tau']:.2f}"
    sel = {"family": FAMILY, "method": method, "model": model, "matcher": matcher,
           "val_csv": val_tag(val_csv), "n_queries": res["n_queries"],
           "base_r1_pct": res["base_r1_pct"], "full_rerank_r1_pct": res["full_rerank_r1_pct"],
           "adaptive_r1_pct": float(best["r1_val"]), "reranked_pct": float(best["pct_val"]),
           "matches_per_query": mpq, "saving_pct": saving,
           "params": f"criterion={criterion};{params};model_json={model_json.name}"}

    paths = [
        write_threshold_csv(out_dir / f"threshold_{model}_{matcher}.csv", thr),
        write_selection_csv(out_dir / f"selection_{model}_{matcher}.csv", sel),
    ]
    sweep_path = out_dir / f"sweep_{model}_{matcher}.csv"
    res["sweeps"][criterion].to_csv(sweep_path, index=False)
    paths.append(sweep_path)
    paths.append(upsert_summary(sel))

    print(f"  {method}: {params}  R@1={best['r1_val']:.2f}%  "
          f"(base {res['base_r1_pct']:.2f}%, full {res['full_rerank_r1_pct']:.2f}%)  "
          f"reranked={best['pct_val']:.1f}%  saving={saving:.2f}%")
    print_written(paths)
    return sel


def parse_args():
    p = argparse.ArgumentParser(description="Validation — logistic on num_inliers (hard | help | cost_sensitive)")
    p.add_argument("--method",     required=True, choices=sorted(METHODS.keys()))
    p.add_argument("--val-csv",    required=True, help="validation candidate-level CSV chosen by the user")
    p.add_argument("--model",      required=True, help="cosplace or megaloc")
    p.add_argument("--matcher",    required=True, help="superpoint-lg or loftr")
    p.add_argument("--model-json", default=None, help="default: validation/<subdir>/<template>_<model>_<matcher>.json")
    p.add_argument("--top-k",      type=int, default=20)
    p.add_argument("--out-dir",    default=None, help="default: validation/<subdir of the method>")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.method, a.val_csv, a.model, a.matcher, model_json=a.model_json, top_k=a.top_k, out_dir=a.out_dir)
