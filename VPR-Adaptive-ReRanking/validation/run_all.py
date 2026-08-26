"""
validation/run_all.py — Run EVERY validation method on EVERY (model, matcher)
pair in one command, so that all threshold_<model>_<matcher>.csv files and
validation/summary.csv come out automatically.

The validation dataset is chosen by the user through a path TEMPLATE with the
placeholders {model} and {matcher} (glob wildcards * are allowed), e.g.
    --val-csv-template "/content/drive/MyDrive/VPR/candidate_level/val_{model}_{matcher}.csv"
    --val-csv-template "/content/drive/MyDrive/VPR/candidate_level/*sfxs_val*{model}*{matcher}*.csv"
Pairs whose CSV is not found are skipped and listed at the end.

Usage (Colab cell):
    !python VPR-Adaptive-ReRanking/validation/run_all.py \\
        --val-csv-template "/content/drive/MyDrive/VPR/candidate_level/val_{model}_{matcher}.csv"
    # subset of methods / pairs:
    !python VPR-Adaptive-ReRanking/validation/run_all.py --val-csv-template "..." \\
        --methods youden help su --models cosplace --matchers superpoint-lg
"""
import argparse
import sys
import traceback
from glob import glob
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
import thresholds as _thr
import logistic as _log
import su as _su
import sequential as _seq
from _outputs import SUMMARY_CSV, canon_model, canon_matcher

MODELS   = ("cosplace", "megaloc")
MATCHERS = ("superpoint-lg", "loftr")

# method key -> (family script, callable(val_csv, model, matcher, args))
METHODS = {
    "youden":         ("thresholds", lambda v, m, t, a: _thr.run("youden",     v, m, t, top_k=a.top_k)),
    "best_r1":        ("thresholds", lambda v, m, t, a: _thr.run("best_r1",    v, m, t, top_k=a.top_k)),
    "efficiency":     ("thresholds", lambda v, m, t, a: _thr.run("efficiency", v, m, t, top_k=a.top_k, retention=a.retention)),
    "local":          ("thresholds", lambda v, m, t, a: _thr.run("local",      v, m, t, top_k=a.top_k)),
    "hard":           ("logistic",   lambda v, m, t, a: _log.run("hard",           v, m, t, top_k=a.top_k)),
    "help":           ("logistic",   lambda v, m, t, a: _log.run("help",           v, m, t, top_k=a.top_k)),
    "cost_sensitive": ("logistic",   lambda v, m, t, a: _log.run("cost_sensitive", v, m, t, top_k=a.top_k)),
    "su":             ("su",         lambda v, m, t, a: _su.run("su",         v, m, t, top_k=a.top_k)),
    "su_inliers":     ("su",         lambda v, m, t, a: _su.run("su_inliers", v, m, t, top_k=a.top_k)),
    "sequential":     ("sequential", lambda v, m, t, a: _seq.run(v, m, t, tau_step=a.tau_step, k_full=a.top_k)),
}
DEFAULT_METHODS = [k for k in METHODS if k != "local"]   # local: not implemented yet


def resolve_val_csv(template, model, matcher):
    """Template -> existing file (glob allowed). None if not found / ambiguous."""
    pattern = template.format(model=model, matcher=matcher)
    hits = sorted(glob(pattern)) if any(ch in pattern for ch in "*?[") else ([pattern] if Path(pattern).exists() else [])
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        print(f"  [ambiguous] {pattern} matches {len(hits)} files: {hits}")
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Run all validation methods on all (model, matcher) pairs")
    p.add_argument("--val-csv-template", required=True,
                   help="path of the validation candidate-level CSV with {model} and {matcher} placeholders (glob * allowed)")
    p.add_argument("--methods",   nargs="+", default=DEFAULT_METHODS, choices=list(METHODS.keys()))
    p.add_argument("--models",    nargs="+", default=list(MODELS))
    p.add_argument("--matchers",  nargs="+", default=list(MATCHERS))
    p.add_argument("--top-k",     type=int, default=20)
    p.add_argument("--retention", type=float, default=0.95, help="efficiency only")
    p.add_argument("--tau-step",  type=float, default=0.02, help="sequential only")
    return p.parse_args()


def main():
    a = parse_args()
    status = []          # (method, model, matcher, "ok" | "skipped: ..." | "FAILED: ...")
    for model in a.models:
        for matcher in a.matchers:
            model, matcher = canon_model(model), canon_matcher(matcher)
            val_csv = resolve_val_csv(a.val_csv_template, model, matcher)
            print("\n" + "=" * 78)
            print(f"{model} / {matcher}   val_csv = {val_csv}")
            print("=" * 78)
            if val_csv is None:
                for method in a.methods:
                    status.append((method, model, matcher, "skipped: validation CSV not found"))
                continue
            for method in a.methods:
                print(f"\n--- {method} ---")
                try:
                    METHODS[method][1](val_csv, model, matcher, a)
                    status.append((method, model, matcher, "ok"))
                except FileNotFoundError as e:
                    status.append((method, model, matcher, f"skipped: {str(e).splitlines()[0]}"))
                    print(f"  [skipped] {e}")
                except NotImplementedError as e:
                    status.append((method, model, matcher, f"skipped: {e}"))
                    print(f"  [skipped] {e}")
                except Exception as e:
                    status.append((method, model, matcher, f"FAILED: {type(e).__name__}: {e}"))
                    traceback.print_exc()

    print("\n" + "=" * 78 + "\nRESULT\n" + "=" * 78)
    w = max(len(s[0]) for s in status) if status else 10
    for method, model, matcher, st in status:
        print(f"  {method:<{w}}  {model:<9} {matcher:<14} {st}")
    n_ok = sum(1 for s in status if s[3] == "ok")
    n_fail = sum(1 for s in status if s[3].startswith("FAILED"))
    print(f"\n{n_ok} ok, {len(status) - n_ok - n_fail} skipped, {n_fail} failed.")
    print(f"All validated thresholds and parameters: {SUMMARY_CSV}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
