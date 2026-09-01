"""
train_logistic.py — TRAINING-ONLY for the logistic family on num_inliers_top1
(hard | help | cost_sensitive).

Trains on the TRAINING candidate-level CSV chosen by the user and writes the
model.json file(s) in validation/<subdir>/, ready to be validated with: validation/logistic.py

It chooses NO threshold: that is the job of validation/logistic.py,
which runs on a validation dataset.

Output format (must match methods/logistic.py):
    hard / help      -> FLAT JSON (a single regressor)
    cost_sensitive    -> NESTED JSON {"feature_sets": {"num_inliers": {"regressors": {"help":..., "hurts":...}}}}
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
from _common import load_query_level, fit_regressor, regressor_to_dict

FEAT_COLS = ["num_inliers_top1"]

# method -> (validation subfolder, target(s) to train, file template)
METHODS = {
    "hard":           ("logistic_hard",           ["hard"],         "model_{model}_{matcher}.json"),
    "help":           ("logistic_help",           ["helps"],        "model_{model}_{matcher}.json"),
    "cost_sensitive": ("logistic_cost_sensitive", ["helps", "hurts"], "model_logistic_cost_sensitive_{model}_{matcher}.json"),
}
TARGET_TO_KEY = {"hard": "hard", "helps": "help", "hurts": "hurts"}


def _fit(df, target_col):
    X = df[FEAT_COLS].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=int)
    if len(set(y)) < 2:
        raise RuntimeError(
            f"'{target_col}': only one class present ({int(y.sum())}/{len(y)} positives) "
            "-> regressor not trainable on this training set")
    clf = fit_regressor(X, y)
    print(f"  {target_col:<6s}  positives={int(y.sum())}/{len(y)}")
    return regressor_to_dict(clf, FEAT_COLS)


def train_and_save(method, train_csv, out_dir, model, matcher, k=20):
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}' (choose among {sorted(METHODS)})")
    subdir, targets, fname_tmpl = METHODS[method]

    print(f"[{method}] TRAINING from {train_csv}  (feat_cols={FEAT_COLS})")

    df = load_query_level(train_csv, k=k, needs_l2=False)    # l2_distance is not needed, k aligned with the number of candidates per query

    print(f"  N queries: {len(df)}")

    regressors = {TARGET_TO_KEY[t]: _fit(df, t) for t in targets}

    out_dir = Path(out_dir if out_dir is not None else _HERE / "validation" / subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname_tmpl.format(model=model, matcher=matcher)
    # payload: content of the JSON file. in the first case we layout the values by hand, in the second it is just one regressor
    if method == "cost_sensitive":
        payload = {
            "metadata": {
                "vpr_model": model, "matcher": matcher, "method": method,
                "train_csv": str(train_csv), "n_queries_train": int(len(df)),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "feature_sets": {"num_inliers": {"feat_cols": FEAT_COLS, "regressors": regressors}},
        }
    else:
        payload = regressors[TARGET_TO_KEY[targets[0]]]

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  -> {out_path}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Training — logistic on num_inliers_top1 (hard | help | cost_sensitive)")
    p.add_argument("--method",    required=True, choices=sorted(METHODS.keys()))
    p.add_argument("--train-csv", required=True, help="training candidate-level CSV (file or dir)")
    p.add_argument("--model",     required=True, help="cosplace or megaloc")
    p.add_argument("--matcher",   required=True, help="superpoint-lg or loftr")
    p.add_argument("--k",         type=int, default=20, help="minimum candidates per query required (default: 20, same as --num-preds)")
    p.add_argument("--out-dir",   default=None, help="default: validation/<subdir>/ (depending on --method)")
    return p.parse_args()


def main():
    a = parse_args()
    train_and_save(a.method, a.train_csv, a.out_dir, a.model, a.matcher, a.k)


if __name__ == "__main__":
    main()
