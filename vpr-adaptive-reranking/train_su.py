"""
train_su.py — TRAINING-ONLY for the SU family (su | su_inliers).

Fits the 3 logistic regressors (hard, help, hurts) on the training
candidate-level CSV chosen by the user and writes model.json in
validation/<subdir>/, ready to be validated with:
    validation/su.py --features <features> --val-csv <candidate_level_val.csv> ...

It chooses NO threshold: that is the job of validation/su.py, which runs on the
validation dataset and takes the model.json produced here as input.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
from _common import load_query_level, fit_regressor, regressor_to_dict

SU_K_DEFAULT     = 10
SU_ALPHA_DEFAULT = 0.5

FEATURES = {
    "su":         ("SU",         ["SU"],            "model_su_{model}_{matcher}.json"),
    "su_inliers": ("SU+inliers", ["SU", "inliers"], "model_su_num_inliers_{model}_{matcher}.json"),
}

TARGET_COLUMN = {"hard": "hard", "help": "helps", "hurts": "hurts"}


def train_and_save(features, train_csv, out_dir, model, matcher,
                    k=SU_K_DEFAULT, alpha=SU_ALPHA_DEFAULT):
    if features not in FEATURES:
        raise ValueError(f"unknown features '{features}' (choose among {sorted(FEATURES)})")
    feature_set, feat_cols, fname_tmpl = FEATURES[features]
    needs_l2 = "SU" in feat_cols   # always true for su/su_inliers
    print(f"[{features}] TRAINING from {train_csv}  (feat_cols={feat_cols})")
    df = load_query_level(train_csv, k=k, alpha=alpha, needs_l2=needs_l2)
    X = df[feat_cols].to_numpy(dtype=float)
    print(f"  N queries: {len(df)}")
    regressors = {}
    for target, col in TARGET_COLUMN.items():
        y = df[col].to_numpy(dtype=int)
        if len(set(y)) < 2:
            print(f"  [skip] {target}: only one class present in '{col}', regressor not trainable")
            continue
        clf = fit_regressor(X, y)
        regressors[target] = regressor_to_dict(clf, feat_cols)
        print(f"  {target:<6s} (target='{col}')  positives={int(y.sum())}/{len(y)}")
    if not regressors:
        raise RuntimeError("No regressor trained: check the training CSV.")
    out = {
        "metadata": {
            "vpr_model": model, "matcher": matcher, "feature_set": feature_set,
            "train_csv": str(train_csv), "n_queries_train": int(len(df)),
            "su_k": k, "su_alpha": alpha,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "feature_sets": {feature_set: {"feat_cols": feat_cols, "regressors": regressors}},
    }
    out_path = Path(out_dir) / fname_tmpl.format(model=model, matcher=matcher)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> {out_path}")
    return out_path
                        
def parse_args():
    p = argparse.ArgumentParser(description="Training — SU | SU+inliers (trains hard/help/hurts)")
    p.add_argument("--features",  required=True, choices=sorted(FEATURES.keys()))
    p.add_argument("--train-csv", required=True, help="training candidate-level CSV (file or dir)")
    p.add_argument("--model",     required=True, help="cosplace or megaloc")
    p.add_argument("--matcher",   required=True, help="superpoint-lg or loftr")
    p.add_argument("--su-k",      type=int,   default=SU_K_DEFAULT)
    p.add_argument("--su-alpha",  type=float, default=SU_ALPHA_DEFAULT)
    p.add_argument("--out-dir",   default=None,
                   help="default: validation/su/ or validation/su_inliers/ (depending on --features)")
    return p.parse_args()
def main():
    a = parse_args()
    out_dir = a.out_dir or (_HERE / "validation" / a.features)
    train_and_save(a.features, a.train_csv, out_dir, a.model, a.matcher, a.su_k, a.su_alpha)
if __name__ == "__main__":
    main()
