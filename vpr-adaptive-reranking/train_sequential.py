"""
train_sequential.py — TRAINING-ONLY for the sequential cascade (gates 1/5/10).

Fits the 3 logistic gates on the TRAINING candidate-level CSV and writes
    validation/sequential/seq_model_continue_{1,5,10}_<model>_<matcher>.json
ready to be calibrated with:
    validation/sequential.py --val-csv <candidate_level_val.csv> --model ... --matcher ...

It chooses NO threshold: tau1/tau5/tau10 are picked by validation/sequential.py
on a DIFFERENT split.

Features and labels are built by the very functions that validation/sequential.py
uses, so training and validation can never drift apart.

Usage:
    python vpr-adaptive-reranking/train_sequential.py \
        --train-csv <candidate_level_train.csv> --model cosplace --matcher superpoint-lg
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
sys.path.append(str(_HERE / "validation"))
from _common import fit_regressor, regressor_to_dict
from sequential import (load_query_seq, FEATURES_CONTINUE_1,
                        FEATURES_CONTINUE_5, FEATURES_CONTINUE_10)
from _outputs import canon_model, canon_matcher

# gate number -> (feature list, target column)
GATES = {
    1:  (FEATURES_CONTINUE_1,  "helps_20"),
    5:  (FEATURES_CONTINUE_5,  "continue_5"),
    10: (FEATURES_CONTINUE_10, "continue_10"),
}


def train_and_save(train_csv, out_dir, model, matcher):
    model, matcher = canon_model(model), canon_matcher(matcher)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sequential] TRAINING from {train_csv}")
    df = load_query_seq(train_csv)          # features + labels, same code as validation
    print(f"  N queries: {len(df)}")

    paths = []
    for gate, (feat_cols, target) in GATES.items():
        y = df[target].to_numpy(dtype=int)
        if len(set(y)) < 2:
            raise RuntimeError(
                f"gate{gate}: only one class in '{target}' ({int(y.sum())}/{len(y)} positives) "
                "-> not trainable on this training set")
        X = df[feat_cols].to_numpy(dtype=float)
        clf = fit_regressor(X, y)
        payload = regressor_to_dict(clf, feat_cols)
        payload["metadata"] = {
            "vpr_model": model, "matcher": matcher, "gate": gate, "target": target,
            "train_csv": str(train_csv), "n_queries_train": int(len(df)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        out_path = out_dir / f"seq_model_continue_{gate}_{model}_{matcher}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  gate{gate:<2d} ({len(feat_cols)} feat, target='{target}')  "
              f"positives={int(y.sum())}/{len(y)}")
        print(f"    -> {out_path}")
        paths.append(out_path)
    return paths


def parse_args():
    p = argparse.ArgumentParser(
        description="Training — sequential cascade (gates 1 -> 5 -> 10)")
    p.add_argument("--train-csv", required=True,
                   help="training candidate-level CSV (file or dir)")
    p.add_argument("--model",     required=True, help="cosplace or megaloc")
    p.add_argument("--matcher",   required=True, help="superpoint-lg or loftr")
    p.add_argument("--out-dir",   default=None,
                   help="default: validation/sequential/")
    return p.parse_args()


def main():
    a = parse_args()
    train_and_save(a.train_csv, a.out_dir or (_HERE / "validation" / "sequential"),
                   a.model, a.matcher)


if __name__ == "__main__":
    main()
