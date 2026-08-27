"""
train_logistic.py — TRAINING-ONLY per la famiglia logistica su num_inliers_top1
(hard | help | cost_sensitive).

Allena sul candidate-level CSV di TRAINING scelto dall'utente e salva il/i
model.json in validation/<subdir>/, pronti per essere validati con:
    validation/logistic.py --method <method> --val-csv <candidate_level_val.csv> ...

Non fa nessuna scelta di soglia: quella e' compito di validation/logistic.py,
che gira su un dataset DIVERSO (quello di validation).

Formato di output (deve combaciare con methods/logistic.py):
    hard / help      -> JSON PIATTO (un solo regressore)
    cost_sensitive    -> JSON ANNIDATO {"feature_sets": {"num_inliers": {"regressors": {"help":..., "hurts":...}}}}

Usage:
    python VPR-Adaptive-ReRanking/train_logistic.py --method hard \
        --train-csv <candidate_level_train.csv> --model cosplace --matcher superpoint-lg
    python VPR-Adaptive-ReRanking/train_logistic.py --method help \
        --train-csv <candidate_level_train.csv> --model cosplace --matcher superpoint-lg
    python VPR-Adaptive-ReRanking/train_logistic.py --method cost_sensitive \
        --train-csv <candidate_level_train.csv> --model cosplace --matcher superpoint-lg
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

# metodo -> (sottocartella validation, target/i da allenare, template file)
METHODS = {
    "hard":           ("logistic_hard",           ["hard"],         "model_{model}_{matcher}.json"),
    "help":           ("logistic_help",           ["helps"],        "model_{model}_{matcher}.json"),
    "cost_sensitive": ("logistic_cost_sensitive", ["helps", "hurts"], "model_logistic_cost_sensitive_{model}_{matcher}.json"),
}
# nome target -> nome regressore nel JSON di output
TARGET_TO_KEY = {"hard": "hard", "helps": "help", "hurts": "hurts"}


def _fit(df, target_col):
    X = df[FEAT_COLS].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=int)
    if len(set(y)) < 2:
        raise RuntimeError(
            f"'{target_col}': una sola classe presente ({int(y.sum())}/{len(y)} positivi) "
            "-> regressore non allenabile su questo training set")
    clf = fit_regressor(X, y)
    print(f"  {target_col:<6s}  positivi={int(y.sum())}/{len(y)}")
    return regressor_to_dict(clf, FEAT_COLS)


def train_and_save(method, train_csv, out_dir, model, matcher, k=20):
    if method not in METHODS:
        raise ValueError(f"method sconosciuto '{method}' (scegli tra {sorted(METHODS)})")
    subdir, targets, fname_tmpl = METHODS[method]

    print(f"[{method}] TRAINING da {train_csv}  (feat_cols={FEAT_COLS})")
    # niente SU in gioco: la l2_distance non serve, k allineato al numero di
    # candidati per query (default 20, come --num-preds nel resto della pipeline)
    df = load_query_level(train_csv, k=k, needs_l2=False)
    print(f"  N query: {len(df)}")

    regressors = {TARGET_TO_KEY[t]: _fit(df, t) for t in targets}

    out_dir = Path(out_dir if out_dir is not None else _HERE / "validation" / subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname_tmpl.format(model=model, matcher=matcher)

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
        # hard/help: JSON piatto, un solo regressore (methods/logistic.py lo
        # passa cosi' com'e' a apply_sigmoid, senza scartare wrapper)
        payload = regressors[TARGET_TO_KEY[targets[0]]]

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  -> {out_path}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Training — logistic su num_inliers_top1 (hard | help | cost_sensitive)")
    p.add_argument("--method",    required=True, choices=sorted(METHODS.keys()))
    p.add_argument("--train-csv", required=True, help="candidate-level CSV di training (file o dir)")
    p.add_argument("--model",     required=True, help="cosplace or megaloc")
    p.add_argument("--matcher",   required=True, help="superpoint-lg or loftr")
    p.add_argument("--k",         type=int, default=20,
                   help="candidati minimi per query richiesti (default: 20, come --num-preds)")
    p.add_argument("--out-dir",   default=None,
                   help="default: validation/<subdir>/ (a seconda di --method)")
    return p.parse_args()


def main():
    a = parse_args()
    train_and_save(a.method, a.train_csv, a.out_dir, a.model, a.matcher, a.k)


if __name__ == "__main__":
    main()
