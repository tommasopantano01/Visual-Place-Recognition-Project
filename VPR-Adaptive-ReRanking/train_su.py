"""
train_su.py — TRAINING-ONLY per la famiglia SU (su | su_inliers).

Allena i 3 regressori logistici (hard, help, hurts) sul candidate-level CSV di
TRAINING scelto dall'utente e salva model.json in validation/<subdir>/, pronto
per essere validato con:
    validation/su.py --features <features> --val-csv <candidate_level_val.csv> ...

Non fa nessuna scelta di soglia: quella e' compito di validation/su.py, che
gira su un dataset DIVERSO (quello di validation) e usa il model.json prodotto
qui come input.

Usage:
    python VPR-Adaptive-ReRanking/train_su.py --features su \
        --train-csv <candidate_level_train.csv> --model cosplace --matcher superpoint-lg
    python VPR-Adaptive-ReRanking/train_su.py --features su_inliers \
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

SU_K_DEFAULT     = 10
SU_ALPHA_DEFAULT = 0.5

# features -> (nome feature_set nel JSON, colonne, nome file di output)
FEATURES = {
    "su":         ("SU",         ["SU"],            "model_su_{model}_{matcher}.json"),
    "su_inliers": ("SU+inliers", ["SU", "inliers"], "model_su_num_inliers_{model}_{matcher}.json"),
}

# target del regressore -> colonna prodotta da load_query_level (helps, non help)
TARGET_COLUMN = {"hard": "hard", "help": "helps", "hurts": "hurts"}


def train_and_save(features, train_csv, out_dir, model, matcher,
                    k=SU_K_DEFAULT, alpha=SU_ALPHA_DEFAULT):
    if features not in FEATURES:
        raise ValueError(f"features sconosciuto '{features}' (scegli tra {sorted(FEATURES)})")
    feature_set, feat_cols, fname_tmpl = FEATURES[features]
    needs_l2 = "SU" in feat_cols   # sempre vero per su/su_inliers

    print(f"[{features}] TRAINING da {train_csv}  (feat_cols={feat_cols})")
    df = load_query_level(train_csv, k=k, alpha=alpha, needs_l2=needs_l2)
    X = df[feat_cols].to_numpy(dtype=float)
    print(f"  N query: {len(df)}")

    regressors = {}
    for target, col in TARGET_COLUMN.items():
        y = df[col].to_numpy(dtype=int)
        if len(set(y)) < 2:
            print(f"  [skip] {target}: una sola classe presente in '{col}', regressore non allenabile")
            continue
        clf = fit_regressor(X, y)
        regressors[target] = regressor_to_dict(clf, feat_cols)
        print(f"  {target:<6s} (target='{col}')  positivi={int(y.sum())}/{len(y)}")

    if not regressors:
        raise RuntimeError("Nessun regressore allenato: controlla il CSV di training.")

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
    p = argparse.ArgumentParser(description="Training — SU | SU+inliers (allena hard/help/hurts)")
    p.add_argument("--features",  required=True, choices=sorted(FEATURES.keys()))
    p.add_argument("--train-csv", required=True, help="candidate-level CSV di training (file o dir)")
    p.add_argument("--model",     required=True, help="cosplace or megaloc")
    p.add_argument("--matcher",   required=True, help="superpoint-lg or loftr")
    p.add_argument("--su-k",      type=int,   default=SU_K_DEFAULT)
    p.add_argument("--su-alpha",  type=float, default=SU_ALPHA_DEFAULT)
    p.add_argument("--out-dir",   default=None,
                   help="default: validation/su/ oppure validation/su_inliers/ (a seconda di --features)")
    return p.parse_args()


def main():
    a = parse_args()
    out_dir = a.out_dir or (_HERE / "validation" / a.features)
    train_and_save(a.features, a.train_csv, out_dir, a.model, a.matcher, a.su_k, a.su_alpha)


if __name__ == "__main__":
    main()
