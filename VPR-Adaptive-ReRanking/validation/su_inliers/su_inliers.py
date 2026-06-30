"""
validation/su_inliers — SOLA VALIDATION del metodo SU+inliers (['SU','inliers']).

NON allena. Legge training/su_inliers/model.json (default) e cerca le soglie sul
dataset di VALIDATION indicato con --val-csv. Scrive threshold.csv qui.

Uso:
    python VPR-Adaptive-ReRanking/validation/su_inliers/su_inliers.py \
        --val-csv <dir-o-file.csv>
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))            # validation/
sys.path.append(str(_HERE.parent.parent))     # VPR-Adaptive-ReRanking/  (per _common)
from _su_validation import validate_and_save

_DEFAULT_MODEL = _HERE.parent.parent / "training" / "su_inliers" / "model.json"


def parse_args():
    p = argparse.ArgumentParser(description="Validation — SU+inliers (solo grid-search)")
    p.add_argument("--val-csv", required=True,
                   help="CSV candidate-level di validation scelto dall'utente (dir o file)")
    p.add_argument("--model-json", default=str(_DEFAULT_MODEL),
                   help=f"model.json del training (default: {_DEFAULT_MODEL})")
    p.add_argument("--out-dir", default=str(_HERE),
                   help="dove scrivere threshold.csv (default: questa cartella)")
    return p.parse_args()


def main(args):
    validate_and_save(args.out_dir, args.model_json, args.val_csv)


if __name__ == "__main__":
    main(parse_args())
