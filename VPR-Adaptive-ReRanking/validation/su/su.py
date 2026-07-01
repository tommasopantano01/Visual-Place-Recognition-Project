"""
validation/su — SOLA VALIDATION del metodo SU (feature set ['SU']).

NON allena. Legge il model.json gia' prodotto dal training
(training/su/model.json di default) e cerca le soglie ottime sul dataset di
VALIDATION indicato dall'utente con --val-csv. Scrive
threshold_<model>_<matcher>.csv qui. --model/--matcher identificano la coppia
(retrieval, image matching) su cui e' calibrata la soglia.

Uso:
    python VPR-Adaptive-ReRanking/validation/su/su.py \
        --val-csv <dir-o-file.csv> --model cosplace --matcher superpoint-lg

    # model.json in una posizione non standard:
    python VPR-Adaptive-ReRanking/validation/su/su.py \
        --val-csv <...> --model-json <path/model.json> \
        --model cosplace --matcher superpoint-lg
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))            # validation/  (per _su_validation)
sys.path.append(str(_HERE.parent.parent))     # VPR-Adaptive-ReRanking/  (per _common)
from _su_validation import validate_and_save

# model.json prodotto dal training, posizione di default
_DEFAULT_MODEL = _HERE.parent.parent / "training" / "su" / "model.json"


def parse_args():
    p = argparse.ArgumentParser(description="Validation — SU (solo grid-search soglie)")
    p.add_argument("--val-csv", required=True,
                   help="CSV candidate-level di validation scelto dall'utente (dir o file)")
    p.add_argument("--model-json", default=str(_DEFAULT_MODEL),
                   help=f"model.json del training (default: {_DEFAULT_MODEL})")
    p.add_argument("--out-dir", default=str(_HERE),
                   help="dove scrivere threshold.csv (default: questa cartella)")
    p.add_argument("--model", required=True, help="cosplace or megaloc")
    p.add_argument("--matcher", required=True, help="superpoint-lg or loftr")
    return p.parse_args()


def main(args):
    validate_and_save(args.out_dir, args.model_json, args.val_csv, args.model, args.matcher)


if __name__ == "__main__":
    main(parse_args())
