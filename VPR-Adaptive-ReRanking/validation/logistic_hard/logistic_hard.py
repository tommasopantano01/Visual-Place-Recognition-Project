import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))            # validation/  (per _su_validation)
sys.path.append(str(_HERE.parent.parent))     # VPR-Adaptive-ReRanking/  (per _common)

from _su_validation import validate_and_save


TAUS_GRID_PROB = np.round(np.arange(0.0, 1.001, 0.01), 2)

CRITERIA = ("P(hard)",)

_DEFAULT_MODEL = _HERE.parent.parent / "training" / "logistic_hard" / "model.json"


def parse_args():
    p = argparse.ArgumentParser(description="Validation - logistic_hard (P(hard)>tau)")

    p.add_argument(
        "--val-csv",
        required=True,
        help="CSV candidate-level di validation scelto dall'utente, dir o file"
    )

    p.add_argument(
        "--model-json",
        default=str(_DEFAULT_MODEL),
        help=f"model.json del training, default: {_DEFAULT_MODEL}"
    )

    p.add_argument(
        "--out-dir",
        default=str(_HERE),
        help="dove scrivere threshold.csv, default: questa cartella"
    )

    p.add_argument(
        "--model",
        required=True,
        help="cosplace or megaloc"
    )

    p.add_argument(
        "--matcher",
        required=True,
        help="superpoint-lg or loftr"
    )

    return p.parse_args()


def main(args):
    validate_and_save(
        args.out_dir,
        args.model_json,
        args.val_csv,
        args.model,
        args.matcher,
        criteria=CRITERIA,
        taus=TAUS_GRID_PROB,
    )


if __name__ == "__main__":
    main(parse_args())
