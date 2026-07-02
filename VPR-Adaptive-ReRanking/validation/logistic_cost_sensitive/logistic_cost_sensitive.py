import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))            # validation/  (per _su_validation)
sys.path.append(str(_HERE.parent.parent))     # VPR-Adaptive-ReRanking/  (per _common)
from _su_validation import validate_and_save, ALPHAS_GRID, TAUS_GRID

CRITERIA = ("P(help)-aP(hurts)",)


def parse_args():
    p = argparse.ArgumentParser(
        description="Validation — logistic_cost_sensitive (P(help)-alpha*P(hurts) > tau)"
    )
    p.add_argument("--val-csv", required=True,
                   help="CSV candidate-level di validation scelto dall'utente (dir o file)")
    p.add_argument("--model-json", default=None,
                   help="path al model.json; se omesso usa model_logistic_cost_sensitive_<model>_<matcher>.json")
    p.add_argument("--out-dir", default=str(_HERE),
                   help="dove scrivere threshold.csv (default: questa cartella)")
    p.add_argument("--model", required=True, help="cosplace or megaloc")
    p.add_argument("--matcher", required=True, help="superpoint-lg or loftr")
    return p.parse_args()


def main(args):
    model_json = args.model_json
    if model_json is None:
        model_json = _HERE / f"model_logistic_cost_sensitive_{args.model}_{args.matcher}.json"

    validate_and_save(
        args.out_dir,
        model_json,
        args.val_csv,
        args.model,
        args.matcher,
        criteria=CRITERIA,
        taus=TAUS_GRID,
        alphas=ALPHAS_GRID
    )


if __name__ == "__main__":
    main(parse_args())
