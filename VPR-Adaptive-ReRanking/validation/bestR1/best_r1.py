import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))     # validation/  (per _sweep)
from _sweep import sweep_from_candidate, select_best_r1_threshold, save_outputs


def parse_args():
    p = argparse.ArgumentParser(description="Validation — best_r1 (max R@1 adattiva)")
    p.add_argument("--val-csv", required=True, help="candidate-level CSV di validation (file o dir)")
    p.add_argument("--out-dir", default=str(_HERE), help="dove scrivere gli output (default: questa cartella)")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--model", required=True, help="cosplace or megaloc")
    p.add_argument("--matcher", required=True, help="superpoint-lg or loftr")
    return p.parse_args()


def main(args):
    sweep = sweep_from_candidate(args.val_csv, top_k=args.top_k)
    save_outputs(args.out_dir, sweep, select_best_r1_threshold(sweep), args.model, args.matcher)


if __name__ == "__main__":
    main(parse_args())
