import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import load_threshold_csv, run_scalar_method

_ARR_DIR = Path(__file__).resolve().parent.parent

# nome metodo -> sottocartella in validation/
METHODS = {
    "youden":     "youden",
    "best_r1":    "bestR1",
    "efficiency": "efficiency",
    "local":      "local",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive reranking — threshold-based (youden | best_r1 | efficiency | local)")
    parser.add_argument("--method",     required=True, choices=sorted(METHODS.keys()))
    parser.add_argument("--preds-dir",  required=True)
    parser.add_argument("--model",      required=True, help="cosplace or megaloc")
    parser.add_argument("--matcher",    required=True)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--im-size",    type=int, default=512)
    parser.add_argument("--num-preds",  type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main(args):
    validation_dir = _ARR_DIR / "validation" / METHODS[args.method]
    threshold_csv = validation_dir / f"threshold_{args.model}_{args.matcher}.csv"
    threshold = int(load_threshold_csv(threshold_csv)["threshold"])
    print(f"threshold ({args.method}) = {threshold}  [{args.model}/{args.matcher}]")
    run_scalar_method(args.preds_dir, threshold, args.matcher, args.device,
                      args.im_size, args.num_preds, args.output_dir)


if __name__ == "__main__":
    main(parse_args())
