import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import load_threshold_csv, run_scalar_method

_VALIDATION_DIR = Path(__file__).resolve().parent.parent / "validation" / "local"


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive reranking — local")
    parser.add_argument("--preds-dir",  required=True)
    parser.add_argument("--model",      required=True, help="cosplace or megaloc")
    parser.add_argument("--matcher",    required=True)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--im-size",    type=int, default=512)
    parser.add_argument("--num-preds",  type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main(args):
    threshold_csv = _VALIDATION_DIR / f"threshold_{args.model}_{args.matcher}.csv"
    threshold = int(load_threshold_csv(threshold_csv)["threshold"])
    print(f"threshold (local) = {threshold}  [{args.model}/{args.matcher}]")
    run_scalar_method(args.preds_dir, threshold, args.matcher, args.device,
                       args.im_size, args.num_preds, args.output_dir)


if __name__ == "__main__":
    main(parse_args())
