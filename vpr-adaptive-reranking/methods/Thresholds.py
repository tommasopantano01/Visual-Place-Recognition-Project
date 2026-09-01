"""
Rule: rerank the top-20 if num_inliers_top1 < T, otherwise keep the retrieval order.
T is read from validation/<subdir>/threshold_<model>_<matcher>.csv.
"""
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import load_threshold_csv, run_scalar_method
_ARR_DIR = Path(__file__).resolve().parent.parent
METHODS = {
    "youden":     "youden",
    "best_r1":    "bestR1",
    "efficiency": "efficiency",
}
def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive reranking — threshold-based (youden | best_r1 | efficiency)")
    parser.add_argument("--method",     required=True, choices=sorted(METHODS.keys()))
    parser.add_argument("--preds-dir",  required=True)
    parser.add_argument("--model",      required=True, help="cosplace or megaloc")
    parser.add_argument("--matcher",    required=True, help="superpoint-lg or loftr")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--im-size",    type=int, default=512)
    parser.add_argument("--num-preds",  type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inliers-dir", default=None,
                        help="OFFLINE: folder with the already computed top-20 .torch files "
                             "(no image matching, read only)")
    parser.add_argument("--threshold-csv", default=None,
                        help="default: validation/<subdir>/threshold_<model>_<matcher>.csv")
    return parser.parse_args()
def main(args):
    threshold_csv = args.threshold_csv or (
        _ARR_DIR / "validation" / METHODS[args.method] / f"threshold_{args.model}_{args.matcher}.csv")
    if not Path(threshold_csv).exists():
        raise FileNotFoundError(
            f"Threshold not found: {threshold_csv}\n"
            f"  -> run first: validation/thresholds.py --method {args.method} "
            f"--model {args.model} --matcher {args.matcher} --val-csv <candidate_level_val.csv>")
    threshold = int(load_threshold_csv(threshold_csv)["threshold"])
    print(f"threshold ({args.method}) = {threshold}  [{args.model}/{args.matcher}]")
    run_scalar_method(args.preds_dir, threshold, args.matcher, args.device,
                      args.im_size, args.num_preds, args.output_dir,
                      inliers_dir=args.inliers_dir)
if __name__ == "__main__":
    main(parse_args())
