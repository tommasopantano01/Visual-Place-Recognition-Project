"""
methods/Thresholds.py — Deploy della famiglia hard-threshold
(youden | best_r1 | efficiency | local).

Regola: rerank del top-20 se num_inliers_top1 < T, altrimenti si tiene
l'ordine di retrieval. T viene letto da validation/<subdir>/threshold_<model>_<matcher>.csv.

    python VPR-Adaptive-ReRanking/methods/Thresholds.py --method youden \
        --preds-dir <preds/> --model cosplace --matcher superpoint-lg \
        --inliers-dir <top20 .torch/> --output-dir <out/>
"""
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
    parser = argparse.ArgumentParser(
        description="Adaptive reranking — threshold-based (youden | best_r1 | efficiency | local)")
    parser.add_argument("--method",     required=True, choices=sorted(METHODS.keys()))
    parser.add_argument("--preds-dir",  required=True)
    parser.add_argument("--model",      required=True, help="cosplace or megaloc")
    parser.add_argument("--matcher",    required=True, help="superpoint-lg or loftr")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--im-size",    type=int, default=512)
    parser.add_argument("--num-preds",  type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inliers-dir", default=None,
                        help="OFFLINE: cartella con i .torch top-20 gia' calcolati "
                             "(niente image matching, solo lettura)")
    parser.add_argument("--threshold-csv", default=None,
                        help="default: validation/<subdir>/threshold_<model>_<matcher>.csv")
    return parser.parse_args()


def main(args):
    threshold_csv = args.threshold_csv or (
        _ARR_DIR / "validation" / METHODS[args.method] / f"threshold_{args.model}_{args.matcher}.csv")
    if not Path(threshold_csv).exists():
        raise FileNotFoundError(
            f"Soglia non trovata: {threshold_csv}\n"
            f"  -> esegui prima: validation/thresholds.py --method {args.method} "
            f"--model {args.model} --matcher {args.matcher} --val-csv <candidate_level_val.csv>")

    threshold = int(load_threshold_csv(threshold_csv)["threshold"])
    print(f"threshold ({args.method}) = {threshold}  [{args.model}/{args.matcher}]")
    run_scalar_method(args.preds_dir, threshold, args.matcher, args.device,
                      args.im_size, args.num_preds, args.output_dir,
                      inliers_dir=args.inliers_dir)


if __name__ == "__main__":
    main(parse_args())
