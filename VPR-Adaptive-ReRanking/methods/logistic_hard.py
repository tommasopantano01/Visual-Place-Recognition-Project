"""
logistic_hard/logistic_hard.py — Regressione logistica P(hard | num_inliers).
Un solo regressore, una sola feature. Coefficienti in
validation/logistic_hard/model_<model>_<matcher>.json, soglia di probabilita' in
validation/logistic_hard/threshold_<model>_<matcher>.csv (entrambi per coppia
(retrieval, image matching)).

probability > tau -> rerank su top-20. Altrimenti skip: salva il .torch del
solo top-1 (gia' calcolato).

Uso:
    python VPR-Adaptive-ReRanking/methods/logistic_hard.py \
        --preds-dir preds/ --model cosplace --matcher loftr --output-dir out/
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import load_threshold_csv, load_model_json, run_logistic_single

_VALIDATION_DIR = Path(__file__).resolve().parent.parent / "validation" / "logistic_hard"


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive reranking — logistic P(hard)")
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
    model_json    = _VALIDATION_DIR / f"model_{args.model}_{args.matcher}.json"

    tau = load_threshold_csv(threshold_csv)["tau"]
    regressor = load_model_json(model_json)   # apply_sigmoid gestisce piatto o annidato

    print(f"tau (P_hard) = {tau}  [{args.model}/{args.matcher}]")
    run_logistic_single(args.preds_dir, tau, regressor, args.matcher, args.device,
                        args.im_size, args.num_preds, args.output_dir)


if __name__ == "__main__":
    main(parse_args())
