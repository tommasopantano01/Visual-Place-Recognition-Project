"""
logistic_help/logistic_help.py — Regressione logistica P(help | num_inliers).
Un solo regressore, una sola feature. Coefficienti in
validation/logistic_help/model.json, soglia di probabilita' in
validation/logistic_help/threshold_<model>_<matcher>.csv (entrambi gia'
calibrati, vedi extension/helps_estimator.py --method logistic --criterion help).

probability > tau -> rerank su top-20 (torch_folder). Altrimenti skip:
copia il .txt originale (txt_folder).

Uso:
    python VPR-adaptive-re-ranking/logistic_help/logistic_help.py \
        --preds-dir preds/ --model cosplace --matcher superpoint-lg --output-dir out/
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import load_threshold_csv, load_model_json, run_logistic_single

_VALIDATION_DIR = Path(__file__).resolve().parent.parent / "validation" / "logistic_help"


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive reranking — logistic P(help)")
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
    tau = load_threshold_csv(threshold_csv)["tau"]
    regressor = load_model_json(_VALIDATION_DIR / f"model_{args.model}_{args.matcher}.json")
    print(f"tau (P_help) = {tau}  [{args.model}/{args.matcher}]")
    run_logistic_single(args.preds_dir, tau, regressor, args.matcher, args.device,
                         args.im_size, args.num_preds, args.output_dir)


if __name__ == "__main__":
    main(parse_args())
