"""
logistic_cost_sensitive/logistic_cost_sensitive.py — Utility-based logistic
policy: S = P(help|inliers) - lambda*P(hurt|inliers). Due regressori distinti
nello stesso model.json {"help": {...}, "hurt": {...}}, lambda e tau gia'
calibrati su validation.

S > tau -> rerank su top-20 (top20/). Altrimenti skip -> top1/ (.torch del
solo top-1, gia' calcolato).

Uso:
    python VPR-adaptive-re-ranking/logistic_cost_sensitive/logistic_cost_sensitive.py \
        --preds-dir preds/ --matcher superpoint-lg --output-dir out/
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import (
    load_threshold_csv, load_model_json, run_im_top1_with_results, apply_sigmoid,
    partition_by_probability, save_results_torch, run_im_topN_subset,
    print_summary, budget_folder,
)

_THRESHOLD_CSV = Path(__file__).parent / "threshold.csv"
_MODEL_JSON    = Path(__file__).parent / "model.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive reranking — cost-sensitive (P_help - lambda*P_hurt)")
    parser.add_argument("--preds-dir",  required=True)
    parser.add_argument("--matcher",    required=True)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--im-size",    type=int, default=512)
    parser.add_argument("--num-preds",  type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main(args):
    hp     = load_threshold_csv(_THRESHOLD_CSV)
    lam    = hp["lambda"]
    tau    = hp["tau"]
    models = load_model_json(_MODEL_JSON)   # {"help": {...}, "hurt": {...}}
    print(f"lambda = {lam}   tau = {tau}")

    results_top1 = run_im_top1_with_results(args.preds_dir, args.matcher, args.device, args.im_size)
    signals = {q: {"inliers": r["num_inliers"]} for q, r in results_top1.items()}

    p_help = apply_sigmoid(signals, models["help"])
    p_hurt = apply_sigmoid(signals, models["hurt"])
    scores = {q: p_help[q] - lam * p_hurt[q] for q in results_top1}

    rerank_ids, skip_ids = partition_by_probability(scores, tau)
    print_summary(rerank_ids, skip_ids)

    folder1 = budget_folder(args.output_dir, 1)
    for q in skip_ids:
        save_results_torch(q, [results_top1[q]], folder1)
    run_im_topN_subset(args.preds_dir, rerank_ids, args.output_dir, args.num_preds,
                        args.matcher, args.device, args.im_size)


if __name__ == "__main__":
    main(parse_args())
