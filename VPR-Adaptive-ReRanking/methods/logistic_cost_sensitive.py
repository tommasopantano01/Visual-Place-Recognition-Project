"""
logistic_cost_sensitive/logistic_cost_sensitive.py — Utility-based logistic
regressor policy: S = P(help|num_inliers) - lambda*P(hurt|num_inliers).
Due regressori distinti (entrambi nello stesso model.json), lambda e tau
gia' calibrati su validation (vedi extension/helps_estimator.py
--method logistic --criterion cost_sensitive).

S > tau -> rerank su top-20 (torch_folder). Altrimenti skip: copia il
.txt originale (txt_folder).

Uso:
    python VPR-adaptive-re-ranking/logistic_cost_sensitive/logistic_cost_sensitive.py \
        --preds-dir preds/ --matcher superpoint-lg --output-dir out/
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import (
    load_threshold_csv, load_model_json, run_im_top1_all, apply_sigmoid,
    partition_by_probability, save_skipped_as_txt, run_im_top20_subset,
    print_summary, output_subdirs,
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

    txt_folder, torch_folder = output_subdirs(args.output_dir)

    inliers = run_im_top1_all(args.preds_dir, args.matcher, args.device, args.im_size)
    signals = {q: {"inliers": n} for q, n in inliers.items()}

    p_help = apply_sigmoid(signals, models["help"])
    p_hurt = apply_sigmoid(signals, models["hurt"])
    scores = {q: p_help[q] - lam * p_hurt[q] for q in inliers}

    rerank_ids, skip_ids = partition_by_probability(scores, tau)
    print_summary(rerank_ids, skip_ids)

    save_skipped_as_txt(skip_ids, args.preds_dir, txt_folder)
    run_im_top20_subset(args.preds_dir, rerank_ids, torch_folder,
                         args.matcher, args.device, args.im_size, args.num_preds)


if __name__ == "__main__":
    main(parse_args())
