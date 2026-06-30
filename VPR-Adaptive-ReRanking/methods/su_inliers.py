"""
su_inliers/su_inliers.py — Adaptive reranking su SU + num_inliers nello stesso
regressore. [METODO ACCANTONATO dal team — tenuto solo per riferimento.]
A differenza di su/, qui SERVE IM minimale sul top-1 (inliers e' una feature).

Convenzione Rocco: inliers = -num_inliers (pochi inlier -> feature alta ->
query incerta). Non invertire.

Input: --preds-dir PIU' --z-data.

probability > tau -> rerank su top-20 (top20/). Altrimenti skip -> top1/
(.torch del solo top-1, gia' calcolato).

Uso:
    python VPR-adaptive-re-ranking/su_inliers/su_inliers.py \
        --preds-dir preds/ --z-data z_data.torch --matcher superpoint-lg \
        --output-dir out/
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import (
    load_threshold_csv, load_model_json, load_z_data_distances, l2_to_su,
    run_im_top1_with_results, apply_sigmoid, partition_by_probability,
    save_results_torch, run_im_topN_subset, print_summary, budget_folder,
)

_THRESHOLD_CSV = Path(__file__).parent / "threshold.csv"
_MODEL_JSON    = Path(__file__).parent / "model.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive reranking — SU + inliers (accantonato)")
    parser.add_argument("--preds-dir",  required=True)
    parser.add_argument("--z-data",     required=True, help="path a z_data.torch del retrieval")
    parser.add_argument("--matcher",    required=True)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--im-size",    type=int, default=512)
    parser.add_argument("--num-preds",  type=int, default=20)
    parser.add_argument("--su-k",       type=int, default=10)
    parser.add_argument("--su-alpha",   type=float, default=0.5)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main(args):
    tau   = load_threshold_csv(_THRESHOLD_CSV)["tau"]
    model = load_model_json(_MODEL_JSON)
    print(f"tau (SU+inliers) = {tau}")

    results_top1 = run_im_top1_with_results(args.preds_dir, args.matcher, args.device, args.im_size)
    l2_by_query  = load_z_data_distances(args.z_data)

    signals = {
        q: {
            "SU":       l2_to_su(l2_by_query[q], k=args.su_k, alpha=args.su_alpha),
            "inliers": -float(r["num_inliers"]),   # convenzione Rocco: inliers negato
        }
        for q, r in results_top1.items() if q in l2_by_query
    }
    probs = apply_sigmoid(signals, model)

    rerank_ids, skip_ids = partition_by_probability(probs, tau)
    print_summary(rerank_ids, skip_ids)

    folder1 = budget_folder(args.output_dir, 1)
    for q in skip_ids:
        save_results_torch(q, [results_top1[q]], folder1)
    run_im_topN_subset(args.preds_dir, rerank_ids, args.output_dir, args.num_preds,
                        args.matcher, args.device, args.im_size)


if __name__ == "__main__":
    main(parse_args())
