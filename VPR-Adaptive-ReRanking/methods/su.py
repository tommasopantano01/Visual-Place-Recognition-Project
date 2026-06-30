"""
su/su.py — Adaptive reranking sul solo segnale SU (Score Uncertainty) dalle
distanze L2 del retrieval: NESSUN image matching prima della decisione.
[METODO ACCANTONATO dal team — tenuto solo per riferimento.]

Input: serve --z-data (z_data.torch da --save_for_uncertainty), non solo
--preds-dir.

probability > tau -> rerank su top-20 (top20/). Altrimenti skip -> top1/ come
.txt: su non fa IM sulle skip, quindi non c'e' nessun inlier da salvare.

Uso:
    python VPR-adaptive-re-ranking/su/su.py \
        --preds-dir preds/ --z-data z_data.torch --matcher superpoint-lg \
        --output-dir out/
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import (
    load_threshold_csv, load_model_json, load_z_data_distances, l2_to_su,
    apply_sigmoid, partition_by_probability, save_skipped_as_txt,
    run_im_topN_subset, print_summary, get_query_ids, budget_folder,
)

_THRESHOLD_CSV = Path(__file__).parent / "threshold.csv"
_MODEL_JSON    = Path(__file__).parent / "model.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive reranking — SU only (accantonato)")
    parser.add_argument("--preds-dir",  required=True)
    parser.add_argument("--z-data",     required=True, help="path a z_data.torch del retrieval")
    parser.add_argument("--matcher",    required=True, help="usato solo per il ramo rerank")
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
    print(f"tau (SU) = {tau}")

    query_ids   = get_query_ids(args.preds_dir)
    l2_by_query = load_z_data_distances(args.z_data)
    signals = {
        q: {"SU": l2_to_su(l2_by_query[q], k=args.su_k, alpha=args.su_alpha)}
        for q in query_ids if q in l2_by_query
    }
    probs = apply_sigmoid(signals, model)

    rerank_ids, skip_ids = partition_by_probability(probs, tau)
    print_summary(rerank_ids, skip_ids)

    # skip -> top1/ come .txt (nessun IM fatto sulle skip)
    save_skipped_as_txt(skip_ids, args.preds_dir, budget_folder(args.output_dir, 1))
    run_im_topN_subset(args.preds_dir, rerank_ids, args.output_dir, args.num_preds,
                        args.matcher, args.device, args.im_size)


if __name__ == "__main__":
    main(parse_args())
