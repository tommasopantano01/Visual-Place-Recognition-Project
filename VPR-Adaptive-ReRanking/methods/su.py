"""
su/su.py — Adaptive reranking basato sul solo segnale SU (Score Uncertainty),
calcolato dalle distanze L2 del retrieval. A differenza di tutti gli altri
metodi, NON richiede alcun image matching prima della decisione.

Input diverso dagli altri metodi: serve --z-data (z_data.torch dal retrieval
con --save_for_uncertainty), non solo --preds-dir.

probability > tau -> rerank su top-20 (torch_folder). Altrimenti skip:
copia il .txt originale del retrieval (txt_folder) — coerente con gli
altri metodi, qui ancora piu' naturale dato che zero IM e' mai stato fatto.

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
    run_im_top20_subset, print_summary, get_query_ids, output_subdirs,
)

_THRESHOLD_CSV = Path(__file__).parent / "threshold.csv"
_MODEL_JSON    = Path(__file__).parent / "model.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive reranking — SU only (Rocco)")
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

    txt_folder, torch_folder = output_subdirs(args.output_dir)

    query_ids   = get_query_ids(args.preds_dir)
    l2_by_query = load_z_data_distances(args.z_data)
    signals = {
        q: {"SU": l2_to_su(l2_by_query[q], k=args.su_k, alpha=args.su_alpha)}
        for q in query_ids if q in l2_by_query
    }
    probs = apply_sigmoid(signals, model)

    rerank_ids, skip_ids = partition_by_probability(probs, tau)
    print_summary(rerank_ids, skip_ids)

    save_skipped_as_txt(skip_ids, args.preds_dir, txt_folder)
    run_im_top20_subset(args.preds_dir, rerank_ids, torch_folder,
                         args.matcher, args.device, args.im_size, args.num_preds)


if __name__ == "__main__":
    main(parse_args())
