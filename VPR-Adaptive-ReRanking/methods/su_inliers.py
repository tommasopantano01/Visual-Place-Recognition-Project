"""
methods/su_inliers.py — Adaptive reranking SU+inliers. RIATTIVATO.

Come methods/su.py ma il feature set e' ['SU','inliers']: oltre a SU (dalle L2)
usa num_inliers della top-1, che richiede image matching MINIMALE sul solo
top-1 di ogni query (molto piu' economico del rerank completo).

    score > tau  ->  rerank: IM su TUTTI i candidati top-N (top{N}/)
    score <= tau ->  skip:   top-1 va bene -> top1/ (.torch del solo top-1,
                             gia' calcolato per la feature, viene riusato)

Convenzione: inliers = -num_inliers (pochi inlier -> feature alta -> incerta).
Modelli/soglie da validation/su_inliers/. Criteri come in methods/su.py.

Input: --preds-dir e --z-data.

Uso:
    python VPR-Adaptive-ReRanking/methods/su_inliers.py \
        --preds-dir preds/ --z-data z_data.torch --matcher superpoint-lg \
        --output-dir out/ --criterion "P(help)-aP(hurts)"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))              # per 'import su' (stessa cartella)
sys.path.append(str(_HERE.parent))       # per 'import _common' (cartella radice del metodo)
from _common import (
    load_z_data_distances, l2_to_su, run_im_top1_with_results,
    save_results_torch, run_im_topN_subset, budget_folder, print_summary,
)
# riuso le stesse funzioni di scoring del decisore SU
from su import compute_scores, VALID_CRITERIA  # noqa: F401

_VAL_DIR    = Path(__file__).resolve().parent.parent / "validation" / "su_inliers"
_MODEL_JSON = _VAL_DIR / "model.json"
_THR_JSON   = _VAL_DIR / "threshold.csv"

FEATURE_SET = "SU+inliers"
FEAT_COLS   = ["SU", "inliers"]


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive reranking — SU+inliers (riattivato)")
    p.add_argument("--preds-dir",  required=True)
    p.add_argument("--z-data",     required=True, help="path a z_data.torch del retrieval")
    p.add_argument("--matcher",    required=True)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--im-size",    type=int, default=512)
    p.add_argument("--num-preds",  type=int, default=20)
    p.add_argument("--su-k",       type=int, default=10)
    p.add_argument("--su-alpha",   type=float, default=0.5)
    p.add_argument("--criterion",  default="P(help)-aP(hurts)", choices=VALID_CRITERIA)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main(args):
    with open(_MODEL_JSON) as f:
        model = json.load(f)
    with open(_THR_JSON) as f:
        thr = json.load(f)

    regressors = model["feature_sets"][FEATURE_SET]["regressors"]
    hp = thr["feature_sets"][FEATURE_SET]["criteria"][args.criterion]
    print(f"criterio = {args.criterion}   params = {hp}")

    # IM minimale sul top-1 di tutte le query: serve per la feature 'inliers'
    results_top1 = run_im_top1_with_results(args.preds_dir, args.matcher,
                                            args.device, args.im_size)
    l2_by_query = load_z_data_distances(args.z_data)

    ids, feats = [], []
    for q, r in results_top1.items():
        if q not in l2_by_query:
            continue
        su = l2_to_su(l2_by_query[q], k=args.su_k, alpha=args.su_alpha)
        inl = -float(r["num_inliers"])           # convenzione: negato
        ids.append(q)
        feats.append([su, inl])
    X = np.asarray(feats, dtype=float).reshape(-1, 2)

    score, tau = compute_scores(X, regressors, args.criterion, hp)
    rerank_ids = [q for q, s in zip(ids, score) if s > tau]
    skip_ids   = [q for q, s in zip(ids, score) if s <= tau]
    print_summary(rerank_ids, skip_ids)

    # skip -> top1/: riusa il risultato IM del top-1 gia' calcolato
    folder1 = budget_folder(args.output_dir, 1)
    for q in skip_ids:
        save_results_torch(q, [results_top1[q]], folder1)
    # rerank -> top{num_preds}/: IM completo
    run_im_topN_subset(args.preds_dir, rerank_ids, args.output_dir, args.num_preds,
                       args.matcher, args.device, args.im_size)


if __name__ == "__main__":
    main(parse_args())
