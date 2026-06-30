"""
methods/su.py — Adaptive reranking SU-only. RIATTIVATO (non piu' accantonato).

Decisione basata SOLO sulle distanze L2 del retrieval (nessun image matching
prima della decisione). Per ogni query calcola SU, applica il regressore del
criterio scelto e decide:

    score > tau  ->  rerank: image matching su TUTTI i candidati top-N (top{N}/)
    score <= tau ->  skip:   la top-1 va gia' bene, niente IM (top1/ come .txt)

Legge i modelli/soglie calibrati da validation/su/ (model.json + threshold.csv,
formato annidato feature_sets->...). Tre criteri selezionabili via --criterion:
    P(hard)            usa il regressore 'hard'
    P(help)            usa il regressore 'help'
    P(help)-aP(hurts)  usa 'help' e 'hurts' con alpha calibrato (default)

Input: --preds-dir e --z-data (z_data.torch da --save_for_uncertainty).

Uso:
    python VPR-Adaptive-ReRanking/methods/su.py \
        --preds-dir preds/ --z-data z_data.torch --matcher superpoint-lg \
        --output-dir out/ --criterion "P(help)-aP(hurts)"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import (
    load_z_data_distances, l2_to_su, get_query_ids,
    save_skipped_as_txt, run_im_topN_subset, budget_folder, print_summary,
)

_VAL_DIR    = Path(__file__).resolve().parent.parent / "validation" / "su"
_MODEL_JSON = _VAL_DIR / "model.json"
_THR_JSON   = _VAL_DIR / "threshold.csv"

FEATURE_SET = "SU"
FEAT_COLS   = ["SU"]
VALID_CRITERIA = ("P(hard)", "P(help)", "P(help)-aP(hurts)")


def _proba(reg, X):
    mean  = np.asarray(reg["scaler_mean"], dtype=float)
    scale = np.asarray(reg["scaler_scale"], dtype=float)
    w     = np.asarray(reg["coef"][0], dtype=float)
    b     = float(reg["intercept"][0])
    z = (X - mean) / scale
    return 1.0 / (1.0 + np.exp(-(z @ w + b)))


def compute_scores(X, regressors, criterion, hp):
    """Ritorna (score_array, tau) per il criterio scelto."""
    if criterion == "P(hard)":
        return _proba(regressors["hard"], X), float(hp["tau"])
    if criterion == "P(help)":
        return _proba(regressors["help"], X), float(hp["tau"])
    p_help  = _proba(regressors["help"], X)
    p_hurts = _proba(regressors["hurts"], X)
    return p_help - float(hp["alpha"]) * p_hurts, float(hp["tau"])


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive reranking — SU (riattivato)")
    p.add_argument("--preds-dir",  required=True)
    p.add_argument("--z-data",     required=True, help="path a z_data.torch del retrieval")
    p.add_argument("--matcher",    required=True, help="usato solo nel ramo rerank")
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

    query_ids = get_query_ids(args.preds_dir)
    l2_by_query = load_z_data_distances(args.z_data)

    ids, su_vals = [], []
    for q in query_ids:
        if q not in l2_by_query:
            continue
        ids.append(q)
        su_vals.append(l2_to_su(l2_by_query[q], k=args.su_k, alpha=args.su_alpha))
    X = np.asarray(su_vals, dtype=float).reshape(-1, 1)

    score, tau = compute_scores(X, regressors, args.criterion, hp)
    rerank_ids = [q for q, s in zip(ids, score) if s > tau]
    skip_ids   = [q for q, s in zip(ids, score) if s <= tau]
    print_summary(rerank_ids, skip_ids)

    # skip -> top1/ come .txt (SU non fa IM sulle skip)
    save_skipped_as_txt(skip_ids, args.preds_dir, budget_folder(args.output_dir, 1))
    # rerank -> top{num_preds}/: IM completo su tutti i candidati
    run_im_topN_subset(args.preds_dir, rerank_ids, args.output_dir, args.num_preds,
                       args.matcher, args.device, args.im_size)


if __name__ == "__main__":
    main(parse_args())
