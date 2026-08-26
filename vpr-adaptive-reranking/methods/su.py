"""
methods/su.py — Deploy della famiglia SU (su | su_inliers).

SU (Score Uncertainty) e' calcolato dalle sole distanze L2 del retrieval
(z_data.torch), quindi:
  su          la decisione NON costa nessun image matching. Le query skippate
              finiscono in top0/<id>.txt (budget 0: nessun IM fatto).
  su_inliers  usa anche num_inliers_top1, quindi serve l'IM sul top-1 di tutte
              le query. Le skippate finiscono in top1/<id>.torch.
In entrambi i casi le query incerte vanno in top{num_preds}/<id>.torch.

    python VPR-Adaptive-ReRanking/methods/su.py --features su \
        --preds-dir <preds/> --z-data <z_data.torch> \
        --model cosplace --matcher superpoint-lg \
        --inliers-dir <top20 .torch/> --output-dir <out/>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))          # VPR-Adaptive-ReRanking/  (per _common)
from _common import (
    load_z_data_distances, l2_to_su, get_query_ids, print_summary,
    load_threshold_csv, run_im_top1_with_results, run_im_topN_subset,
    save_results_torch, save_skipped_as_txt, budget_folder,
)

_ARR_DIR = _HERE.parent

VALID_CRITERIA = ("P(hard)", "P(help)", "P(help)-aP(hurts)")
# criterion -> prefisso di colonna in validation/<features>/threshold_<model>_<matcher>.csv
# (CSV piatto scritto da validation/su.py: hard_tau, help_tau, cs_alpha, cs_tau, ...)
CRITERION_PREFIX = {"P(hard)": "hard", "P(help)": "help", "P(help)-aP(hurts)": "cs"}

# features -> (sottocartella validation, chiave feature_set nel json, template model json)
FEATURES = {
    "su":         ("su",         "SU",         "model_su_{model}_{matcher}.json"),
    "su_inliers": ("su_inliers", "SU+inliers", "model_su_num_inliers_{model}_{matcher}.json"),
}


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
    p = argparse.ArgumentParser(description="Adaptive reranking — SU | SU+inliers")
    p.add_argument("--features",   required=True, choices=sorted(FEATURES.keys()))
    p.add_argument("--preds-dir",  required=True)
    p.add_argument("--z-data",     required=True, help="path a z_data.torch del retrieval")
    p.add_argument("--model",      required=True, help="cosplace or megaloc")
    p.add_argument("--matcher",    required=True, help="superpoint-lg or loftr")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--im-size",    type=int, default=512)
    p.add_argument("--num-preds",  type=int, default=20)
    p.add_argument("--su-k",       type=int, default=10)
    p.add_argument("--su-alpha",   type=float, default=0.5)
    p.add_argument("--criterion",  default="P(help)", choices=VALID_CRITERIA)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--inliers-dir", default=None,
                   help="OFFLINE: cartella con i .torch top-20 gia' calcolati "
                        "(niente image matching, solo lettura)")
    p.add_argument("--model-json", default=None,
                   help="default: validation/<features>/<template>")
    p.add_argument("--threshold-csv", default=None,
                   help="default: validation/<features>/threshold_<model>_<matcher>.csv")
    return p.parse_args()


def main(args):
    subdir, feature_set, json_tmpl = FEATURES[args.features]
    val_dir = _ARR_DIR / "validation" / subdir
    model_json = Path(args.model_json or
                      val_dir / json_tmpl.format(model=args.model, matcher=args.matcher))
    threshold_csv = Path(args.threshold_csv or
                         val_dir / f"threshold_{args.model}_{args.matcher}.csv")
    if not model_json.exists():
        raise FileNotFoundError(
            f"Model JSON non trovato: {model_json}\n"
            f"  -> esegui train_su.py --features {args.features} oppure "
            "validation/download_models.py")
    if not threshold_csv.exists():
        raise FileNotFoundError(
            f"Soglia non trovata: {threshold_csv}\n"
            f"  -> esegui prima: validation/su.py --features {args.features} "
            f"--model {args.model} --matcher {args.matcher} --val-csv <candidate_level_val.csv>")

    with open(model_json) as f:
        model_data = json.load(f)
    regressors = model_data["feature_sets"][feature_set]["regressors"]

    pfx = CRITERION_PREFIX[args.criterion] + "_"
    hp = {k[len(pfx):]: v for k, v in load_threshold_csv(threshold_csv).items()
          if k.startswith(pfx)}
    if "tau" not in hp:
        raise ValueError(f"{threshold_csv}: nessuna colonna '{pfx}*' per il criterio "
                         f"{args.criterion} (rilancia validation/su.py per questa coppia)")
    print(f"criterio = {args.criterion}   params = {hp}   [{args.model}/{args.matcher}]")

    query_ids = get_query_ids(args.preds_dir)
    l2_by_query = load_z_data_distances(args.z_data)

    # feature aggiuntiva: inliers top-1 (solo per su_inliers)
    results_top1 = {}
    if args.features == "su_inliers":
        results_top1 = run_im_top1_with_results(
            args.preds_dir, args.matcher, args.device, args.im_size,
            query_ids=query_ids, inliers_dir=args.inliers_dir)

    ids, feats = [], []
    for q in query_ids:
        if q not in l2_by_query:
            continue
        su = l2_to_su(l2_by_query[q], k=args.su_k, alpha=args.su_alpha)
        if args.features == "su":
            ids.append(q)
            feats.append([su])
        else:
            if q not in results_top1:
                continue
            ids.append(q)
            # convenzione di training: inliers negato (pochi inlier -> feature alta)
            feats.append([su, -float(results_top1[q]["num_inliers"])])

    if not ids:
        raise RuntimeError("Nessuna query utilizzabile: controlla --z-data e --preds-dir.")
    X = np.asarray(feats, dtype=float)

    score, tau = compute_scores(X, regressors, args.criterion, hp)
    rerank_ids = [q for q, s in zip(ids, score) if s > tau]
    skip_ids   = [q for q, s in zip(ids, score) if s <= tau]
    print_summary(rerank_ids, skip_ids)

    # SKIP: budget 0 per su puro (nessun IM), budget 1 per su_inliers
    if args.features == "su":
        save_skipped_as_txt(skip_ids, args.preds_dir, budget_folder(args.output_dir, 0))
    else:
        folder1 = budget_folder(args.output_dir, 1)
        for q in skip_ids:
            save_results_torch(q, [results_top1[q]], folder1)

    # RERANK: top-N completo
    run_im_topN_subset(args.preds_dir, rerank_ids, args.output_dir, args.num_preds,
                       args.matcher, args.device, args.im_size,
                       inliers_dir=args.inliers_dir)


if __name__ == "__main__":
    main(parse_args())
