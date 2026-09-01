"""
methods/su.py — test of the SU family (su | su_inliers).

SU is computed from the retrieval L2 distances alone
(z_data.torch), therefore:
  su          the decision costs NO image matching. Skipped queries end up
              in top0/<id>.txt (budget 0: no IM done).
  su_inliers  also uses num_inliers_top1, so the IM on the top-1 of every
              query is needed. Skipped ones end up in top1/<id>.torch.
In both cases the uncertain queries go to top{num_preds}/<id>.torch.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))
sys.path.append(str(_HERE.parent / "validation"))
from _common import (
    load_z_data_distances, l2_to_su, get_query_ids, print_summary,
    load_threshold_csv, run_im_top1_with_results, run_im_topN_subset,
    save_results_torch, save_skipped_as_txt, budget_folder,
)
from _outputs import canon_model, canon_matcher

_ARR_DIR = _HERE.parent

VALID_CRITERIA = ("P(hard)", "P(help)", "P(help)-aP(hurts)")
# criterion -> column prefix in validation/<features>/threshold_<model>_<matcher>.csv
# (flat CSV written by validation/su.py: hard_tau, help_tau, cs_alpha, cs_tau, ...)
CRITERION_PREFIX = {"P(hard)": "hard", "P(help)": "help", "P(help)-aP(hurts)": "cs"}

# features -> (validation subfolder, feature_set key in the json, model json template)
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
    """Returns (score_array, tau) for the chosen criterion."""
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
    p.add_argument("--z-data",     required=True, help="path to the retrieval z_data.torch")
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
                   help="OFFLINE: folder with the already computed top-20 .torch files "
                        "(no image matching, read only)")
    p.add_argument("--model-json", default=None,
                   help="default: validation/<features>/<template>")
    p.add_argument("--threshold-csv", default=None,
                   help="default: validation/<features>/threshold_<model>_<matcher>.csv")
    return p.parse_args()


def main(args):
    model, matcher = canon_model(args.model), canon_matcher(args.matcher)
    subdir, feature_set, json_tmpl = FEATURES[args.features]
    val_dir = _ARR_DIR / "validation" / subdir
    model_json = Path(args.model_json or
                      val_dir / json_tmpl.format(model=model, matcher=matcher))
    threshold_csv = Path(args.threshold_csv or
                         val_dir / f"threshold_{model}_{matcher}.csv")
    if not model_json.exists():
        raise FileNotFoundError(
            f"Model JSON not found: {model_json}\n"
            f"  -> run train_su.py --features {args.features} or download the "
            "trained regressors")
    if not threshold_csv.exists():
        raise FileNotFoundError(
            f"Threshold not found: {threshold_csv}\n"
            f"  -> run first: validation/su.py --features {args.features} "
            f"--model {model} --matcher {matcher} --val-csv <candidate_level_val.csv>")

    with open(model_json) as f:
        model_data = json.load(f)
    regressors = model_data["feature_sets"][feature_set]["regressors"]

    pfx = CRITERION_PREFIX[args.criterion] + "_"
    hp = {k[len(pfx):]: v for k, v in load_threshold_csv(threshold_csv).items()
          if k.startswith(pfx)}
    if "tau" not in hp:
        raise ValueError(f"{threshold_csv}: no '{pfx}*' column for criterion "
                         f"{args.criterion} (re-run validation/su.py for this pair)")
    print(f"criterion = {args.criterion}   params = {hp}   [{model}/{matcher}]")

    query_ids = get_query_ids(args.preds_dir)
    l2_by_query = load_z_data_distances(args.z_data)

    # additional feature: top-1 inliers (only for su_inliers)
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
            # training convention: negated inliers (few inliers -> high feature)
            feats.append([su, -float(results_top1[q]["num_inliers"])])

    if not ids:
        raise RuntimeError("No usable query: check --z-data and --preds-dir.")
    X = np.asarray(feats, dtype=float)

    score, tau = compute_scores(X, regressors, args.criterion, hp)
    rerank_ids = [q for q, s in zip(ids, score) if s > tau]
    skip_ids   = [q for q, s in zip(ids, score) if s <= tau]
    print_summary(rerank_ids, skip_ids)

    # SKIP: budget 0 for pure su (no IM), budget 1 for su_inliers
    if args.features == "su":
        save_skipped_as_txt(skip_ids, args.preds_dir, budget_folder(args.output_dir, 0))
    else:
        folder1 = budget_folder(args.output_dir, 1)
        for q in skip_ids:
            save_results_torch(q, [results_top1[q]], folder1)

    # RERANK: full top-N
    run_im_topN_subset(args.preds_dir, rerank_ids, args.output_dir, args.num_preds,
                       args.matcher, args.device, args.im_size,
                       inliers_dir=args.inliers_dir)


if __name__ == "__main__":
    main(parse_args())
