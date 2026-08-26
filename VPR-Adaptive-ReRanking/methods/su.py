import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))          # VPR-Adaptive-ReRanking/  (per _common)
from _common import load_z_data_distances, l2_to_su, get_query_ids, print_summary, load_threshold_csv

_ARR_DIR   = _HERE.parent
_REPO_ROOT = _ARR_DIR.parent
_MATCH_SCRIPT = _REPO_ROOT / "match_queries_preds.py"

VALID_CRITERIA = ("P(hard)", "P(help)", "P(help)-aP(hurts)")
# criterion -> column prefix in validation/<features>/threshold_<model>_<matcher>.csv
# (flat CSV written by validation/su.py: hard_tau, help_tau, cs_alpha, cs_tau, ...)
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


def write_filtered_preds(query_ids, preds_dir, tmp_dir):
    """Copia i .txt delle query selezionate in tmp_dir (svuotata prima)."""
    tmp = Path(tmp_dir)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    copied = 0
    for q in query_ids:
        src = Path(preds_dir) / f"{q}.txt"
        if src.exists():
            shutil.copy2(src, tmp / f"{q}.txt")
            copied += 1
    return copied


def run_matcher_on_dir(preds_subset_dir, out_dir, matcher, device, im_size, num_preds):
    """Matching totale sulle sole query incerte."""
    if not _MATCH_SCRIPT.exists():
        raise FileNotFoundError(f"match_queries_preds.py non trovato: {_MATCH_SCRIPT}")
    cmd = [
        sys.executable, str(_MATCH_SCRIPT),
        "--preds-dir", str(preds_subset_dir),
        "--out-dir",   str(out_dir),
        "--matcher",   matcher,
        "--device",    device,
        "--im-size",   str(im_size),
        "--num-preds", str(num_preds),
    ]
    print(f"[su] image matching sulle incerte -> {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"match_queries_preds.py terminato con codice {res.returncode}")


def run_matcher_top1_all(preds_dir, top1_dir, matcher, device, im_size):
    """Matching top-1 su TUTTE le query: produce in top1_dir un <id>.torch (feature)."""
    if not _MATCH_SCRIPT.exists():
        raise FileNotFoundError(f"match_queries_preds.py non trovato: {_MATCH_SCRIPT}")
    Path(top1_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(_MATCH_SCRIPT),
        "--preds-dir", str(preds_dir),
        "--out-dir",   str(top1_dir),
        "--matcher",   matcher,
        "--device",    device,
        "--im-size",   str(im_size),
        "--num-preds", "1",
    ]
    print(f"[su_inliers] image matching top-1 su tutte le query -> {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"match_queries_preds.py (top1) codice {res.returncode}")


def load_inliers_top1(top1_dir, query_ids):
    """Legge top1_dir/<id>.torch e ritorna {id: num_inliers} (primo record)."""
    out = {}
    for q in query_ids:
        fp = Path(top1_dir) / f"{q}.torch"
        if not fp.exists():
            continue
        recs = torch.load(fp, map_location="cpu", weights_only=False)
        if recs and isinstance(recs[0], dict) and "num_inliers" in recs[0]:
            out[q] = float(recs[0]["num_inliers"])
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive reranking — SU | SU+inliers (un comando)")
    p.add_argument("--features",   required=True, choices=sorted(FEATURES.keys()))
    p.add_argument("--preds-dir",  required=True)
    p.add_argument("--z-data",     required=True, help="path a z_data.torch del retrieval")
    p.add_argument("--model",      required=True, help="cosplace or megaloc")
    p.add_argument("--matcher",    required=True)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--im-size",    type=int, default=512)
    p.add_argument("--num-preds",  type=int, default=20)
    p.add_argument("--su-k",       type=int, default=10)
    p.add_argument("--su-alpha",   type=float, default=0.5)
    p.add_argument("--criterion",  default="P(help)", choices=VALID_CRITERIA)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-json", default=None,
                   help="default: validation/<features>/<template>_<model>_<matcher>.json")
    p.add_argument("--threshold-csv", default=None,
                   help="default: validation/<features>/threshold_<model>_<matcher>.csv")
    return p.parse_args()


def main(args):
    subdir, feature_set, json_tmpl = FEATURES[args.features]
    val_dir = _ARR_DIR / "validation" / subdir
    model_json     = args.model_json     or str(val_dir / json_tmpl.format(model=args.model, matcher=args.matcher))
    threshold_csv  = args.threshold_csv  or str(val_dir / f"threshold_{args.model}_{args.matcher}.csv")

    with open(model_json) as f:
        model_data = json.load(f)

    regressors = model_data["feature_sets"][feature_set]["regressors"]
    pfx = CRITERION_PREFIX[args.criterion] + "_"
    hp = {k[len(pfx):]: v for k, v in load_threshold_csv(threshold_csv).items() if k.startswith(pfx)}
    if "tau" not in hp:
        raise ValueError(f"{threshold_csv}: no columns '{pfx}*' for criterion {args.criterion} "
                         "(re-run validation/su.py for this model/matcher)")
    print(f"criterio = {args.criterion}   params = {hp}   [{args.model}/{args.matcher}]")

    query_ids = get_query_ids(args.preds_dir)
    l2_by_query = load_z_data_distances(args.z_data)

    # feature aggiuntiva: inliers top-1 (solo per su_inliers)
    inliers = {}
    if args.features == "su_inliers":
        top1_dir = Path(args.output_dir) / "_match_top1"
        run_matcher_top1_all(args.preds_dir, top1_dir, args.matcher, args.device, args.im_size)
        inliers = load_inliers_top1(top1_dir, query_ids)

    ids, feats = [], []
    for q in query_ids:
        if q not in l2_by_query:
            continue
        su = l2_to_su(l2_by_query[q], k=args.su_k, alpha=args.su_alpha)
        if args.features == "su":
            ids.append(q)
            feats.append([su])
        else:
            if q not in inliers:
                continue
            ids.append(q)
            feats.append([su, -inliers[q]])   # convenzione: inliers negato
    X = np.asarray(feats, dtype=float)

    score, tau = compute_scores(X, regressors, args.criterion, hp)
    rerank_ids = [q for q, s in zip(ids, score) if s > tau]
    skip_ids   = [q for q, s in zip(ids, score) if s <= tau]
    print_summary(rerank_ids, skip_ids)

    tmp_dir = Path(args.output_dir) / f"_tmp_rerank_{args.features}"
    n = write_filtered_preds(rerank_ids, args.preds_dir, tmp_dir)
    print(f"[{args.features}] {n} query incerte copiate in {tmp_dir}")
    if n > 0:
        run_matcher_on_dir(tmp_dir, args.output_dir, args.matcher,
                           args.device, args.im_size, args.num_preds)
    else:
        print(f"[{args.features}] nessuna query incerta: nessun matching totale.")

    skip_log = Path(args.output_dir) / "skipped_query_ids.txt"
    skip_log.write_text("\n".join(skip_ids))
    print(f"[{args.features}] {len(skip_ids)} query skip -> {skip_log}")


if __name__ == "__main__":
    main(parse_args())
