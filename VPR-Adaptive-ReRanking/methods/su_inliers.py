"""
methods/su_inliers.py — Adaptive reranking SU+inliers.

UN SOLO COMANDO, fa tutto in locale. Come methods/su.py ma il feature set e'
['SU','inliers']: serve num_inliers della top-1 di OGNI query per decidere.
Quel matching top-1 viene fatto internamente tramite lo script del prof
match_queries_preds.py --num-preds 1 (NON modificato).

Flusso interno (invisibile all'utente):
  1. lancia match_queries_preds.py --num-preds 1 su TUTTE le query (feature top-1)
  2. legge SU (da z_data) + num_inliers (dai .torch top-1) -> score -> decisione
  3. copia i .txt delle sole INCERTE in una cartella temp locale
  4. lancia match_queries_preds.py --num-preds N sulle incerte (matching totale)

Le query NON incerte restano col solo top-1 gia' calcolato al passo 1.

Convenzione: inliers = -num_inliers. Modelli/soglie: training/su_inliers/ e
validation/su_inliers/. Criteri: P(hard) | P(help) | P(help)-aP(hurts).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))                 # methods/  (per 'import su')
sys.path.append(str(_HERE.parent))          # VPR-Adaptive-ReRanking/  (per _common)
from _common import load_z_data_distances, l2_to_su, get_query_ids, print_summary
from su import compute_scores, write_filtered_preds, run_matcher_on_dir, VALID_CRITERIA

_ARR_DIR   = _HERE.parent
_REPO_ROOT = _ARR_DIR.parent
_MATCH_SCRIPT = _REPO_ROOT / "match_queries_preds.py"
_MODEL_JSON = _ARR_DIR / "training" / "su_inliers" / "model.json"
_THR_JSON   = _ARR_DIR / "validation" / "su_inliers" / "threshold.csv"

FEATURE_SET = "SU+inliers"


def run_matcher_top1_all(preds_dir, top1_dir, matcher, device, im_size):
    """Lancia match_queries_preds.py --num-preds 1 su TUTTE le query: produce in
    top1_dir un <id>.torch con il risultato IM del solo top-1 (la feature)."""
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
    p = argparse.ArgumentParser(description="Adaptive reranking — SU+inliers (un comando)")
    p.add_argument("--preds-dir",  required=True)
    p.add_argument("--z-data",     required=True)
    p.add_argument("--matcher",    required=True)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--im-size",    type=int, default=512)
    p.add_argument("--num-preds",  type=int, default=20)
    p.add_argument("--su-k",       type=int, default=10)
    p.add_argument("--su-alpha",   type=float, default=0.5)
    p.add_argument("--criterion",  default="P(help)-aP(hurts)", choices=VALID_CRITERIA)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-json", default=str(_MODEL_JSON))
    p.add_argument("--threshold-json", default=str(_THR_JSON))
    return p.parse_args()


def main(args):
    with open(args.model_json) as f:
        model = json.load(f)
    with open(args.threshold_json) as f:
        thr = json.load(f)

    regressors = model["feature_sets"][FEATURE_SET]["regressors"]
    hp = thr["feature_sets"][FEATURE_SET]["criteria"][args.criterion]
    print(f"criterio = {args.criterion}   params = {hp}")

    # 1. matching top-1 su tutte le query (feature) — via script del prof
    top1_dir = Path(args.output_dir) / "_match_top1"
    run_matcher_top1_all(args.preds_dir, top1_dir, args.matcher, args.device, args.im_size)

    # 2. SU + inliers -> score -> decisione
    query_ids = get_query_ids(args.preds_dir)
    l2_by_query = load_z_data_distances(args.z_data)
    inliers = load_inliers_top1(top1_dir, query_ids)

    ids, feats = [], []
    for q in query_ids:
        if q not in l2_by_query or q not in inliers:
            continue
        su = l2_to_su(l2_by_query[q], k=args.su_k, alpha=args.su_alpha)
        ids.append(q)
        feats.append([su, -inliers[q]])      # convenzione: inliers negato
    X = np.asarray(feats, dtype=float).reshape(-1, 2)

    score, tau = compute_scores(X, regressors, args.criterion, hp)
    rerank_ids = [q for q, s in zip(ids, score) if s > tau]
    skip_ids   = [q for q, s in zip(ids, score) if s <= tau]
    print_summary(rerank_ids, skip_ids)

    # 3-4. cartella temp incerte + matching totale via script del prof
    tmp_dir = Path(args.output_dir) / "_tmp_rerank_su_inliers"
    n = write_filtered_preds(rerank_ids, args.preds_dir, tmp_dir)
    print(f"[su_inliers] {n} query incerte copiate in {tmp_dir}")
    if n > 0:
        run_matcher_on_dir(tmp_dir, args.output_dir, args.matcher,
                           args.device, args.im_size, args.num_preds)
    else:
        print("[su_inliers] nessuna query incerta: nessun matching totale.")

    skip_log = Path(args.output_dir) / "skipped_query_ids.txt"
    skip_log.write_text("\n".join(skip_ids))
    print(f"[su_inliers] {len(skip_ids)} query skip (solo top-1) -> {skip_log}")


if __name__ == "__main__":
    main(parse_args())
