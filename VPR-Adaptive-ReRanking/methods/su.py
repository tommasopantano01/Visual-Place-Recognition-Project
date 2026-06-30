"""
methods/su.py — Adaptive reranking SU-only.

UN SOLO COMANDO, fa tutto in locale. Decisione basata SOLO sulle L2 del
retrieval (nessun image matching per decidere), poi image matching TOTALE solo
sulle query incerte, eseguito tramite lo script del prof match_queries_preds.py
(che NON viene modificato).

Flusso interno (invisibile all'utente):
  1. legge SU da z_data.torch, applica regressore + soglia del criterio scelto
  2. partiziona: score > tau -> INCERTE (rerank);  score <= tau -> skip
  3. copia i .txt delle sole INCERTE in una cartella temp locale
     (output-dir/_tmp_rerank_su/)
  4. lancia match_queries_preds.py --preds-dir <temp> --num-preds N
     -> image matching totale solo sulle incerte
  5. l'output dei .torch finisce in output-dir/ (vedi --out-dir passato allo script)

Le query NON incerte: nessun image matching (resta valida la top-1 del retrieval).

Modelli/soglie: training/su/model.json + validation/su/threshold.csv.
Criteri: P(hard) | P(help) | P(help)-aP(hurts).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))          # VPR-Adaptive-ReRanking/  (per _common)
from _common import load_z_data_distances, l2_to_su, get_query_ids, print_summary

# percorsi di default
_ARR_DIR  = _HERE.parent                                   # VPR-Adaptive-ReRanking/
_REPO_ROOT = _ARR_DIR.parent                               # radice repo
_MATCH_SCRIPT = _REPO_ROOT / "match_queries_preds.py"      # script del prof
_MODEL_JSON = _ARR_DIR / "training" / "su" / "model.json"
_THR_JSON   = _ARR_DIR / "validation" / "su" / "threshold.csv"

FEATURE_SET = "SU"
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
    """Lancia match_queries_preds.py (script del prof, intatto) sulla cartella
    di sole query incerte. Subprocess: un solo comando per l'utente."""
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


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive reranking — SU (un comando)")
    p.add_argument("--preds-dir",  required=True)
    p.add_argument("--z-data",     required=True, help="path a z_data.torch del retrieval")
    p.add_argument("--matcher",    required=True)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--im-size",    type=int, default=512)
    p.add_argument("--num-preds",  type=int, default=20, help="candidati per il matching totale")
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

    # 1-2. SU -> score -> decisione
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

    # 3. cartella temp locale con le sole incerte
    tmp_dir = Path(args.output_dir) / "_tmp_rerank_su"
    n = write_filtered_preds(rerank_ids, args.preds_dir, tmp_dir)
    print(f"[su] {n} query incerte copiate in {tmp_dir}")

    # 4. image matching totale (script del prof) solo sulle incerte
    if n > 0:
        run_matcher_on_dir(tmp_dir, args.output_dir, args.matcher,
                           args.device, args.im_size, args.num_preds)
    else:
        print("[su] nessuna query incerta: nessun image matching da fare.")

    # salva la lista delle skip (utile per la recall finale)
    skip_log = Path(args.output_dir) / "skipped_query_ids.txt"
    skip_log.write_text("\n".join(skip_ids))
    print(f"[su] {len(skip_ids)} query skip (top-1 del retrieval) -> {skip_log}")


if __name__ == "__main__":
    main(parse_args())
