#!/usr/bin/env python3
"""
check_performance.py — Statistiche finali su un output di adaptive reranking.

Per ogni query legge dove si e' fermata (cartella topK in --adaptive-RR-dir),
ricostruisce il ranking finale e calcola recall@N.

Ranking finale per query:
  - file .torch (k risultati IM): ordina i primi k candidati per num_inliers
    desc (tie-break: ordine di retrieval), poi accoda i restanti candidati
    nell'ordine di retrieval (coda non rerankata);
  - file .txt (skip senza IM, es. su/): ordine di retrieval puro.

recall@N: 1 se almeno un candidato positivo nei primi N del ranking finale.

Uso:
    python VPR-Adaptive-ReRanking/check_performance.py \
        --preds-dir       <preds folder> \
        --adaptive-RR-dir <output del reranking adattivo> \
        --num-preds 20 --recall-values 1 5 10 20
"""

import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch


def parse_prediction_txt(txt_file):
    """Legge un .txt di retrieval -> (pred_paths in ordine, set positive_paths)."""
    pred_paths, positive_paths = [], []
    reading_preds = reading_pos = False
    with open(txt_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("Predictions paths"):
                reading_preds, reading_pos = True, False
                continue
            if line.startswith("Positives paths"):
                reading_preds, reading_pos = False, True
                continue
            if reading_preds and line:
                pred_paths.append(line)
            elif reading_pos and line:
                positive_paths.append(line)
    return pred_paths, set(positive_paths)


def find_query_file(adaptive_dir, q_id):
    """Trova (path, budget) del file della query q_id tra le cartelle topK.
    Ritorna (None, None) se non c'e'."""
    for folder in sorted(glob(os.path.join(adaptive_dir, "top*"))):
        budget = int(os.path.basename(folder)[3:])  # 'top20' -> 20
        for ext in (".torch", ".txt"):
            fp = os.path.join(folder, f"{q_id}{ext}")
            if os.path.exists(fp):
                return fp, budget
    return None, None


def final_ranking(fp, pred_paths, num_preds):
    """Lista dei candidati (path) nell'ordine finale, lunga num_preds.

    .torch: primi k riordinati per num_inliers desc (tie-break retrieval),
            poi coda in ordine di retrieval.
    .txt  : ordine di retrieval puro.
    """
    if fp.endswith(".txt"):
        return pred_paths[:num_preds]

    data = torch.load(fp, map_location="cpu", weights_only=False)
    k = len(data)  # candidati effettivamente matchati (budget)

    inliers = np.array([int(d["num_inliers"]) for d in data])
    retr_rank = np.arange(k)  # 0..k-1 = ordine di retrieval dei primi k
    # ordina per inliers desc, a parita' per rank di retrieval asc
    order = np.lexsort((retr_rank, -inliers))

    reranked_head = [pred_paths[i] for i in order]          # primi k riordinati
    tail = pred_paths[k:num_preds]                          # coda non rerankata
    return reranked_head + tail


def recall_at_n(ranking, positives, n):
    """1 se almeno un positivo nei primi n del ranking."""
    return int(any(p in positives for p in ranking[:n]))


def main(args):
    preds_dir = Path(args.preds_dir)
    adaptive_dir = Path(args.adaptive_RR_dir)

    q_ids = sorted((Path(f).stem for f in glob(str(preds_dir / "*.txt"))), key=int)
    if not q_ids:
        raise RuntimeError(f"Nessun .txt in {preds_dir}")

    recalls = {n: 0 for n in args.recall_values}
    stop_counts = {}   # budget -> quante query
    n_eval = 0
    missing = 0

    for q_id in q_ids:
        fp, budget = find_query_file(adaptive_dir, q_id)
        if fp is None:
            missing += 1
            continue
        stop_counts[budget] = stop_counts.get(budget, 0) + 1

        pred_paths, positives = parse_prediction_txt(preds_dir / f"{q_id}.txt")
        ranking = final_ranking(fp, pred_paths, args.num_preds)

        for n in args.recall_values:
            recalls[n] += recall_at_n(ranking, positives, n)
        n_eval += 1

    print(f"\nQuery valutate: {n_eval}" + (f"  (mancanti: {missing})" if missing else ""))
    print("\nDistribuzione stop (dove si sono fermate le query):")
    for b in sorted(stop_counts):
        c = stop_counts[b]
        print(f"  top-{b:<2}: {c:5d}  ({100*c/n_eval:.1f}%)")

    print("\nRecall:")
    for n in args.recall_values:
        print(f"  R@{n:<2} = {100*recalls[n]/n_eval:.2f}%")


def parse_args():
    p = argparse.ArgumentParser(description="Statistiche adaptive reranking")
    p.add_argument("--preds-dir", required=True)
    p.add_argument("--adaptive-RR-dir", required=True)
    p.add_argument("--num-preds", type=int, default=20)
    p.add_argument("--recall-values", type=int, nargs="+", default=[1, 5, 10, 20])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
