"""
check_performance.py — Statistiche finali su un output di adaptive reranking.

Legge le cartelle top{K}/ prodotte dal deploy e riporta, in un colpo solo:
  - dove si sono fermate le query (distribuzione dei budget)
  - il costo in image matching e il risparmio rispetto al full rerank
  - la recall@N ADATTIVA e, come riferimento, quella BASE (solo retrieval)

Un positivo e' determinato dalla sezione "Positives paths" del .txt di
retrieval; se quella sezione manca (retrieval lanciato senza label) si passa
automaticamente alla distanza UTM, con la stessa soglia usata da reranking.py.
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from util import get_utm_from_path, compute_distance


def parse_prediction_txt(txt_file):
    """Legge un .txt di retrieval -> (query_path, pred_paths, positives).

    positives e' None se la sezione "Positives paths" non c'e' proprio
    (retrieval lanciato senza label): solo in quel caso serve il fallback sulla
    distanza. Un set vuoto significa invece "query senza nessun positivo", che
    e' un'informazione valida e va rispettata."""
    query_path = None
    pred_paths, positive_paths = [], []
    has_positives_section = False
    reading_query = reading_preds = reading_pos = False
    with open(txt_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("Query path"):
                reading_query, reading_preds, reading_pos = True, False, False
                continue
            if line.startswith("Predictions paths"):
                reading_query, reading_preds, reading_pos = False, True, False
                continue
            if line.startswith("Positives paths"):
                reading_query, reading_preds, reading_pos = False, False, True
                has_positives_section = True
                continue
            if not line:
                continue
            if reading_query and query_path is None:
                query_path = line
            elif reading_preds:
                pred_paths.append(line)
            elif reading_pos:
                positive_paths.append(line)
    return query_path, pred_paths, (set(positive_paths) if has_positives_section else None)


def positives_by_distance(query_path, pred_paths, dist_threshold):
    """Fallback: un candidato e' positivo se entro dist_threshold metri (UTM)."""
    q_utm = get_utm_from_path(query_path)
    return {p for p in pred_paths
            if compute_distance(q_utm, get_utm_from_path(p)) <= dist_threshold}


def find_query_file(adaptive_dir, q_id):
    """Trova (path, budget) del file della query q_id tra le cartelle topK.
    budget = numero di image matching effettivamente eseguiti per quella query
    (0 = decisione presa dal solo retrieval). (None, None) se non c'e'."""
    for folder in sorted(glob(os.path.join(adaptive_dir, "top*"))):
        name = os.path.basename(folder)
        if not name[3:].isdigit():
            continue
        budget = int(name[3:])              # 'top20' -> 20
        for ext in (".torch", ".txt"):
            fp = os.path.join(folder, f"{q_id}{ext}")
            if os.path.exists(fp):
                return fp, budget
    return None, None


def final_ranking(fp, pred_paths, num_preds):
    """Lista dei candidati (path) nell'ordine finale, lunga num_preds.

    .torch: primi k riordinati per num_inliers desc (tie-break: rank di
            retrieval crescente), poi coda in ordine di retrieval.
    .txt  : ordine di retrieval puro (nessun image matching fatto).
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


def evaluate(preds_dir, adaptive_dir, num_preds=20, recall_values=(1, 5, 10, 20),
             positive_dist_threshold=25):
    """Valuta un output di adaptive reranking e ritorna un dict di metriche.
    Nessuna stampa: usato sia da main() sia da run_all_methods.py."""
    preds_dir, adaptive_dir = Path(preds_dir), Path(adaptive_dir)

    q_ids = sorted((Path(f).stem for f in glob(str(preds_dir / "*.txt"))), key=int)
    if not q_ids:
        raise RuntimeError(f"Nessun .txt in {preds_dir}")
    if not adaptive_dir.exists():
        raise RuntimeError(f"Cartella di output non trovata: {adaptive_dir}")

    recalls = {n: 0 for n in recall_values}
    base_recalls = {n: 0 for n in recall_values}
    stop_counts, n_eval, missing, im_adaptive = {}, 0, 0, 0
    used_distance_fallback = False

    for q_id in q_ids:
        fp, budget = find_query_file(adaptive_dir, q_id)
        if fp is None:
            missing += 1
            continue
        stop_counts[budget] = stop_counts.get(budget, 0) + 1
        im_adaptive += budget

        query_path, pred_paths, positives = parse_prediction_txt(preds_dir / f"{q_id}.txt")
        if positives is None:
            if query_path is None:
                raise RuntimeError(f"{q_id}.txt: ne' 'Positives paths' ne' 'Query path'")
            positives = positives_by_distance(query_path, pred_paths, positive_dist_threshold)
            used_distance_fallback = True
        ranking = final_ranking(fp, pred_paths, num_preds)

        for n in recall_values:
            recalls[n] += recall_at_n(ranking, positives, n)
            base_recalls[n] += recall_at_n(pred_paths, positives, n)
        n_eval += 1

    if not n_eval:
        raise RuntimeError(
            f"Nessuna query trovata in {adaptive_dir}: il deploy non ha prodotto output.")

    im_full = n_eval * num_preds
    return {
        "n_queries": n_eval,
        "missing": missing,
        "distance_fallback": used_distance_fallback,
        "stop_counts": stop_counts,
        "im_adaptive": im_adaptive,
        "im_full": im_full,
        "matches_per_query": im_adaptive / n_eval,
        "saving_pct": 100 * (1 - im_adaptive / im_full) if im_full else 0.0,
        "recall": {n: 100 * recalls[n] / n_eval for n in recall_values},
        "base_recall": {n: 100 * base_recalls[n] / n_eval for n in recall_values},
    }


def print_report(res, recall_values, positive_dist_threshold=25):
    n_eval = res["n_queries"]
    print(f"\nQuery valutate: {n_eval}" +
          (f"  (mancanti: {res['missing']})" if res["missing"] else ""))
    if res["distance_fallback"]:
        print(f"  positivi determinati per distanza UTM <= {positive_dist_threshold} m")

    print("\nDistribuzione stop (dove si sono fermate le query):")
    for b in sorted(res["stop_counts"]):
        c = res["stop_counts"][b]
        label = "top-0 (nessun IM)" if b == 0 else f"top-{b}"
        print(f"  {label:<18}: {c:5d}  ({100*c/n_eval:.1f}%)")

    print("\nCosto (image matching):")
    print(f"  IM adattivi:       {res['im_adaptive']:6d}")
    print(f"  IM full-rerank:    {res['im_full']:6d}   (= {n_eval} query x "
          f"{res['im_full'] // n_eval})")
    print(f"  IM medi per query: {res['matches_per_query']:.2f}")
    print(f"  SAVING:            {res['saving_pct']:.1f}%   (IM risparmiati vs full-rerank)")

    print("\nRecall (adattiva vs base = solo retrieval):")
    for n in recall_values:
        r_ad, r_ba = res["recall"][n], res["base_recall"][n]
        print(f"  R@{n:<3} = {r_ad:6.2f}%   (base {r_ba:6.2f}%,  delta {r_ad - r_ba:+.2f})")


def main(args):
    res = evaluate(args.preds_dir, args.adaptive_RR_dir, args.num_preds,
                   args.recall_values, args.positive_dist_threshold)
    print_report(res, args.recall_values, args.positive_dist_threshold)


def parse_args():
    p = argparse.ArgumentParser(description="Statistiche adaptive reranking")
    p.add_argument("--preds-dir", required=True)
    p.add_argument("--adaptive-RR-dir", required=True)
    p.add_argument("--num-preds", type=int, default=20)
    p.add_argument("--recall-values", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--positive-dist-threshold", type=int, default=25,
                   help="metri; usato solo se il .txt non ha la sezione 'Positives paths'")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
