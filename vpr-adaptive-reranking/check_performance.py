"""
check_performance.py — Final statistics for an adaptive reranking output:
  - where the queries stopped (budget distribution)
  - the image matching cost and the saving vs full rerank
  - the ADAPTIVE recall@N and, as a reference, the BASE one (retrieval only)

A positive is determined from the "Positives paths" section of the retrieval
.txt; if that section is missing (retrieval run without labels), it falls
back automatically to the UTM distance, with the same threshold used by
reranking.py.
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
    """Reads a retrieval .txt -> (query_path, pred_paths, positives).
    """
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
    q_utm = get_utm_from_path(query_path)
    return {p for p in pred_paths
            if compute_distance(q_utm, get_utm_from_path(p)) <= dist_threshold}


def find_query_file(adaptive_dir, q_id):
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
    """List of candidates (paths) in final order, num_preds long.

    .torch: the first k reordered by num_inliers desc (tie-break: ascending
            retrieval rank), then the tail in retrieval order.
    .txt  : pure retrieval order (no image matching done).
    """
    if fp.endswith(".txt"):
        return pred_paths[:num_preds]

    data = torch.load(fp, map_location="cpu", weights_only=False)
    k = len(data)  # candidates actually matched (budget)

    inliers = np.array([int(d["num_inliers"]) for d in data])
    retr_rank = np.arange(k)  # 0..k-1 = retrieval order of the first k
    order = np.lexsort((retr_rank, -inliers))

    reranked_head = [pred_paths[i] for i in order]
    tail = pred_paths[k:num_preds]
    return reranked_head + tail


def recall_at_n(ranking, positives, n):
    """1 if at least one positive is in the first n of the ranking."""
    return int(any(p in positives for p in ranking[:n]))


def evaluate(preds_dir, adaptive_dir, num_preds=20, recall_values=(1, 5, 10, 20),
             positive_dist_threshold=25):
    """Evaluates an adaptive reranking output and returns a dict of metrics."""
    preds_dir, adaptive_dir = Path(preds_dir), Path(adaptive_dir)

    q_ids = sorted((Path(f).stem for f in glob(str(preds_dir / "*.txt"))), key=int)
    if not q_ids:
        raise RuntimeError(f"No .txt in {preds_dir}")
    if not adaptive_dir.exists():
        raise RuntimeError(f"Output folder not found: {adaptive_dir}")

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
                raise RuntimeError(f"{q_id}.txt: neither 'Positives paths' nor 'Query path'")
            positives = positives_by_distance(query_path, pred_paths, positive_dist_threshold)
            used_distance_fallback = True
        ranking = final_ranking(fp, pred_paths, num_preds)

        for n in recall_values:
            recalls[n] += recall_at_n(ranking, positives, n)
            base_recalls[n] += recall_at_n(pred_paths, positives, n)
        n_eval += 1

    if not n_eval:
        raise RuntimeError(
            f"No query found in {adaptive_dir}: the deploy did not produce any output.")

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
    print(f"\nQueries evaluated: {n_eval}" +
          (f"  (missing: {res['missing']})" if res["missing"] else ""))
    if res["distance_fallback"]:
        print(f"  positives determined by UTM distance <= {positive_dist_threshold} m")

    print("\nStop distribution (where the queries stopped):")
    for b in sorted(res["stop_counts"]):
        c = res["stop_counts"][b]
        label = "top-0 (no IM)" if b == 0 else f"top-{b}"
        print(f"  {label:<18}: {c:5d}  ({100*c/n_eval:.1f}%)")

    print("\nCost (image matching):")
    print(f"  Adaptive IM:       {res['im_adaptive']:6d}")
    print(f"  Full-rerank IM:    {res['im_full']:6d}   (= {n_eval} queries x "
          f"{res['im_full'] // n_eval})")
    print(f"  Avg IM per query:  {res['matches_per_query']:.2f}")
    print(f"  SAVING:            {res['saving_pct']:.1f}%   (IM saved vs full-rerank)")

    print("\nRecall (adaptive vs base = retrieval only):")
    for n in recall_values:
        r_ad, r_ba = res["recall"][n], res["base_recall"][n]
        print(f"  R@{n:<3} = {r_ad:6.2f}%   (base {r_ba:6.2f}%,  delta {r_ad - r_ba:+.2f})")


def main(args):
    res = evaluate(args.preds_dir, args.adaptive_RR_dir, args.num_preds,
                   args.recall_values, args.positive_dist_threshold)
    print_report(res, args.recall_values, args.positive_dist_threshold)


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive reranking statistics")
    p.add_argument("--preds-dir", required=True)
    p.add_argument("--adaptive-RR-dir", required=True)
    p.add_argument("--num-preds", type=int, default=20)
    p.add_argument("--recall-values", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--positive-dist-threshold", type=int, default=25,
                   help="meters; used only if the .txt has no 'Positives paths' section")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
