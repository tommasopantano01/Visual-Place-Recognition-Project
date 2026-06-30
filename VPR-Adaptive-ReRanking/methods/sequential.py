"""
sequential/sequential.py — Sequential adaptive re-ranking: cascata a tre
cancelli invece di una singola decisione binaria.

  top-1 -> gate1 -> stop | top-5 -> gate5 -> stop | top-10 -> gate10 -> stop | top-20

Ogni cancello e' un regressore logistico a singola feature:
  gate1  (1->5):   input = num_inliers del top-1
  gate5  (5->10):  input = max(num_inliers) tra i candidati visti (top-5)
  gate10 (10->20): input = max(num_inliers) tra i candidati visti (top-10)
probability > tau -> continua al budget successivo. Altrimenti ferma qui.

NOTA: gate5/gate10 = M5/M10 del report (confermati). gate1 e' ancora
approssimato riusando P(helps|I_1) (local/): target di training da
confermare con Luca.

OUTPUT (una cartella per budget, ogni query in esattamente una):
  output-dir/top1/<id>.torch    fermate al gate1   (1 risultato IM)
  output-dir/top5/<id>.torch    fermate al gate5   (5 risultati)
  output-dir/top10/<id>.torch   fermate al gate10  (10 risultati)
  output-dir/top20/<id>.torch   arrivate a top-20  (20 risultati)
check_performance.py conta i file in ogni topK per la distribuzione degli stop.

RESUME (per-query): se interrotto, una nuova run con lo stesso --output-dir
salta le query gia' finalizzate in un topK; le altre vengono ricalcolate da
zero (niente checkpoint intra-stage).

Uso:
    python VPR-adaptive-re-ranking/sequential/sequential.py \
        --preds-dir preds/ --matcher superpoint-lg --output-dir out/
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import (
    load_threshold_csv, load_model_json, get_query_ids, budget_folder,
    run_im_top1_with_results, run_im_extend, apply_sigmoid,
    partition_by_probability, save_results_torch, query_already_done,
)

_THRESHOLD_CSV = Path(__file__).parent / "threshold.csv"
_MODEL_JSON    = Path(__file__).parent / "model.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive reranking — sequential (1->5->10->20)")
    parser.add_argument("--preds-dir",  required=True)
    parser.add_argument("--matcher",    required=True)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--im-size",    type=int, default=512)
    parser.add_argument("--num-preds",  type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def print_stage(label, stopped_ids, total):
    pct = 100 * len(stopped_ids) / total if total else 0.0
    print(f"  fermate a {label}: {len(stopped_ids):5d}  ({pct:.1f}%)")


def finalize(query_ids, accumulated, output_dir, budget):
    """Salva le query che si fermano a `budget` in output_dir/top{budget}/."""
    folder = budget_folder(output_dir, budget)
    for q in query_ids:
        save_results_torch(q, accumulated[q], folder)


def main(args):
    hp     = load_threshold_csv(_THRESHOLD_CSV)   # tau1, tau5, tau10
    models = load_model_json(_MODEL_JSON)         # gate1, gate5, gate10
    print(f"tau1={hp['tau1']}  tau5={hp['tau5']}  tau10={hp['tau10']}")

    budgets = (1, 5, 10, args.num_preds)
    all_ids = get_query_ids(args.preds_dir)
    total = len(all_ids)

    # Resume: salta le query gia' finalizzate in un topK
    pending = [q for q in all_ids if not query_already_done(args.output_dir, q, budgets)]
    if total - len(pending):
        print(f"Resume: {total - len(pending)} query gia' completate, salto.")

    # --- Stage 1: IM top-1 ---
    accumulated = run_im_top1_with_results(args.preds_dir, args.matcher,
                                           args.device, args.im_size, query_ids=pending)
    accumulated = {q: [r] for q, r in accumulated.items()}

    # --- Gate 1 (1->5) ---
    signals1 = {q: {"inliers": accumulated[q][0]["num_inliers"]} for q in pending}
    continue_5, stop_1 = partition_by_probability(
        apply_sigmoid(signals1, models["gate1"]), hp["tau1"])
    finalize(stop_1, accumulated, args.output_dir, 1)
    print_stage("top-1", stop_1, total)

    # --- Stage 2 + Gate 2 (5->10) ---
    run_im_extend(args.preds_dir, continue_5, accumulated, 2, 5,
                  args.matcher, args.device, args.im_size)
    signals5 = {q: {"inliers": max(r["num_inliers"] for r in accumulated[q])} for q in continue_5}
    continue_10, stop_5 = partition_by_probability(
        apply_sigmoid(signals5, models["gate5"]), hp["tau5"])
    finalize(stop_5, accumulated, args.output_dir, 5)
    print_stage("top-5", stop_5, total)

    # --- Stage 3 + Gate 3 (10->20) ---
    run_im_extend(args.preds_dir, continue_10, accumulated, 6, 10,
                  args.matcher, args.device, args.im_size)
    signals10 = {q: {"inliers": max(r["num_inliers"] for r in accumulated[q])} for q in continue_10}
    continue_20, stop_10 = partition_by_probability(
        apply_sigmoid(signals10, models["gate10"]), hp["tau10"])
    finalize(stop_10, accumulated, args.output_dir, 10)
    print_stage("top-10", stop_10, total)

    # --- Stage 4: top-20, nessun ulteriore cancello ---
    run_im_extend(args.preds_dir, continue_20, accumulated, 11, args.num_preds,
                  args.matcher, args.device, args.im_size)
    finalize(continue_20, accumulated, args.output_dir, args.num_preds)
    print_stage("top-20", continue_20, total)


if __name__ == "__main__":
    main(parse_args())
