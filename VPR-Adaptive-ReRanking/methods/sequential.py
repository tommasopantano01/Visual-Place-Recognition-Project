"""
methods/sequential.py — Sequential adaptive re-ranking: cascata a tre cancelli.

  top-1 -> gate1 -> stop | top-5 -> gate5 -> stop | top-10 -> gate10 -> stop | top-20

MULTI-FEATURE, coerente coi modelli allenati (notebook "POLICY SEQUENZIALE"):
  gate1  (1):  num_inliers_top1                               target: helps_20
  gate5  (6):  num_inliers_top1, max_inliers_top5, second_max_inliers_top5,
               gap_inliers_top5, best_retrieval_rank_top5, top1_is_best_top5
  gate10 (10): num_inliers_top1, max_inliers_top5, gap_inliers_top5,
               best_retrieval_rank_top5, top1_is_best_top5, max_inliers_top10,
               second_max_inliers_top10, gap_inliers_top10,
               best_retrieval_rank_top10, top1_is_best_top10
Le feature progressive sono costruite LIVE dai risultati IM accumulati, con la
stessa logica del notebook (ordine per num_inliers desc, tie retrieval_rank asc).
Vedi _common.sequential_features. probability > tau -> continua.

MODELLI: un JSON per gate (formato regressor_to_dict con feat_cols anonime),
cercati nella --models-dir per model+matcher, come nella validation.
SOGLIE: validation/sequential/threshold_<model>_<matcher>.csv (tau1,tau5,tau10).

OUTPUT (una cartella per budget, ogni query in esattamente una):
  output-dir/top1/<id>.torch    fermate al gate1   (1 risultato IM)
  output-dir/top5/<id>.torch    fermate al gate5   (5 risultati)
  output-dir/top10/<id>.torch   fermate al gate10  (10 risultati)
  output-dir/top20/<id>.torch   arrivate a top-20  (20 risultati)

RESUME (per-query): con lo stesso --output-dir salta le query gia' finalizzate.

Uso:
    python VPR-Adaptive-ReRanking/methods/sequential.py \
        --preds-dir preds/ --models-dir <dir json> \
        --model cosplace --matcher sp-lg --output-dir out/
"""
import argparse
import json
import sys
from glob import glob
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import (
    load_threshold_csv, get_query_ids, budget_folder,
    run_im_top1_with_results, run_im_extend,
    sequential_features, apply_sigmoid_vector,
    save_results_torch, query_already_done,
)

_VALIDATION_DIR = Path(__file__).resolve().parent.parent / "validation" / "sequential"


def _find_gate_json(models_dir, gate_num, model, matcher):
    for pat in (f"*continue_{gate_num}_*{model}*{matcher}*.json",
                f"*continue_{gate_num}*{model}*{matcher}*.json"):
        hits = sorted(glob(str(Path(models_dir) / pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Nessun JSON per gate{gate_num} in {models_dir} (model={model}, matcher={matcher}).")


def load_gate_models(models_dir, model, matcher):
    models = {}
    for g, n in (("gate1", 1), ("gate5", 5), ("gate10", 10)):
        p = _find_gate_json(models_dir, n, model, matcher)
        with open(p) as f:
            models[g] = json.load(f)
        print(f"  {g} <- {Path(p).name}")
    return models


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive reranking — sequential (multi-feature)")
    parser.add_argument("--preds-dir",  required=True)
    parser.add_argument("--models-dir", required=True, help="cartella coi 3 JSON dei gate")
    parser.add_argument("--model",      required=True, help="retrieval model (cosplace/megaloc)")
    parser.add_argument("--matcher",    required=True, help="image matcher (sp-lg/loftr)")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--im-size",    type=int, default=512)
    parser.add_argument("--num-preds",  type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def print_stage(label, stopped_ids, total):
    pct = 100 * len(stopped_ids) / total if total else 0.0
    print(f"  fermate a {label}: {len(stopped_ids):5d}  ({pct:.1f}%)")


def finalize(query_ids, accumulated, output_dir, budget):
    folder = budget_folder(output_dir, budget)
    for q in query_ids:
        save_results_torch(q, accumulated[q], folder)


def _split_by_gate(query_ids, accumulated, gate, model_json, tau):
    """continua se P(gate) > tau. Ritorna (continue_ids, stop_ids)."""
    cont, stop = [], []
    for q in query_ids:
        feats = sequential_features(accumulated[q], gate)
        p = apply_sigmoid_vector(feats, model_json)
        (cont if p > tau else stop).append(q)
    return cont, stop


def main(args):
    threshold_csv = _VALIDATION_DIR / f"threshold_{args.model}_{args.matcher}.csv"
    hp = load_threshold_csv(threshold_csv)   # tau1, tau5, tau10
    models = load_gate_models(args.models_dir, args.model, args.matcher)
    print(f"tau1={hp['tau1']}  tau5={hp['tau5']}  tau10={hp['tau10']}  "
          f"[{args.model}/{args.matcher}]")

    budgets = (1, 5, 10, args.num_preds)
    all_ids = get_query_ids(args.preds_dir)
    total = len(all_ids)

    pending = [q for q in all_ids if not query_already_done(args.output_dir, q, budgets)]
    if total - len(pending):
        print(f"Resume: {total - len(pending)} query gia' completate, salto.")

    # Stage 1: IM top-1
    accumulated = run_im_top1_with_results(args.preds_dir, args.matcher,
                                           args.device, args.im_size, query_ids=pending)
    accumulated = {q: [r] for q, r in accumulated.items()}

    # Gate 1 (1 -> 5)
    continue_5, stop_1 = _split_by_gate(pending, accumulated, "gate1",
                                        models["gate1"], hp["tau1"])
    finalize(stop_1, accumulated, args.output_dir, 1)
    print_stage("top-1", stop_1, total)

    # Stage 2 + Gate 5 (5 -> 10)
    run_im_extend(args.preds_dir, continue_5, accumulated, 2, 5,
                  args.matcher, args.device, args.im_size)
    continue_10, stop_5 = _split_by_gate(continue_5, accumulated, "gate5",
                                         models["gate5"], hp["tau5"])
    finalize(stop_5, accumulated, args.output_dir, 5)
    print_stage("top-5", stop_5, total)

    # Stage 3 + Gate 10 (10 -> 20)
    run_im_extend(args.preds_dir, continue_10, accumulated, 6, 10,
                  args.matcher, args.device, args.im_size)
    continue_20, stop_10 = _split_by_gate(continue_10, accumulated, "gate10",
                                          models["gate10"], hp["tau10"])
    finalize(stop_10, accumulated, args.output_dir, 10)
    print_stage("top-10", stop_10, total)

    # Stage 4: top-20, nessun ulteriore cancello
    run_im_extend(args.preds_dir, continue_20, accumulated, 11, args.num_preds,
                  args.matcher, args.device, args.im_size)
    finalize(continue_20, accumulated, args.output_dir, args.num_preds)
    print_stage("top-20", continue_20, total)


if __name__ == "__main__":
    main(parse_args())
