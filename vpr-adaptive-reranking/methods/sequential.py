"""
methods/sequential.py — Sequential:

  top-1 -> gate1 -> stop | top-5 -> gate5 -> stop | top-10 -> gate10 -> stop | top-20

MULTI-FEATURE, consistent with the trained models:
  gate1  (1):  num_inliers_top1                                                 target: helps_20
  gate5  (6):  num_inliers_top1, max_inliers_top5, second_max_inliers_top5,
               gap_inliers_top5, best_retrieval_rank_top5, top1_is_best_top5
  gate10 (10): num_inliers_top1, max_inliers_top5, gap_inliers_top5,
               best_retrieval_rank_top5, top1_is_best_top5, max_inliers_top10,
               second_max_inliers_top10, gap_inliers_top10,
               best_retrieval_rank_top10, top1_is_best_top10
The progressive features are built LIVE from the accumulated IM results.

MODELS: one JSON per gate, looked up in --models-dir by model+matcher, as in the validation.
THRESHOLDS: validation/sequential/threshold_<model>_<matcher>.csv (tau1,tau5,tau10).

OUTPUT (one folder per budget, each query in exactly one):
  output-dir/top1/<id>.torch    stopped at gate1   (1 IM result)
  output-dir/top5/<id>.torch    stopped at gate5   (5 results)
  output-dir/top10/<id>.torch   stopped at gate10  (10 results)
  output-dir/top20/<id>.torch   reached top-20     (20 results)
"""
import argparse
import json
import sys
from glob import glob
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "validation"))
from _common import (
    load_threshold_csv, get_query_ids, budget_folder,
    run_im_top1_with_results, run_im_extend,
    sequential_features, apply_sigmoid_vector,
    save_results_torch, query_already_done,
)
from _outputs import canon_model, canon_matcher

_VALIDATION_DIR = Path(__file__).resolve().parent.parent / "validation" / "sequential"

MATCHER_FILE_TOKENS = {"superpoint-lg": ("superpoint-lg", "sp-lg"), "loftr": ("loftr",)}


def _find_gate_json(models_dir, gate_num, model, matcher):
    pats = []
    for tok in MATCHER_FILE_TOKENS.get(matcher, (matcher,)):
        pats += [f"*continue_{gate_num}_*{model}*{tok}*.json",
                 f"*continue_{gate_num}*{model}*{tok}*.json"]
    for pat in pats:
        hits = sorted(glob(str(Path(models_dir) / pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No JSON for gate{gate_num} in {models_dir} (model={model}, matcher={matcher}).")


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
    parser.add_argument("--models-dir", default=None,
                        help="folder with the 3 gate JSONs (default: validation/sequential/)")
    parser.add_argument("--model",      required=True, help="retrieval model (cosplace/megaloc)")
    parser.add_argument("--matcher",    required=True, help="image matcher (superpoint-lg/loftr)")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--im-size",    type=int, default=512)
    parser.add_argument("--num-preds",  type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inliers-dir", default=None,
                        help="OFFLINE: folder with the already computed top-20 .torch files "
                             "(no image matching, read only)")
    parser.add_argument("--models-json-dir", default=None,
                        help="alias of --models-dir")
    return parser.parse_args()


def print_stage(label, stopped_ids, total):
    pct = 100 * len(stopped_ids) / total if total else 0.0
    print(f"  stopped at {label}: {len(stopped_ids):5d}  ({pct:.1f}%)")


def finalize(query_ids, accumulated, output_dir, budget):
    folder = budget_folder(output_dir, budget)
    for q in query_ids:
        save_results_torch(q, accumulated[q], folder)


def finalize_partial(query_ids, accumulated, output_dir):
    """Queries that could not be extended (offline: short or missing .torch):
    each one is closed at the budget it actually reached."""
    for q in query_ids:
        n = len(accumulated.get(q, []))
        if n:
            save_results_torch(q, accumulated[q], budget_folder(output_dir, n))


def _split_by_gate(query_ids, accumulated, gate, model_json, tau):
    """continue if P(gate) > tau. Returns (continue_ids, stop_ids)."""
    cont, stop = [], []
    for q in query_ids:
        feats = sequential_features(accumulated[q], gate)
        p = apply_sigmoid_vector(feats, model_json)
        (cont if p > tau else stop).append(q)
    return cont, stop


def main(args):
    model, matcher = canon_model(args.model), canon_matcher(args.matcher)

    threshold_csv = _VALIDATION_DIR / f"threshold_{model}_{matcher}.csv"
    if not threshold_csv.exists():
        raise FileNotFoundError(
            f"Thresholds not found: {threshold_csv}\n"
            f"  -> run first: validation/sequential.py --model {model} "
            f"--matcher {matcher} --val-csv <candidate_level_val.csv>")
    hp = load_threshold_csv(threshold_csv)   # tau1, tau5, tau10

    models_dir = args.models_dir or args.models_json_dir or _VALIDATION_DIR
    models = load_gate_models(models_dir, model, matcher)
    print(f"tau1={hp['tau1']}  tau5={hp['tau5']}  tau10={hp['tau10']}  "
          f"[{model}/{matcher}]")

    budgets = (1, 5, 10, args.num_preds)
    all_ids = get_query_ids(args.preds_dir)
    total = len(all_ids)

    pending = [q for q in all_ids if not query_already_done(args.output_dir, q, budgets)]
    if total - len(pending):
        print(f"Resume: {total - len(pending)} queries already completed, skipping.")

    # Stage 1: IM top-1
    accumulated = run_im_top1_with_results(args.preds_dir, args.matcher,
                                           args.device, args.im_size,
                                           query_ids=pending,
                                           inliers_dir=args.inliers_dir)
    accumulated = {q: [r] for q, r in accumulated.items()}
    pending = [q for q in pending if q in accumulated]

    # Gate 1 (1 -> 5)
    continue_5, stop_1 = _split_by_gate(pending, accumulated, "gate1",
                                        models["gate1"], hp["tau1"])
    finalize(stop_1, accumulated, args.output_dir, 1)
    print_stage("top-1", stop_1, total)

    # Stage 2 + Gate 5 (5 -> 10)
    failed = run_im_extend(args.preds_dir, continue_5, accumulated, 2, 5,
                           args.matcher, args.device, args.im_size,
                           inliers_dir=args.inliers_dir)
    finalize_partial(failed, accumulated, args.output_dir)
    continue_5 = [q for q in continue_5 if q not in set(failed)]
    continue_10, stop_5 = _split_by_gate(continue_5, accumulated, "gate5",
                                         models["gate5"], hp["tau5"])
    finalize(stop_5, accumulated, args.output_dir, 5)
    print_stage("top-5", stop_5, total)

    # Stage 3 + Gate 10 (10 -> 20)
    failed = run_im_extend(args.preds_dir, continue_10, accumulated, 6, 10,
                           args.matcher, args.device, args.im_size,
                           inliers_dir=args.inliers_dir)
    finalize_partial(failed, accumulated, args.output_dir)
    continue_10 = [q for q in continue_10 if q not in set(failed)]
    continue_20, stop_10 = _split_by_gate(continue_10, accumulated, "gate10",
                                          models["gate10"], hp["tau10"])
    finalize(stop_10, accumulated, args.output_dir, 10)
    print_stage("top-10", stop_10, total)

    # Stage 4: top-20, no further gate
    failed = run_im_extend(args.preds_dir, continue_20, accumulated, 11, args.num_preds,
                           args.matcher, args.device, args.im_size,
                           inliers_dir=args.inliers_dir)
    finalize_partial(failed, accumulated, args.output_dir)
    continue_20 = [q for q in continue_20 if q not in set(failed)]
    finalize(continue_20, accumulated, args.output_dir, args.num_preds)
    print_stage("top-20", continue_20, total)


if __name__ == "__main__":
    main(parse_args())
