"""
methods/logistic.py — Test of the logistic family on num_inliers_top1
(hard | help | cost_sensitive).

Rule:
  hard            rerank if P(hard) > tau
  help            rerank if P(help) > tau
  cost_sensitive  rerank if P(help) - alpha*P(hurts) > tau

The regressor is read from the model JSON in validation/<subdir>/, the
parameters (tau, alpha) from validation/<subdir>/threshold_<model>_<matcher>.csv.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "validation"))
from _common import (
    load_threshold_csv, load_model_json, run_logistic_single,
    run_im_top1_with_results, apply_sigmoid, partition_by_probability,
    save_results_torch, run_im_topN_subset, print_summary, budget_folder,
)
from _outputs import canon_model, canon_matcher

_ARR_DIR = Path(__file__).resolve().parent.parent

# method name -> (validation subfolder, model json template)
METHODS = {
    "hard":           ("logistic_hard",           "model_{model}_{matcher}.json"),
    "help":           ("logistic_help",           "model_{model}_{matcher}.json"),
    "cost_sensitive": ("logistic_cost_sensitive", "model_logistic_cost_sensitive_{model}_{matcher}.json"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive reranking — logistic on num-inliers (hard | help | cost_sensitive)")
    parser.add_argument("--method",     required=True, choices=sorted(METHODS.keys()))
    parser.add_argument("--preds-dir",  required=True)
    parser.add_argument("--model",      required=True, help="cosplace or megaloc")
    parser.add_argument("--matcher",    required=True, help="superpoint-lg or loftr")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--im-size",    type=int, default=512)
    parser.add_argument("--num-preds",  type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inliers-dir", default=None,
                        help="OFFLINE: folder with the already computed top-20 .torch files "
                             "(no image matching, read only)")
    parser.add_argument("--model-json", default=None,
                        help="default: validation/<subdir>/<template>")
    parser.add_argument("--threshold-csv", default=None,
                        help="default: validation/<subdir>/threshold_<model>_<matcher>.csv")
    return parser.parse_args()


def _resolve(args, model, matcher):
    """Threshold and model JSON paths"""
    subdir, json_tmpl = METHODS[args.method]
    val_dir = _ARR_DIR / "validation" / subdir
    threshold_csv = Path(args.threshold_csv or val_dir / f"threshold_{model}_{matcher}.csv")
    model_json = Path(args.model_json or
                      val_dir / json_tmpl.format(model=model, matcher=matcher))
    if not threshold_csv.exists():
        raise FileNotFoundError(
            f"Threshold not found: {threshold_csv}\n"
            f"  -> run first: validation/logistic.py --method {args.method} "
            f"--model {model} --matcher {matcher} --val-csv <candidate_level_val.csv>")
    if not model_json.exists():
        raise FileNotFoundError(
            f"Model JSON not found: {model_json}\n"
            "  -> download the trained regressors or pass --model-json")
    return threshold_csv, model_json


def main(args):
    model, matcher = canon_model(args.model), canon_matcher(args.matcher)
    threshold_csv, model_json = _resolve(args, model, matcher)
    hp = load_threshold_csv(threshold_csv)

    if args.method in ("hard", "help"):
        tau = hp["tau"]
        regressor = load_model_json(model_json)
        print(f"tau (P_{args.method}) = {tau}  [{model}/{matcher}]")
        run_logistic_single(args.preds_dir, tau, regressor, args.matcher, args.device,
                            args.im_size, args.num_preds, args.output_dir,
                            inliers_dir=args.inliers_dir)
        return

    # cost_sensitive: score = P(help) - alpha * P(hurts)
    lam, tau = hp["alpha"], hp["tau"]
    data = load_model_json(model_json)
    # nested JSON: {"feature_sets": {<fs>: {"regressors": {"help": {...}, "hurts": {...}}}}}
    if "feature_sets" in data:
        models = data["feature_sets"][next(iter(data["feature_sets"]))]["regressors"]
    else:                                  # flat: {"help": {...}, "hurts": {...}}
        models = data
    hurt_key = "hurts" if "hurts" in models else "hurt"
    print(f"alpha = {lam}   tau = {tau}   [{model}/{matcher}]")

    results_top1 = run_im_top1_with_results(args.preds_dir, args.matcher, args.device,
                                            args.im_size, inliers_dir=args.inliers_dir)
    signals = {q: {"inliers": r["num_inliers"]} for q, r in results_top1.items()}
    p_help = apply_sigmoid(signals, models["help"])
    p_hurt = apply_sigmoid(signals, models[hurt_key])
    scores = {q: p_help[q] - lam * p_hurt[q] for q in results_top1}

    rerank_ids, skip_ids = partition_by_probability(scores, tau)
    print_summary(rerank_ids, skip_ids)

    folder1 = budget_folder(args.output_dir, 1)
    for q in skip_ids:
        save_results_torch(q, [results_top1[q]], folder1)
    run_im_topN_subset(args.preds_dir, rerank_ids, args.output_dir, args.num_preds,
                       args.matcher, args.device, args.im_size,
                       inliers_dir=args.inliers_dir)


if __name__ == "__main__":
    main(parse_args())
