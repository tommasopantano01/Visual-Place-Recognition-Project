import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import (
    load_threshold_csv, load_model_json, run_logistic_single,
    run_im_top1_with_results, apply_sigmoid, partition_by_probability,
    save_results_torch, run_im_topN_subset, print_summary, budget_folder,
)

_ARR_DIR = Path(__file__).resolve().parent.parent

# nome metodo -> (sottocartella validation, template del model json)
METHODS = {
    "hard":           ("logistic_hard",           "model_{model}_{matcher}.json"),
    "help":           ("logistic_help",           "model_{model}_{matcher}.json"),
    "cost_sensitive": ("logistic_cost_sensitive", "model_logistic_cost_sensitive_{model}_{matcher}.json"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive reranking — logistic su num-inliers (hard | help | cost_sensitive)")
    parser.add_argument("--method",     required=True, choices=sorted(METHODS.keys()))
    parser.add_argument("--preds-dir",  required=True)
    parser.add_argument("--model",      required=True, help="cosplace or megaloc")
    parser.add_argument("--matcher",    required=True)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--im-size",    type=int, default=512)
    parser.add_argument("--num-preds",  type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main(args):
    subdir, json_tmpl = METHODS[args.method]
    val_dir = _ARR_DIR / "validation" / subdir
    hp = load_threshold_csv(val_dir / f"threshold_{args.model}_{args.matcher}.csv")
    model_json = val_dir / json_tmpl.format(model=args.model, matcher=args.matcher)

    if args.method in ("hard", "help"):
        tau = hp["tau"]
        regressor = load_model_json(model_json)
        print(f"tau (P_{args.method}) = {tau}  [{args.model}/{args.matcher}]")
        run_logistic_single(args.preds_dir, tau, regressor, args.matcher, args.device,
                            args.im_size, args.num_preds, args.output_dir)
        return

    # cost_sensitive: score = P(help) - lambda * P(hurt)
    lam, tau = hp["alpha"], hp["tau"]
    models = load_model_json(model_json)   # {"help": {...}, "hurt": {...}}
    print(f"lambda = {lam}   tau = {tau}   [{args.model}/{args.matcher}]")

    results_top1 = run_im_top1_with_results(args.preds_dir, args.matcher, args.device, args.im_size)
    signals = {q: {"inliers": r["num_inliers"]} for q, r in results_top1.items()}
    p_help = apply_sigmoid(signals, models["help"])
    p_hurt = apply_sigmoid(signals, models["hurt"])
    scores = {q: p_help[q] - lam * p_hurt[q] for q in results_top1}

    rerank_ids, skip_ids = partition_by_probability(scores, tau)
    print_summary(rerank_ids, skip_ids)

    folder1 = budget_folder(args.output_dir, 1)
    for q in skip_ids:
        save_results_torch(q, [results_top1[q]], folder1)
    run_im_topN_subset(args.preds_dir, rerank_ids, args.output_dir, args.num_preds,
                       args.matcher, args.device, args.im_size)


if __name__ == "__main__":
    main(parse_args())
