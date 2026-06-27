import os
import sys
import argparse
import torch
from glob import glob
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from util import read_file_preds

sys.path.append(str(Path(__file__).parent.joinpath("image-matching-models")))
from matching import get_matcher, available_models
from matching.utils import get_default_device


# ---------------------------------------------------------------------------
# Extension 6.1 — Threshold lookup table
#
# Key:   (threshold_type, matcher_name, vpr_method)
# Value: int — if num_inliers of top-1 > value, skip matching the other 19
#
# 4 threshold types × 4 (matcher, vpr_method) combinations = 16 entries.
# TODO: replace None with the actual values once estimated on the datasets.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    # threshold estimated only on "easy" query-DB pairs
    ("easy_only",  "superpoint-lg", "cosplace"):  None,  # TODO
    ("easy_only",  "superpoint-lg", "method2"):   None,  # TODO
    ("easy_only",  "superpoint-lg", "method3"):   None,  # TODO
    ("easy_only",  "superpoint-lg", "method4"):   None,  # TODO
    # threshold estimated on "easy" + "hard" pairs
    ("easy_hard",  "superpoint-lg", "cosplace"):  None,  # TODO
    ("easy_hard",  "superpoint-lg", "method2"):   None,  # TODO
    ("easy_hard",  "superpoint-lg", "method3"):   None,  # TODO
    ("easy_hard",  "superpoint-lg", "method4"):   None,  # TODO
    # hard threshold tuned for computational savings (adaptive)
    ("adaptive",   "superpoint-lg", "cosplace"):  None,  # TODO
    ("adaptive",   "superpoint-lg", "method2"):   None,  # TODO
    ("adaptive",   "superpoint-lg", "method3"):   None,  # TODO
    ("adaptive",   "superpoint-lg", "method4"):   None,  # TODO
    # decision boundary from a fitted logistic regression
    ("logistic",   "superpoint-lg", "cosplace"):  None,  # TODO
    ("logistic",   "superpoint-lg", "method2"):   None,  # TODO
    ("logistic",   "superpoint-lg", "method3"):   None,  # TODO
    ("logistic",   "superpoint-lg", "method4"):   None,  # TODO
}

THRESHOLD_TYPES = ["easy_only", "easy_hard", "adaptive", "logistic"]


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--preds-dir", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--out-dir", type=str, default=None, help="output directory of image matching results")

    # Choose matcher
    parser.add_argument(
        "--matcher",
        type=str,
        default="sift-lg",
        choices=available_models,
        help="choose your matcher",
    )
    parser.add_argument("--device", type=str, default=get_default_device(), choices=["cpu", "cuda"])
    parser.add_argument("--im-size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--num-preds", type=int, default=100, help="number of predictions to match")
    parser.add_argument("--start-query", type=int, default=-1, help="query to start from")
    parser.add_argument("--num-queries", type=int, default=-1, help="number of queries")

    # Extension 6.1: adaptive re-ranking skip
    parser.add_argument(
        "--threshold-type",
        type=str,
        default=None,
        choices=THRESHOLD_TYPES,
        help="(Extension 6.1) if set, skip re-ranking when top-1 inliers exceed the threshold",
    )
    parser.add_argument(
        "--vpr-method",
        type=str,
        default=None,
        help="(Extension 6.1) VPR method used to generate the predictions (required when --threshold-type is set)",
    )

    return parser.parse_args()


def get_threshold(threshold_type, matcher_name, vpr_method):
    """Return the inlier threshold for the given (type, matcher, vpr_method) combination."""
    key = (threshold_type, matcher_name, vpr_method)
    if key not in THRESHOLDS:
        raise ValueError(f"No threshold defined for key {key}. Add it to THRESHOLDS.")
    value = THRESHOLDS[key]
    if value is None:
        raise ValueError(f"Threshold for {key} is not filled in yet (None). Update THRESHOLDS.")
    return value


def main(args):
    device = args.device
    matcher_name = args.matcher
    img_size = args.im_size
    num_preds = args.num_preds
    matcher = get_matcher(matcher_name, device=device)
    preds_folder = args.preds_dir
    start_query = args.start_query
    num_queries = args.num_queries

    # --- Extension 6.1: resolve threshold if requested ---
    use_threshold = args.threshold_type is not None
    if use_threshold:
        if args.vpr_method is None:
            raise ValueError("--vpr-method must be specified when --threshold-type is set.")
        threshold = get_threshold(args.threshold_type, matcher_name, args.vpr_method)

    output_folder = Path(preds_folder + f"_{matcher_name}") if args.out_dir is None else Path(args.out_dir)
    output_folder.mkdir(exist_ok=True)

    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))
    start_query = start_query if start_query >= 0 else 0
    num_queries = num_queries if num_queries >= 0 else len(txt_files)

    for txt_file in tqdm(txt_files[start_query : start_query + num_queries]):
        q_num = Path(txt_file).stem
        out_file = output_folder.joinpath(f"{q_num}.torch")
        if out_file.exists():
            continue

        results = []
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=img_size)

        if use_threshold:
            # --- Extension 6.1: match only top-1 first, then decide ---

            img1 = matcher.load_image(pred_paths[0], resize=img_size)
            result_top1 = matcher(deepcopy(img0), img1)
            result_top1["all_desc0"] = result_top1["all_desc1"] = None
            results.append(result_top1)

            if result_top1["num_inliers"] > threshold:
                # Top-1 is confident: skip the remaining candidates.
                # Fill with num_inliers=0 so reranking.py gets the expected
                # num_preds entries and naturally ranks top-1 first.
                for _ in range(num_preds - 1):
                    results.append({"num_inliers": 0})
            else:
                # Top-1 is not confident: match all remaining candidates normally.
                for pred_path in pred_paths[1:num_preds]:
                    img1 = matcher.load_image(pred_path, resize=img_size)
                    result = matcher(deepcopy(img0), img1)
                    result["all_desc0"] = result["all_desc1"] = None
                    results.append(result)

        else:
            # --- Standard path: match all predictions (original behaviour) ---
            for pred_path in pred_paths[:num_preds]:
                img1 = matcher.load_image(pred_path, resize=img_size)
                result = matcher(deepcopy(img0), img1)
                result["all_desc0"] = result["all_desc1"] = None
                results.append(result)

        torch.save(results, out_file)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
