"""
best_r1/best_r1.py — Applica la threshold gia' calibrata in threshold.csv.

Soglia: num_inliers(top-1) < threshold -> rerank su top-20 (torch_folder).
Altrimenti skip: copia il .txt originale del retrieval (txt_folder).

Uso:
    python VPR-adaptive-re-ranking/best_r1/best_r1.py \
        --preds-dir preds/ --matcher superpoint-lg --output-dir out/
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import load_threshold_csv, run_scalar_method

_THRESHOLD_CSV = Path(__file__).parent / "threshold.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive reranking — best_r1")
    parser.add_argument("--preds-dir",  required=True)
    parser.add_argument("--matcher",    required=True)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--im-size",    type=int, default=512)
    parser.add_argument("--num-preds",  type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main(args):
    threshold = int(load_threshold_csv(_THRESHOLD_CSV)["threshold"])
    print(f"threshold (best_r1) = {threshold}")
    run_scalar_method(args.preds_dir, threshold, args.matcher, args.device,
                       args.im_size, args.num_preds, args.output_dir)


if __name__ == "__main__":
    main(parse_args())
