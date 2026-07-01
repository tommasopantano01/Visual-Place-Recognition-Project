"""
validation/efficiency — SOLA VALIDATION: sceglie la soglia hard su
num_inliers_top1 piu' EFFICIENTE che trattiene >= 95% del guadagno di R@1 del
miglior adattivo (target = pre_r1 + 0.95*(best_adaptive_r1 - pre_r1)), poi
saving massimo. best_adaptive_r1 e' il passo interno (= R@1 di best_r1).
NON allena.

Catena: --val-csv (candidate-level) -> sweep (_sweep) -> selettore eff95 ->
threshold.csv. Niente l2_distance, niente model.json.

OUTPUT in --out-dir (default: questa cartella):
  threshold_<model>_<matcher>.csv   numerico (threshold, r1_adaptive_pct,
                  saving_pct), letto dal deploy via load_threshold_csv -> ["threshold"].
  sweep.csv / selection.csv   per il report.

--retention cambia la frazione trattenuta (default 0.95).

Uso:
    python VPR-Adaptive-ReRanking/validation/efficiency/efficiency.py --val-csv <dir-o-file.csv> \
        --model cosplace --matcher superpoint-lg
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))     # validation/  (per _sweep)
from _sweep import sweep_from_candidate, select_eff95_threshold, save_outputs


def parse_args():
    p = argparse.ArgumentParser(description="Validation — efficiency (T_95: 95% del guadagno, max saving)")
    p.add_argument("--val-csv", required=True, help="candidate-level CSV di validation (file o dir)")
    p.add_argument("--out-dir", default=str(_HERE), help="dove scrivere gli output (default: questa cartella)")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--retention", type=float, default=0.95, help="frazione del guadagno R@1 da trattenere")
    p.add_argument("--model", required=True, help="cosplace or megaloc")
    p.add_argument("--matcher", required=True, help="superpoint-lg or loftr")
    return p.parse_args()


def main(args):
    sweep = sweep_from_candidate(args.val_csv, top_k=args.top_k)
    save_outputs(args.out_dir, sweep, select_eff95_threshold(sweep, retention=args.retention),
                 args.model, args.matcher)


if __name__ == "__main__":
    main(parse_args())
