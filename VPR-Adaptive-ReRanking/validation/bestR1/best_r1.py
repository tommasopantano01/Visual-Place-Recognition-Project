"""
validation/best_r1 — SOLA VALIDATION: sceglie la soglia hard su num_inliers_top1
che MASSIMIZZA la R@1 adattiva sul dataset di validation (tie: saving max, poi
soglia minima). NON allena.

Catena: --val-csv (candidate-level) -> sweep (_sweep) -> selettore best_r1 ->
threshold.csv. Niente l2_distance, niente model.json.

OUTPUT in --out-dir (default: questa cartella):
  threshold.csv   numerico (threshold, r1_adaptive_pct, saving_pct), letto dal
                  deploy via load_threshold_csv -> ["threshold"].
  sweep.csv / selection.csv   per il report.

NB: best_r1 e' deployabile a se' (cartella propria), ma R@1_best e' anche il
riferimento interno usato da efficiency (target = 95% del guadagno di best_r1).

Uso:
    python VPR-Adaptive-ReRanking/validation/best_r1/best_r1.py --val-csv <dir-o-file.csv>
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))     # validation/  (per _sweep)
from _sweep import sweep_from_candidate, select_best_r1_threshold, save_outputs


def parse_args():
    p = argparse.ArgumentParser(description="Validation — best_r1 (max R@1 adattiva)")
    p.add_argument("--val-csv", required=True, help="candidate-level CSV di validation (file o dir)")
    p.add_argument("--out-dir", default=str(_HERE), help="dove scrivere gli output (default: questa cartella)")
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def main(args):
    sweep = sweep_from_candidate(args.val_csv, top_k=args.top_k)
    save_outputs(args.out_dir, sweep, select_best_r1_threshold(sweep))


if __name__ == "__main__":
    main(parse_args())
