"""
validation/youden — SOLA VALIDATION: sceglie la soglia hard su num_inliers_top1
col criterio di Youden (argmax TPR_hard - FPR_easy) sul dataset di validation.
NON allena (i metodi scalari non hanno regressore: la soglia E' la scelta).

Catena: --val-csv (candidate-level) -> sweep delle soglie (_sweep) ->
selettore youden -> threshold.csv. Niente l2_distance, niente model.json.

OUTPUT in --out-dir (default: questa cartella):
  threshold.csv   PIATTO e numerico (threshold, r1_adaptive_pct, saving_pct);
                  il deploy lo legge con load_threshold_csv -> ["threshold"].
  sweep.csv       sweep completo (per il report).
  selection.csv   riga scelta con la colonna 'method' (per il report).

--val-csv accetta un file o una directory di candidate-level CSV (colonne
query_id, retrieval_rank, num_inliers, rerank_rank_topK, is_positive, K).

Uso:
    python VPR-Adaptive-ReRanking/validation/youden/youden.py --val-csv <dir-o-file.csv>
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))     # validation/  (per _sweep)
from _sweep import sweep_from_candidate, select_youden_threshold, save_outputs


def parse_args():
    p = argparse.ArgumentParser(description="Validation — youden (soglia su num_inliers_top1)")
    p.add_argument("--val-csv", required=True, help="candidate-level CSV di validation (file o dir)")
    p.add_argument("--out-dir", default=str(HERE), help="dove scrivere gli output (default: methods folder)")
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def main(args):
    sweep = sweep_from_candidate(args.val_csv, top_k=args.top_k)
    save_outputs(args.out_dir, sweep, select_youden_threshold(sweep))


if __name__ == "__main__":
    main(parse_args())
