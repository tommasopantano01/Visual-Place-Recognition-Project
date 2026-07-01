"""
validation/logistic_cost_sensitive — SOLA VALIDATION del metodo cost-sensitive.

Policy:  S = P(helps | num_inliers_top1) - alpha * P(hurts | num_inliers_top1)
         rerank se  S > tau.
Cerca (alpha*, tau*) che massimizzano la R@1 adattiva sul dataset di
validation, con tie-break "meno query rerankate". 1:1 con la Cella 2 del
notebook P(helps)/P(hurts) (grid: alpha in [0,5] passo 0.1, tau in [-1,1]
passo 0.01; argmax R@1; a parita' preferisce rerankare meno).

NON allena. I due regressori (help, hurts), single-feature su num_inliers_top1,
arrivano gia' allenati dal training. model.json atteso (UN solo feature set):
    { "feature_sets": { "<nome>": {
          "feat_cols": ["num_inliers_top1"],
          "regressors": { "help": {...}, "hurts": {...} } } },
      "metadata": { "su_k": 20, ... } }
(stessa struttura prodotta da regressor_to_dict in _common; "hurts" con la 's').

Riusa l'engine condiviso _su_validation, identico a su.py / logistic_help.py,
solo con criteria=("P(help)-aP(hurts)",) e le griglie di default (= Cella 2).

INPUT: --val-csv e' un CSV candidate-level (dir o file singolo) con le colonne
standard (query_id, l2_distance, retrieval_rank, num_inliers, rerank_rank_topK,
is_positive). Il loader condiviso (load_query_level) richiede l2_distance e
>=20 candidati/query anche se il cost-sensitive usa solo num_inliers_top1.
L'utente cambia il set con --val-csv. Scrive threshold_<model>_<matcher>.csv in
questa cartella. --model/--matcher identificano la coppia (retrieval, image
matching) su cui e' calibrata la soglia.

Uso:
    python VPR-Adaptive-ReRanking/validation/logistic_cost_sensitive/logistic_cost_sensitive.py \
        --val-csv <dir-o-file.csv> --model cosplace --matcher superpoint-lg
    # pesi/output in posizioni non standard:
    python VPR-Adaptive-ReRanking/validation/logistic_cost_sensitive/logistic_cost_sensitive.py \
        --val-csv <...> --model-json <path/model.json> --out-dir <path/> \
        --model cosplace --matcher superpoint-lg
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))            # validation/  (per _su_validation)
sys.path.append(str(_HERE.parent.parent))     # VPR-Adaptive-ReRanking/  (per _common)
from _su_validation import validate_and_save, ALPHAS_GRID, TAUS_GRID

CRITERIA = ("P(help)-aP(hurts)",)
_DEFAULT_MODEL = _HERE / "model.json"   # pesi accanto al validation, in questa cartella


def parse_args():
    p = argparse.ArgumentParser(
        description="Validation — logistic_cost_sensitive (P(help)-alpha*P(hurts) > tau)")
    p.add_argument("--val-csv", required=True,
                   help="CSV candidate-level di validation scelto dall'utente (dir o file)")
    p.add_argument("--model-json", default=str(_DEFAULT_MODEL),
                   help=f"model.json del training (default: {_DEFAULT_MODEL})")
    p.add_argument("--out-dir", default=str(_HERE),
                   help="dove scrivere threshold.csv (default: questa cartella)")
    p.add_argument("--model", required=True, help="cosplace or megaloc")
    p.add_argument("--matcher", required=True, help="superpoint-lg or loftr")
    return p.parse_args()


def main(args):
    # alpha/tau di default = griglie della Cella 2 (alpha in [0,5], tau in [-1,1])
    validate_and_save(args.out_dir, args.model_json, args.val_csv, args.model, args.matcher,
                      criteria=CRITERIA, taus=TAUS_GRID, alphas=ALPHAS_GRID)


if __name__ == "__main__":
    main(parse_args())
