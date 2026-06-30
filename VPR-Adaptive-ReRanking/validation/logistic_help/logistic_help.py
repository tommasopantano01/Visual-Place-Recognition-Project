"""
validation/logistic_help — SOLA VALIDATION del metodo logistic_help.

Modella P(helps | num_inliers_top1) con un regressore logistico (allenato a
parte, nel training) e cerca la soglia tau che massimizza la R@1 adattiva.
Policy: rerank se P(helps) > tau. UN solo criterio, una sola soglia.

Riusa l'engine di validazione di _su_validation (stessa matematica: grid su tau,
argmax R@1 adattiva, tie-break meno reranking) ma con:
  - criteria = ("P(help)",)      un solo criterio
  - griglia tau in [0,1]         (P e' una probabilita', non serve [-1,1])
  - feature num_inliers_top1     (il loader la fornisce grezza, non negata)

NON allena: legge il model.json gia' prodotto dal training
(training/logistic_help/model.json di default). Scrive threshold.csv qui, con la
STESSA struttura dei metodi SU (feature_sets -> criteria -> P(help)).

Il model.json deve avere un regressore 'help' con feat_cols=['num_inliers_top1'].

Uso:
    python VPR-Adaptive-ReRanking/validation/logistic_help/logistic_help.py \
        --val-csv <dir-o-file.csv>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))            # validation/  (per _su_validation)
sys.path.append(str(_HERE.parent.parent))     # VPR-Adaptive-ReRanking/  (per _common)
from _su_validation import validate_and_save

# griglia tau in [0,1] passo 0.01, come la Cella 2 del notebook logistic_help
TAUS_GRID_PROB = np.round(np.arange(0.0, 1.001, 0.01), 2)
CRITERIA = ("P(help)",)

_DEFAULT_MODEL = _HERE.parent.parent / "training" / "logistic_help" / "model.json"


def parse_args():
    p = argparse.ArgumentParser(description="Validation — logistic_help (P(help)>tau)")
    p.add_argument("--val-csv", required=True,
                   help="CSV candidate-level di validation scelto dall'utente (dir o file)")
    p.add_argument("--model-json", default=str(_DEFAULT_MODEL),
                   help=f"model.json del training (default: {_DEFAULT_MODEL})")
    p.add_argument("--out-dir", default=str(_HERE),
                   help="dove scrivere threshold.csv (default: questa cartella)")
    return p.parse_args()


def main(args):
    validate_and_save(args.out_dir, args.model_json, args.val_csv,
                      criteria=CRITERIA, taus=TAUS_GRID_PROB)


if __name__ == "__main__":
    main(parse_args())
