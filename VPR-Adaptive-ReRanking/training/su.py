"""
training/su — SOLO TRAINING del metodo SU (feature set ['SU']).

Allena i 3 regressori (hard/help/hurts) sui NOSTRI dataset di training e salva
model.json in questa cartella. La ricerca delle soglie e' un passo separato
(validation/su/su.py), eseguito dall'utente su un dataset a sua scelta.

Uso:
    python VPR-Adaptive-ReRanking/training/su/train_su.py \
        --train-csv <dir-o-file.csv>
"""
import argparse
import sys
from pathlib import Path

# _su_train sta in training/; _common (con le funzioni SU) nella radice
_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent))            # training/
sys.path.append(str(_HERE.parent.parent))     # VPR-Adaptive-ReRanking/
from _su_train import train_and_save, SU_K_DEFAULT, SU_ALPHA_DEFAULT

FEATURE_SET = "SU"
FEAT_COLS   = ["SU"]


def parse_args():
    p = argparse.ArgumentParser(description="Training — SU (solo retrieval)")
    p.add_argument("--train-csv", required=True,
                   help="CSV candidate-level di training (dir o file)")
    p.add_argument("--su-k",     type=int,   default=SU_K_DEFAULT)
    p.add_argument("--su-alpha", type=float, default=SU_ALPHA_DEFAULT)
    p.add_argument("--out-dir",  default=str(_HERE),
                   help="dove scrivere model.json (default: questa cartella)")
    return p.parse_args()


def main(args):
    train_and_save(FEATURE_SET, FEAT_COLS, args.out_dir,
                   train_csv=args.train_csv, k=args.su_k, alpha=args.su_alpha)


if __name__ == "__main__":
    main(parse_args())
