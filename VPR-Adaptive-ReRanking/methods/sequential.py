"""
sequential/sequential_calibrate.py — Calibra i tre cancelli della cascata
1->5->10->20. Scrive threshold.csv (tau1,tau5,tau10) + model.json
(gate1, gate5, gate10).

NOTA — gate1 ancora provvisorio: il report indica che il primo cancello
"e' approssimato riusando il modello P(helps|I_1) gia' allenato", ma il
target esatto di training non e' confermato (segnalato da Davide: va
chiarito con Luca). Qui gate1 e' allenato come un logistic_help standard
su num_inliers_top1 — stessa logica di logistic_help_calibrate.py — solo
per coerenza di formato col resto della cascata. Da aggiornare quando
arriva la conferma.

gate5/gate10 invece sono allenati secondo la definizione del report:
  y_cont5  = 1{correct_5=0  AND max(correct_10, correct_20)=1}
  y_cont10 = 1{correct_10=0 AND correct_20=1}
su feature = max(num_inliers) tra i candidati visti fino a quel budget.

I tau vengono calibrati IN CASCATA su validation: tau5 e' scelto solo
sulle query che gate1 manda avanti (con tau1 appena calibrato), tau10
solo su quelle che gate5 manda avanti — replica il comportamento reale
della pipeline invece di calibrare ogni soglia in isolamento.

Uso:
    python VPR-adaptive-re-ranking/sequential/sequential_calibrate.py \
        --train-csv train.csv --val-csv val.csv
"""
import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _calibrate_common import (
    load_query_level, compute_metrics, fit_logistic, regressor_to_dict,
    save_threshold_csv, save_model_json,
)

_THRESHOLD_CSV = Path(__file__).parent / "threshold.csv"
_MODEL_JSON    = Path(__file__).parent / "model.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Calibra la cascata sequenziale (gate1/gate5/gate10)")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv",   required=True)
    parser.add_argument("--tau-step",  type=float, default=0.02)
    return parser.parse_args()


def best_tau_for(p_val, correct_low, correct_high, tau_values):
    """Tau che massimizza l'R@1 adattivo: sotto soglia -> correct_low, sopra -> correct_high."""
    best_r1, best_tau = -1.0, tau_values[0]
    for tau in tau_values:
        cont = p_val > tau
        m = compute_metrics(cont, correct_low, correct_high)
        if m["adaptive_R@1"] > best_r1:
            best_r1, best_tau = m["adaptive_R@1"], tau
    return best_tau, best_r1


def main(args):
    train_df = load_query_level(args.train_csv, budgets=(5, 10))
    val_df   = load_query_level(args.val_csv,   budgets=(5, 10))

    needed = ["correct_5", "correct_10", "max_inliers_top5", "max_inliers_top10"]
    train_df = train_df.dropna(subset=needed)
    val_df   = val_df.dropna(subset=needed)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} (con budget intermedi disponibili)")

    tau_values = np.round(np.arange(0.0, 1.0 + args.tau_step, args.tau_step), 4)

    # ===================== GATE 1 (1->5), provvisorio =====================
    clf1 = fit_logistic(train_df[["num_inliers_top1"]].values, train_df["helps_20"].values)
    p1_val = clf1.predict_proba(val_df[["num_inliers_top1"]].values)[:, 1]

    tau1, r1_1 = best_tau_for(p1_val, val_df["correct_0"].values, val_df["correct_20"].values, tau_values)
    print(f"gate1: tau1={tau1}  R@1={r1_1:.4f}  [provvisorio]")

    continue_5_mask = p1_val > tau1
    val_after_gate1 = val_df[continue_5_mask]
    print(f"  -> {len(val_after_gate1)}/{len(val_df)} query continuano oltre il top-1")

    # ===================== GATE 2 (5->10) =====================
    y_cont5_train = ((train_df["correct_5"] == 0) &
                      (np.maximum(train_df["correct_10"], train_df["correct_20"]) == 1)).astype(int)
    clf5 = fit_logistic(train_df[["max_inliers_top5"]].values, y_cont5_train.values)
    p5_val = clf5.predict_proba(val_after_gate1[["max_inliers_top5"]].values)[:, 1]

    tau5, r1_5 = best_tau_for(p5_val, val_after_gate1["correct_5"].values,
                               val_after_gate1["correct_20"].values, tau_values)
    print(f"gate5: tau5={tau5}  R@1(subset)={r1_5:.4f}")

    continue_10_mask = p5_val > tau5
    val_after_gate2 = val_after_gate1[continue_10_mask]
    print(f"  -> {len(val_after_gate2)}/{len(val_after_gate1)} query continuano oltre il top-5")

    # ===================== GATE 3 (10->20) =====================
    y_cont10_train = ((train_df["correct_10"] == 0) & (train_df["correct_20"] == 1)).astype(int)
    clf10 = fit_logistic(train_df[["max_inliers_top10"]].values, y_cont10_train.values)
    p10_val = clf10.predict_proba(val_after_gate2[["max_inliers_top10"]].values)[:, 1]

    tau10, r1_10 = best_tau_for(p10_val, val_after_gate2["correct_10"].values,
                                  val_after_gate2["correct_20"].values, tau_values)
    print(f"gate10: tau10={tau10}  R@1(subset)={r1_10:.4f}")

    save_model_json(_MODEL_JSON, {
        "gate1":  regressor_to_dict(clf1,  ["inliers"]),
        "gate5":  regressor_to_dict(clf5,  ["inliers"]),
        "gate10": regressor_to_dict(clf10, ["inliers"]),
    })
    save_threshold_csv(_THRESHOLD_CSV, tau1=tau1, tau5=tau5, tau10=tau10)


if __name__ == "__main__":
    main(parse_args())
