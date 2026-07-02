"""
validation/sequential — SOLA VALIDATION della policy sequenziale (cascata
1 -> 5 -> 10 -> 20).

NON allena. Carica i TRE regressori gia' allenati (un JSON per gate) da una
INPUT DIR, scelti in base a retrieval model e matcher, e cerca sul dataset di
validation le soglie (tau1, tau5, tau10) che massimizzano la R@1 adattiva.

MODELLI IN INPUT (--models-dir): tre file, uno per gate:
    seq_model_continue_1_phelps_<model>_svox_train_<matcher>.json   (gate1)
    seq_model_continue_5_<model>_svox_train_<matcher>.json          (gate5)
    seq_model_continue_10_<model>_svox_train_<matcher>.json         (gate10)
La ricerca e' tollerante: glob *continue_{1,5,10}*<model>*<matcher>*.json.
Ogni JSON ha il formato di regressor_to_dict (_common): feat_cols anonime
("feature_0"...), scaler_mean/scale, coef, intercept, classes. Le feature sono
passate POSIZIONALMENTE nell'ordine esatto sotto (i nomi nel JSON sono anonimi).

FEATURE :
  gate1  (1):  [num_inliers_top1]
  gate5  (6):  [num_inliers_top1, max_inliers_top5, second_max_inliers_top5,
                gap_inliers_top5, best_retrieval_rank_top5, top1_is_best_top5]
  gate10 (10): [num_inliers_top1, max_inliers_top5, gap_inliers_top5,
                best_retrieval_rank_top5, top1_is_best_top5,
                max_inliers_top10, second_max_inliers_top10, gap_inliers_top10,
                best_retrieval_rank_top10, top1_is_best_top10]
  NB asimmetria: gate10 NON include second_max_inliers_top5 (solo gap per top5),
  ma include second_max_inliers_top10. Fedele al notebook.

TARGET:
  gate1:  helps_20   = (correct_0==0) & (correct_20==1)
  gate5:  continue_5 = (correct_5==0) & (max(correct_10,correct_20)==1)
  gate10: continue_10= (correct_10==0) & (correct_20==1)

POLICY:
  stop retrieval se p1<=tau1; stop top5 se p1>tau1 & p5<=tau5;
  stop top10 se p5>tau5 & p10<=tau10; full top20 se p10>tau10.
  Costo: budget 0 costa 1 (serve I1). Tie-break: R@1 max, poi avg_matches min.

INPUT --val-csv: CSV candidate-level (dir o file) con
    query_id, candidate_path, retrieval_rank, num_inliers, is_positive
oppure un CSV gia' query-seq con le colonne feature pronte.

OUTPUT: threshold_<model>_<matcher>.csv.

Uso:
    python VPR-Adaptive-ReRanking/validation/sequential/sequential.py \
        --val-csv <dir-o-file.csv> --models-dir <dir coi 3 json> \
        --model cosplace --matcher sp-lg
"""
import argparse
import csv
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent.parent))     # VPR-Adaptive-ReRanking/  (per _common)
from _common import regressor_from_dict, predict_proba_pos

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **k):
        return x

# ── ORDINE ESATTO FEATURE ─────────────────────────────────
FEATURES_CONTINUE_1 = ["num_inliers_top1"]
FEATURES_CONTINUE_5 = [
    "num_inliers_top1", "max_inliers_top5", "second_max_inliers_top5",
    "gap_inliers_top5", "best_retrieval_rank_top5", "top1_is_best_top5",
]
FEATURES_CONTINUE_10 = [
    "num_inliers_top1", "max_inliers_top5", "gap_inliers_top5",
    "best_retrieval_rank_top5", "top1_is_best_top5",
    "max_inliers_top10", "second_max_inliers_top10", "gap_inliers_top10",
    "best_retrieval_rank_top10", "top1_is_best_top10",
]
REQUIRED_CAND_COLS = ["query_id", "candidate_path", "retrieval_rank",
                      "num_inliers", "is_positive"]
_PROG_BUDGETS = (5, 10, 20)


# ── candidate-level ────────

def rerank_correct_with_budget(group, budget):
    sub = group[group["retrieval_rank"] <= budget]
    if len(sub) == 0:
        return np.nan
    winner = sub.sort_values(["num_inliers", "retrieval_rank"],
                             ascending=[False, True]).iloc[0]
    return int(winner["is_positive"])


def progressive_features_for_group(group, b):
    sub = group[group["retrieval_rank"] <= b]
    if len(sub) == 0:
        return {f"max_inliers_top{b}": np.nan, f"second_max_inliers_top{b}": np.nan,
                f"gap_inliers_top{b}": np.nan, f"best_retrieval_rank_top{b}": np.nan,
                f"top1_is_best_top{b}": np.nan}
    ss = sub.sort_values(["num_inliers", "retrieval_rank"], ascending=[False, True])
    best = ss.iloc[0]
    max_inl = float(best["num_inliers"])
    best_rank = int(best["retrieval_rank"])
    second_max = float(ss.iloc[1]["num_inliers"]) if len(ss) >= 2 else 0.0
    return {f"max_inliers_top{b}": max_inl,
            f"second_max_inliers_top{b}": second_max,
            f"gap_inliers_top{b}": max_inl - second_max,
            f"best_retrieval_rank_top{b}": best_rank,
            f"top1_is_best_top{b}": int(best_rank == 1)}


def candidate_to_query_seq_df(cdf):
    missing = [c for c in REQUIRED_CAND_COLS if c not in cdf.columns]
    if missing:
        raise ValueError(f"Mancano colonne candidate-level: {missing}")
    cdf = cdf.copy()
    cdf["query_id"] = cdf["query_id"].astype(str)
    cdf["retrieval_rank"] = cdf["retrieval_rank"].astype(int)
    cdf["num_inliers"] = cdf["num_inliers"].astype(float)
    cdf["is_positive"] = cdf["is_positive"].astype(int)

    rows = []
    for qid, g in cdf.groupby("query_id", sort=False):
        top1 = g[g["retrieval_rank"] == 1]
        if len(top1) == 0:
            continue
        top1 = top1.iloc[0]
        row = {"query_id": str(qid),
               "num_inliers_top1": float(top1["num_inliers"]),
               "correct_0": int(top1["is_positive"]),
               "correct_5": rerank_correct_with_budget(g, 5),
               "correct_10": rerank_correct_with_budget(g, 10),
               "correct_20": rerank_correct_with_budget(g, 20)}
        for b in _PROG_BUDGETS:
            row.update(progressive_features_for_group(g, b))
        rows.append(row)
    return pd.DataFrame(rows)


def add_sequential_labels(df):
    df = df.copy()
    for c in ("correct_0", "correct_5", "correct_10", "correct_20"):
        df[c] = df[c].astype(int)
    df["helps_20"]    = ((df["correct_0"] == 0) & (df["correct_20"] == 1)).astype(int)
    df["continue_5"]  = ((df["correct_5"] == 0) &
                         (df[["correct_10", "correct_20"]].max(axis=1) == 1)).astype(int)
    df["continue_10"] = ((df["correct_10"] == 0) & (df["correct_20"] == 1)).astype(int)
    return df


def load_query_seq(val_csv):
    if os.path.isdir(val_csv):
        files = sorted(glob(os.path.join(val_csv, "*.csv")))
        if not files:
            raise FileNotFoundError(f"Nessun CSV in {val_csv}")
        raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        raw = pd.read_csv(val_csv)

    seq_needed = set(["query_id", "correct_0", "correct_5", "correct_10",
                      "correct_20"] + FEATURES_CONTINUE_10)
    cand_cols = set(REQUIRED_CAND_COLS)
    cols = set(raw.columns)
    if seq_needed.issubset(cols):
        df = raw.copy()
    elif cand_cols.issubset(cols):
        df = candidate_to_query_seq_df(raw)
    else:
        raise ValueError(f"Formato CSV non riconosciuto. Colonne trovate: {list(raw.columns)}")

    df = add_sequential_labels(df)
    all_feats = sorted(set(FEATURES_CONTINUE_5) | set(FEATURES_CONTINUE_10))
    keep = ["query_id", "correct_0", "correct_5", "correct_10", "correct_20"] + all_feats
    df = df.dropna(subset=[c for c in keep if c in df.columns]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("Nessuna query valida dopo il dropna (servono >=10 candidati/query).")
    for c in ("correct_0", "correct_5", "correct_10", "correct_20"):
        df[c] = df[c].astype(int)
    return df


# ── caricamento dei 3 JSON dalla input dir ───────────────────────────

def _find_gate_json(models_dir, gate_num, model, matcher):
    """Cerca il JSON del gate nella input dir, tollerante al naming."""
    pats = [
        os.path.join(models_dir, f"*continue_{gate_num}_*{model}*{matcher}*.json"),
        os.path.join(models_dir, f"*continue_{gate_num}*{model}*{matcher}*.json"),
    ]
    for pat in pats:
        hits = sorted(glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Nessun JSON per gate{gate_num} in {models_dir} "
        f"(model='{model}', matcher='{matcher}'). Pattern provati: {pats}")


def load_gate_models(models_dir, model, matcher):
    g1 = _find_gate_json(models_dir, 1, model, matcher)
    g5 = _find_gate_json(models_dir, 5, model, matcher)
    g10 = _find_gate_json(models_dir, 10, model, matcher)
    print(f"  gate1  <- {os.path.basename(g1)}")
    print(f"  gate5  <- {os.path.basename(g5)}")
    print(f"  gate10 <- {os.path.basename(g10)}")
    with open(g1) as f:  m1 = json.load(f)
    with open(g5) as f:  m5 = json.load(f)
    with open(g10) as f: m10 = json.load(f)
    # controllo n feature atteso (fedelta' al notebook)
    for name, m, exp in (("gate1", m1, 1), ("gate5", m5, 6), ("gate10", m10, 10)):
        n = len(m["feat_cols"])
        if n != exp:
            raise ValueError(f"{name}: atteso {exp} feature, il JSON ne ha {n} "
                             f"(feat_cols={m['feat_cols']}).")
    return m1, m5, m10


def gate_proba(df, gate_model, feature_names):
    """Costruisce X POSIZIONALMENTE nell'ordine feature_names (i nomi nel JSON
    sono anonimi feature_0..N) e applica il regressore."""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Feature richieste ma assenti nel CSV: {missing}")
    X = df[feature_names].to_numpy(dtype=float)
    if X.shape[1] != len(gate_model["feat_cols"]):
        raise ValueError(f"Mismatch n feature: costruite {X.shape[1]}, "
                         f"il modello ne vuole {len(gate_model['feat_cols'])}.")
    return predict_proba_pos(regressor_from_dict(gate_model), X)


# ── grid-search ──────────────────────────────

def _policy_stats(p1, p5, p10, t1, t5, t10, c0, c5, c10, c20):
    go5  = p1 > t1
    go10 = go5 & (p5 > t5)
    go20 = go10 & (p10 > t10)
    budget = np.select([~go5, go5 & ~go10, go10 & ~go20, go20], [0, 5, 10, 20])
    cost   = np.select([budget == 0, budget == 5, budget == 10, budget == 20], [1, 5, 10, 20])
    corr   = np.select([budget == 0, budget == 5, budget == 10, budget == 20], [c0, c5, c10, c20])
    return float(corr.mean()), float(cost.mean()), budget


def grid_search(p1, p5, p10, c0, c5, c10, c20, taus):
    best = {"r1": -np.inf, "avg": np.inf, "tau1": float(taus[0]),
            "tau5": float(taus[0]), "tau10": float(taus[0])}
    for t1 in tqdm(taus, desc="grid tau1"):
        go5 = p1 > t1
        for t5 in taus:
            go10 = go5 & (p5 > t5)
            for t10 in taus:
                go20 = go10 & (p10 > t10)
                budget = np.select([~go5, go5 & ~go10, go10 & ~go20, go20], [0, 5, 10, 20])
                cost   = np.select([budget == 0, budget == 5, budget == 10, budget == 20], [1, 5, 10, 20])
                corr   = np.select([budget == 0, budget == 5, budget == 10, budget == 20], [c0, c5, c10, c20])
                r1  = float(corr.mean())
                avg = float(cost.mean())
                if r1 > best["r1"] or (r1 == best["r1"] and avg < best["avg"]):
                    best.update(r1=r1, avg=avg, tau1=float(t1), tau5=float(t5), tau10=float(t10))
    return best


# ── orchestrazione ───────────────────────────────────────────────────

def validate_and_save(out_dir, models_dir, val_csv, model, matcher,
                      tau_step=0.02, k_full=20):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Modelli (input dir): {models_dir}  [model={model}, matcher={matcher}]")
    m1, m5, m10 = load_gate_models(models_dir, model, matcher)

    print(f"Carico val-csv: {val_csv}")
    df = load_query_seq(val_csv)
    c0  = df["correct_0"].to_numpy(int)
    c5  = df["correct_5"].to_numpy(int)
    c10 = df["correct_10"].to_numpy(int)
    c20 = df["correct_20"].to_numpy(int)
    print(f"  N query: {len(df)}")
    print(f"  baseline R@1={c0.mean()*100:.2f}%  top5={c5.mean()*100:.2f}%  "
          f"top10={c10.mean()*100:.2f}%  top20(full)={c20.mean()*100:.2f}%")

    p1  = gate_proba(df, m1,  FEATURES_CONTINUE_1)
    p5  = gate_proba(df, m5,  FEATURES_CONTINUE_5)
    p10 = gate_proba(df, m10, FEATURES_CONTINUE_10)

    taus = np.round(np.arange(0.0, 1.0 + tau_step / 2, tau_step), 4)
    print(f"  grid tau in [0,1] passo {tau_step}  ({len(taus)}^3 = {len(taus)**3} combinazioni)")
    best = grid_search(p1, p5, p10, c0, c5, c10, c20, taus)

    r1, avg, budget = _policy_stats(p1, p5, p10, best["tau1"], best["tau5"], best["tau10"],
                                    c0, c5, c10, c20)
    dist = {b: float(np.mean(budget == b) * 100) for b in (0, 5, 10, 20)}
    print(f"\n  SCELTA: tau1={best['tau1']:.2f}  tau5={best['tau5']:.2f}  tau10={best['tau10']:.2f}")
    print(f"          R@1 adattiva={r1*100:.2f}%  match/query={avg:.2f}  "
          f"risparmio vs full={100*(1-avg/k_full):.1f}%")
    print(f"          stop  top1={dist[0]:.1f}%  top5={dist[5]:.1f}%  "
          f"top10={dist[10]:.1f}%  top20={dist[20]:.1f}%")

    thr_path = os.path.join(out_dir, f"threshold_{model}_{matcher}.csv")
    with open(thr_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau1", "tau5", "tau10"])
        w.writerow([f"{best['tau1']:.2f}", f"{best['tau5']:.2f}", f"{best['tau10']:.2f}"])
    print(f"  -> {thr_path}")
    return thr_path


def parse_args():
    p = argparse.ArgumentParser(description="Validation — sequential (multi-feature, fedele al notebook)")
    p.add_argument("--val-csv", required=True,
                   help="CSV candidate-level di validation (dir o file) scelto dall'utente")
    p.add_argument("--models-dir", required=True,
                   help="cartella coi 3 JSON dei gate (continue_1/5/10) per model+matcher")
    p.add_argument("--model", required=True, help="retrieval model (es. cosplace, megaloc)")
    p.add_argument("--matcher", required=True, help="image matcher (es. sp-lg, loftr)")
    p.add_argument("--out-dir", default=str(_HERE),
                   help="dove scrivere threshold.csv (default: questa cartella)")
    p.add_argument("--tau-step", type=float, default=0.02, help="passo griglia tau (default 0.02)")
    p.add_argument("--k-full", type=int, default=20, help="budget massimo (default 20)")
    return p.parse_args()


def main(args):
    validate_and_save(args.out_dir, args.models_dir, args.val_csv, args.model, args.matcher,
                      tau_step=args.tau_step, k_full=args.k_full)


if __name__ == "__main__":
    main(parse_args())
