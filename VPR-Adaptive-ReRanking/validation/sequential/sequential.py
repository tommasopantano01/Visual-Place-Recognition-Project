"""
validation/sequential — SOLA VALIDATION della policy sequenziale (cascata
1 -> 5 -> 10 -> 20 con tre cancelli).

NON allena. Legge i tre regressori gia' allenati dal training (model.json) e
cerca, sul dataset di validation scelto dall'utente, le soglie ottime
(tau1, tau5, tau10) che massimizzano la R@1 adattiva, con tie-break:
    go_to_5  = p1 > tau1
    go_to_10 = go_to_5  & (p5  > tau5)
    go_to_20 = go_to_10 & (p10 > tau10)
  budget scelto in {0,5,10,20}; costo {1,5,10,20}; R@1 = media(correct_budget).

model.json atteso (stesse chiavi del deploy methods/sequential.py):
    { "gate1": {regressore},     # P(continua 1->5)
      "gate5": {regressore},     # P(continua 5->10)
      "gate10": {regressore} }   # P(continua 10->20)
ogni {regressore} ha la struttura di regressor_to_dict (_common):
    feat_cols, scaler_mean, scaler_scale, coef, intercept, classes.
La validation e' AGNOSTICA alle feature: ogni gate usa le sue feat_cols, prese
dalle colonne costruite dal CSV. Feature disponibili: num_inliers_top1; per
top5 e top10: max_inliers, second_max_inliers, gap_inliers,
best_retrieval_rank, top1_is_best. (Single-feature o multi-feature: decide il
model.json.)

INPUT --val-csv: CSV candidate-level (dir o file) con colonne
    query_id, candidate_path, retrieval_rank, num_inliers, is_positive
(in alternativa un CSV gia' query-seq con le colonne pronte). Da qui calcola
correct_0/5/10/20 (winner = max num_inliers entro il budget, tie su
retrieval_rank piu' basso) e le feature progressive.

OUTPUT: threshold.csv PIATTO (una riga: tau1,tau5,tau10), leggibile da
load_threshold_csv del deploy. Scritto in questa cartella.

ATTENZIONE (deploy): methods/sequential.py oggi calcola per gate5/gate10 solo
la feature singola max(num_inliers). Per applicare un model.json multi-feature
(come il notebook) il deploy va aggiornato per ricostruire live le stesse
feature progressive. La validation invece le gestisce gia' tutte.

Uso:
    python VPR-Adaptive-ReRanking/validation/sequential/sequential.py \
        --val-csv <dir-o-file.csv>
"""
import argparse
import os
import sys
import csv
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

_MODEL_JSON_DEFAULT = _HERE / "model.json"
_GATES = ("gate1", "gate5", "gate10")
# feature progressive costruite (coprono i feature set continue_1/5/10 del notebook)
_PROG_BUDGETS = (5, 10)


# ── COSTRUZIONE FEATURE/LABEL DAL CANDIDATE-LEVEL (porting dal notebook) ──

def rerank_correct_with_budget(group, budget):
    """Correttezza dopo rerank dei soli candidati con retrieval_rank <= budget.
    Tie: vince num_inliers piu' alto; a parita', retrieval_rank piu' basso."""
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
    return {f"max_inliers_top{b}": max_inl, f"second_max_inliers_top{b}": second_max,
            f"gap_inliers_top{b}": max_inl - second_max,
            f"best_retrieval_rank_top{b}": best_rank,
            f"top1_is_best_top{b}": int(best_rank == 1)}


def candidate_to_query_seq_df(cdf):
    req = ["query_id", "candidate_path", "retrieval_rank", "num_inliers", "is_positive"]
    missing = [c for c in req if c not in cdf.columns]
    if missing:
        raise ValueError(f"Colonne candidate-level mancanti: {missing}")
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
        row = {
            "query_id":         str(qid),
            "num_inliers_top1": float(top1["num_inliers"]),
            "correct_0":        int(top1["is_positive"]),
            "correct_5":        rerank_correct_with_budget(g, 5),
            "correct_10":       rerank_correct_with_budget(g, 10),
            "correct_20":       rerank_correct_with_budget(g, 20),
        }
        for b in _PROG_BUDGETS:
            row.update(progressive_features_for_group(g, b))
        rows.append(row)
    return pd.DataFrame(rows)


def add_sequential_labels(df):
    """helps_20 / continue_5 / continue_10 (solo per diagnostica a schermo)."""
    df = df.copy()
    for c in ("correct_0", "correct_5", "correct_10", "correct_20"):
        df[c] = df[c].astype(int)
    df["helps_20"]    = ((df["correct_0"] == 0) & (df["correct_20"] == 1)).astype(int)
    df["continue_5"]  = ((df["correct_5"] == 0) &
                         (df[["correct_10", "correct_20"]].max(axis=1) == 1)).astype(int)
    df["continue_10"] = ((df["correct_10"] == 0) & (df["correct_20"] == 1)).astype(int)
    return df


def load_query_seq(val_csv):
    """Legge un candidate-level (file o dir) e costruisce il dataframe query-seq;
    se il CSV e' gia' query-seq lo usa direttamente."""
    if os.path.isdir(val_csv):
        files = sorted(glob(os.path.join(val_csv, "*.csv")))
        if not files:
            raise FileNotFoundError(f"Nessun CSV in {val_csv}")
        raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        raw = pd.read_csv(val_csv)

    cand_cols = {"query_id", "candidate_path", "retrieval_rank", "num_inliers", "is_positive"}
    seq_cols = {"query_id", "num_inliers_top1", "correct_0", "correct_5", "correct_10",
                "correct_20", "max_inliers_top5", "gap_inliers_top5",
                "best_retrieval_rank_top5", "top1_is_best_top5", "max_inliers_top10",
                "gap_inliers_top10", "best_retrieval_rank_top10", "top1_is_best_top10"}
    cols = set(raw.columns)
    if seq_cols.issubset(cols):
        df = raw.copy()
    elif cand_cols.issubset(cols):
        df = candidate_to_query_seq_df(raw)
    else:
        raise ValueError(f"Formato CSV non riconosciuto. Colonne trovate: {list(raw.columns)}")

    df = add_sequential_labels(df)
    feat_cols = (["num_inliers_top1"] +
                 [f"{n}_top{b}" for b in _PROG_BUDGETS
                  for n in ("max_inliers", "second_max_inliers", "gap_inliers",
                            "best_retrieval_rank", "top1_is_best")])
    keep = ["query_id", "correct_0", "correct_5", "correct_10", "correct_20"] + feat_cols
    df = df.dropna(subset=[c for c in keep if c in df.columns]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("Nessuna query valida dopo il dropna (controlla che ci siano >=10 candidati/query).")
    return df


# ── PROBABILITA' DEI CANCELLI ────────────────────────────────────────

def gate_proba(df, gate_dict):
    """p = P(continua) del gate, usando le sue feat_cols sul dataframe."""
    feat_cols = gate_dict["feat_cols"]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature richieste dal gate ma assenti nel CSV: {missing}")
    X = df[feat_cols].to_numpy(dtype=float)
    return predict_proba_pos(regressor_from_dict(gate_dict), X)


# ── GRID-SEARCH 3D (streaming argmax, fedele alla Cella 2) ───────────

def _policy_stats(p1, p5, p10, t1, t5, t10, c0, c5, c10, c20, k_full):
    go5  = p1 > t1
    go10 = go5 & (p5 > t5)
    go20 = go10 & (p10 > t10)
    budget = np.select([~go5, go5 & ~go10, go10 & ~go20, go20], [0, 5, 10, 20])
    cost   = np.select([budget == 0, budget == 5, budget == 10, budget == 20], [1, 5, 10, 20])
    corr   = np.select([budget == 0, budget == 5, budget == 10, budget == 20], [c0, c5, c10, c20])
    return float(corr.mean()), float(cost.mean()), budget


def grid_search(df, p1, p5, p10, taus, k_full=20):
    c0  = df["correct_0"].to_numpy(int)
    c5  = df["correct_5"].to_numpy(int)
    c10 = df["correct_10"].to_numpy(int)
    c20 = df["correct_20"].to_numpy(int)

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


# ── ORCHESTRAZIONE ───────────────────────────────────────────────────

def validate_and_save(out_dir, model_json_path, val_csv, vpr_model, matcher,
                      tau_step=0.02, k_full=20):
    import json
    os.makedirs(out_dir, exist_ok=True)
    with open(model_json_path) as f:
        model = json.load(f)
    miss = [g for g in _GATES if g not in model]
    if miss:
        raise ValueError(f"model.json: cancelli mancanti {miss} (servono {list(_GATES)})")

    print(f"Carico val-csv: {val_csv}")
    df = load_query_seq(val_csv)
    c0  = df["correct_0"].to_numpy(int)
    c5  = df["correct_5"].to_numpy(int)
    c10 = df["correct_10"].to_numpy(int)
    c20 = df["correct_20"].to_numpy(int)
    print(f"  N query: {len(df)}")
    print(f"  baseline R@1={c0.mean()*100:.2f}%  top5={c5.mean()*100:.2f}%  "
          f"top10={c10.mean()*100:.2f}%  top20(full)={c20.mean()*100:.2f}%")

    p1  = gate_proba(df, model["gate1"])
    p5  = gate_proba(df, model["gate5"])
    p10 = gate_proba(df, model["gate10"])

    taus = np.round(np.arange(0.0, 1.0 + tau_step / 2, tau_step), 4)
    print(f"  grid tau in [0,1] passo {tau_step}  ({len(taus)}^3 = {len(taus)**3} combinazioni)")
    best = grid_search(df, p1, p5, p10, taus, k_full=k_full)

    # statistiche della policy vincente
    r1, avg, budget = _policy_stats(p1, p5, p10, best["tau1"], best["tau5"], best["tau10"],
                                    c0, c5, c10, c20, k_full)
    dist = {b: float(np.mean(budget == b) * 100) for b in (0, 5, 10, 20)}
    print(f"\n  SCELTA: tau1={best['tau1']:.2f}  tau5={best['tau5']:.2f}  tau10={best['tau10']:.2f}")
    print(f"          R@1 adattiva={r1*100:.2f}%  match/query={avg:.2f}  "
          f"risparmio vs full={100*(1-avg/k_full):.1f}%")
    print(f"          stop  top1={dist[0]:.1f}%  top5={dist[5]:.1f}%  "
          f"top10={dist[10]:.1f}%  top20={dist[20]:.1f}%")

    thr_path = os.path.join(out_dir, f"threshold_{vpr_model}_{matcher}.csv")
    with open(thr_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau1", "tau5", "tau10"])
        w.writerow([f"{best['tau1']:.2f}", f"{best['tau5']:.2f}", f"{best['tau10']:.2f}"])
    print(f"  -> {thr_path}")
    return thr_path


def parse_args():
    p = argparse.ArgumentParser(description="Validation — sequential (grid 3D tau1/tau5/tau10)")
    p.add_argument("--val-csv", required=True,
                   help="CSV candidate-level di validation (dir o file) scelto dall'utente")
    p.add_argument("--model-json", default=str(_MODEL_JSON_DEFAULT),
                   help=f"model.json del training (default: {_MODEL_JSON_DEFAULT})")
    p.add_argument("--out-dir", default=str(_HERE),
                   help="dove scrivere threshold.csv (default: questa cartella)")
    p.add_argument("--tau-step", type=float, default=0.02, help="passo griglia tau (default 0.02)")
    p.add_argument("--k-full", type=int, default=20, help="budget massimo (default 20)")
    p.add_argument("--model", required=True, help="cosplace or megaloc")
    p.add_argument("--matcher", required=True, help="superpoint-lg or loftr")
    return p.parse_args()


def main(args):
    validate_and_save(args.out_dir, args.model_json, args.val_csv, args.model, args.matcher,
                      tau_step=args.tau_step, k_full=args.k_full)


if __name__ == "__main__":
    main(parse_args())
