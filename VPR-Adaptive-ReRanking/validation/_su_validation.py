"""
_su_validation.py — Engine di SOLA VALIDATION per i metodi SU.

NON allena nulla: legge un model.json gia' prodotto dal training (cartella
training/) e cerca, sul dataset di validation indicato dall'utente, le soglie
ottime per ciascun criterio massimizzando la R@1 adattiva.

Le funzioni base (load_query_level, regressor_from_dict, predict_proba_pos,
clean_scores, costanti) vivono in _common.py, blocco "AGGIUNTE SU".

Grid-search fedele 1:1 alla Cella 3:
  - griglie FISSE: tau in [-1,1] passo 0.01, alpha in [0,5] passo 0.1
  - tie-break: a parita' di R@1, preferisce rerankare MENO query
  - R@1 in percentuale
Criteri con regressore mancante nel model.json -> SALTATI (non e' errore).
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from _common import (
    load_query_level, regressor_from_dict, predict_proba_pos, clean_scores,
    SU_CRITERIA,
)

SU_K_DEFAULT     = 10
SU_ALPHA_DEFAULT = 0.5

# griglie FISSE
ALPHAS_GRID = np.round(np.arange(0.0, 5.01, 0.1), 2)
TAUS_GRID   = np.round(np.arange(-1.0, 1.01, 0.01), 2)


# ── R@1 adattiva + ricerca soglie ───────────────────────────────────

def r1_from_mask(mask, c0, c20):
    """Se mask=True usa il reranking (c20), altrimenti il top-1 (c0). In %."""
    return float(np.where(mask, c20, c0).mean() * 100)


def select_threshold_r1(scores, c0, c20, taus=TAUS_GRID):
    """tau* = argmax R@1 adattiva; a parita' preferisce meno reranking."""
    scores = clean_scores(scores)
    best_r1, best_tau, best_pct = -np.inf, float(taus[0]), 100.0
    for tau in taus:
        mask = scores > tau
        r1   = r1_from_mask(mask, c0, c20)
        pct  = float(mask.mean() * 100)
        if r1 > best_r1 or (r1 == best_r1 and pct < best_pct):
            best_r1, best_tau, best_pct = r1, float(tau), pct
    return best_tau, best_r1, best_pct


def select_alpha_tau_r1(p_help, p_hurts, c0, c20,
                        alphas=ALPHAS_GRID, taus=TAUS_GRID):
    """(alpha*, tau*) = argmax R@1 adattiva su P(help)-alpha*P(hurts)."""
    p_help  = clean_scores(p_help)
    p_hurts = clean_scores(p_hurts)
    best_r1, best_alpha, best_tau, best_pct = -np.inf, 0.0, float(taus[0]), 100.0
    for alpha in alphas:
        scores = p_help - alpha * p_hurts
        for tau in taus:
            mask = scores > tau
            r1   = r1_from_mask(mask, c0, c20)
            pct  = float(mask.mean() * 100)
            if r1 > best_r1 or (r1 == best_r1 and pct < best_pct):
                best_r1, best_alpha, best_tau, best_pct = (
                    r1, float(alpha), float(tau), pct
                )
    return best_alpha, best_tau, best_r1, best_pct


def grid_search_criteria(df_val, regressors, feat_cols, criteria=SU_CRITERIA,
                          taus=TAUS_GRID, alphas=ALPHAS_GRID):
    """Per ciascun criterio trova i parametri ottimi sulla validation.
    Salta i criteri che richiedono un regressore assente (come Cella 3).
    taus/alphas: griglie di ricerca (SU usa [-1,1]; i metodi su P pura [0,1])."""
    X = df_val[feat_cols].to_numpy(dtype=float)
    c0  = df_val["correct_0"].to_numpy(dtype=int)
    c20 = df_val["correct_full_rerank"].to_numpy(dtype=int)

    p = {t: predict_proba_pos(regressor_from_dict(regressors[t]), X)
         for t in ("hard", "help", "hurts") if t in regressors}

    def _have(*targets):
        miss = [t for t in targets if t not in p]
        return (len(miss) == 0), miss

    out = {}
    if "P(hard)" in criteria:
        ok, miss = _have("hard")
        if ok:
            tau, r1, pct = select_threshold_r1(p["hard"], c0, c20, taus=taus)
            out["P(hard)"] = {"tau": tau, "r1_val": r1, "pct_val": pct}
            print(f"  P(hard)>tau*:        tau*={tau:.2f}  R@1={r1:.2f}%  rer={pct:.1f}%")
        else:
            print(f"  [skip] P(hard): regressore mancante {miss}")
    if "P(help)" in criteria:
        ok, miss = _have("help")
        if ok:
            tau, r1, pct = select_threshold_r1(p["help"], c0, c20, taus=taus)
            out["P(help)"] = {"tau": tau, "r1_val": r1, "pct_val": pct}
            print(f"  P(help)>tau*:        tau*={tau:.2f}  R@1={r1:.2f}%  rer={pct:.1f}%")
        else:
            print(f"  [skip] P(help): regressore mancante {miss}")
    if "P(help)-aP(hurts)" in criteria:
        ok, miss = _have("help", "hurts")
        if ok:
            a, tau, r1, pct = select_alpha_tau_r1(p["help"], p["hurts"], c0, c20,
                                                  alphas=alphas, taus=taus)
            out["P(help)-aP(hurts)"] = {"alpha": a, "tau": tau,
                                        "r1_val": r1, "pct_val": pct}
            print(f"  P(help)-a*P(hurts):  alpha*={a:.2f}  tau*={tau:.2f}  "
                  f"R@1={r1:.2f}%  rer={pct:.1f}%")
        else:
            print(f"  [skip] P(help)-aP(hurts): regressori mancanti {miss}")

    if not out:
        print("  [ATTENZIONE] nessun criterio calibrato (regressori insufficienti "
              "nel model.json).")
    return out


# ── orchestrazione: legge model.json, valida, scrive threshold.csv ──

def validate_and_save(val_dir, model_json_path, val_csv, vpr_model, matcher,
                      criteria=SU_CRITERIA, taus=TAUS_GRID, alphas=ALPHAS_GRID):
    """Legge model.json (training), esegue grid-search su val_csv (dataset
    scelto dall'utente) e scrive val_dir/threshold_<vpr_model>_<matcher>.csv.
    NON allena. vpr_model/matcher identificano la coppia (retrieval, image
    matching) su cui e' stata calibrata questa soglia.
    criteria/taus/alphas: configurabili dal metodo chiamante (SU usa i default;
    logistic_help passa criteria=('P(help)',) e taus in [0,1])."""
    os.makedirs(val_dir, exist_ok=True)

    print(f"Carico regressori (training): {model_json_path}")
    with open(model_json_path) as f:
        model = json.load(f)

    fs_names = list(model["feature_sets"].keys())
    if len(fs_names) != 1:
        raise ValueError(f"Atteso 1 feature set nel model.json, trovati: {fs_names}")
    feature_set = fs_names[0]
    fs = model["feature_sets"][feature_set]
    feat_cols  = fs["feat_cols"]
    regressors = fs["regressors"]

    k     = int(model.get("metadata", {}).get("su_k", SU_K_DEFAULT))
    alpha = float(model.get("metadata", {}).get("su_alpha", SU_ALPHA_DEFAULT))

    print(f"[{feature_set}] VALIDATION da {val_csv}  (feat_cols={feat_cols})")
    df_va = load_query_level(val_csv, k=k, alpha=alpha)
    c0  = df_va["correct_0"].to_numpy(dtype=int)
    c20 = df_va["correct_full_rerank"].to_numpy(dtype=int)
    base_r1 = float(c0.mean() * 100)
    full_r1 = float(c20.mean() * 100)
    print(f"  N query: {len(df_va)}  |  base R@1={base_r1:.2f}%  "
          f"full-RR R@1={full_r1:.2f}%")

    crit = grid_search_criteria(df_va, regressors, feat_cols, criteria,
                                taus=taus, alphas=alphas)

    thr = {
        "metadata": {
            "vpr_model": vpr_model,
            "matcher": matcher,
            "feature_set": feature_set,
            "model_json": str(model_json_path),
            "val_csv": str(val_csv),
            "n_queries_val": int(len(df_va)),
            "base_r1_val": base_r1, "full_rr_r1_val": full_r1,
            "su_k": k, "su_alpha": alpha,
            "alphas_grid": [float(a) for a in alphas],
            "taus_grid":   [float(t) for t in taus],
            "criteria":    list(criteria),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "feature_sets": {feature_set: {"feat_cols": list(feat_cols), "criteria": crit}},
    }
    thr_path = os.path.join(val_dir, f"threshold_{vpr_model}_{matcher}.csv")
    with open(thr_path, "w") as f:
        json.dump(thr, f, indent=2)
    print(f"  -> {thr_path}")
    return thr_path
