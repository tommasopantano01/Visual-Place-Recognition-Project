"""
_su_validation.py — VALIDATION-ONLY engine for the regressor-based methods
(logistic hard/help/cost_sensitive, SU, SU+inliers).

It trains NOTHING: it loads a model JSON already produced by training and
searches, on the validation dataset chosen by the user, the parameters of each
criterion that maximise the adaptive R@1.

Base functions (load_query_level, regressor_from_dict, predict_proba_pos,
clean_scores, constants) live in ../_common.py.

Grid search:
  - FIXED grids: tau in [-1,1] step 0.01, alpha in [0,5] step 0.1
    (the logistic methods on a pure probability pass taus in [0,1]);
  - tie-break: same R@1 -> prefer reranking FEWER queries;
  - R@1 in percent.
Criteria whose regressor is missing in the JSON are SKIPPED.

ACCEPTED model JSON FORMATS (see _normalize_model):
  1) nested: {"feature_sets": {<fs>: {"feat_cols": [...], "regressors": {...}}}}
  2) flat:   {"feat_cols": [...], "scaler_mean": [...], "scaler_scale": [...],
              "coef": [[...]], "intercept": [...], "classes": [...]}
     -> wrapped automatically as a single regressor, under the key deduced
        from the requested criterion (P(hard) -> 'hard', P(help) -> 'help').

FEATURE NAMES: the JSON feat_cols are mapped to the columns produced by
load_query_level through FEATURE_ALIASES. In particular 'num_inliers' (used by
the SU+inliers regressors, trained on the NEGATED inlier count, as in
methods/su.py at deploy) maps to the column 'inliers' = -num_inliers_top1.

L2: the l2_distance column is required in the validation CSV ONLY if the
feature set uses SU ("SU" in feat_cols).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))   # VPR-Adaptive-ReRanking/ (for _common)
from _common import (
    load_query_level, regressor_from_dict, predict_proba_pos, clean_scores,
    SU_CRITERIA,
)

SU_K_DEFAULT     = 10
SU_ALPHA_DEFAULT = 0.5

# name of the feature column produced by load_query_level used to wrap FLAT
# single-feature regressors on num_inliers
FLAT_FEAT_DEFAULT   = "num_inliers_top1"
FLAT_TARGET_DEFAULT = "help"

# criterion -> regressor key (target) used to wrap flat files
CRIT_TO_TARGET = {
    "P(hard)":            "hard",
    "P(help)":            "help",
    "P(help)-aP(hurts)":  "help",
}

# JSON feat_cols -> column of the DataFrame returned by load_query_level
FEATURE_ALIASES = {
    "SU":               "SU",
    "num_inliers_top1": "num_inliers_top1",   # raw inlier count
    "feature_0":        "num_inliers_top1",   # anonymous single feature (flat JSON)
    "num_inliers":      "inliers",            # NEGATED inlier count (SU convention)
    "inliers":          "inliers",
}

# FIXED grids
ALPHAS_GRID    = np.round(np.arange(0.0, 5.01, 0.1), 2)
TAUS_GRID      = np.round(np.arange(-1.0, 1.01, 0.01), 2)
TAUS_GRID_PROB = np.round(np.arange(0.0, 1.001, 0.01), 2)


# ── model JSON normalisation ─────────────────────────────────────────

def _target_from_criteria(criteria):
    if len(criteria) == 1:
        return CRIT_TO_TARGET.get(criteria[0], FLAT_TARGET_DEFAULT)
    return FLAT_TARGET_DEFAULT


def _normalize_model(model, default_target=FLAT_TARGET_DEFAULT, feat_name=FLAT_FEAT_DEFAULT):
    """Bring the model JSON to the engine format
    (feature_sets -> <fs> -> {feat_cols, regressors})."""
    if "feature_sets" in model:
        return model
    if "coef" in model and "intercept" in model:
        reg = {**model, "feat_cols": [feat_name]}
        return {
            "metadata": model.get("metadata", {}),
            "feature_sets": {
                feat_name: {"feat_cols": [feat_name], "regressors": {default_target: reg}}
            },
        }
    raise ValueError(
        f"Unrecognised model JSON format (keys={list(model.keys())}). "
        "Expected 'feature_sets' (nested) or 'coef'+'intercept' (flat)."
    )


def _pick_feature_set(model, feature_set=None):
    """Return (name, dict) of the feature set to use. If feature_set is given
    it must exist; otherwise the JSON must contain one feature set (or several
    identical copies, e.g. 'SU+num_inliers' and 'SU+inliers')."""
    fsets = model["feature_sets"]
    if feature_set is not None:
        if feature_set not in fsets:
            raise KeyError(f"feature set '{feature_set}' not in model JSON (found: {list(fsets)})")
        return feature_set, fsets[feature_set]
    names = list(fsets)
    if len(names) == 1 or all(fsets[n] == fsets[names[0]] for n in names[1:]):
        return names[0], fsets[names[0]]
    raise ValueError(f"Several different feature sets in model JSON {names}: pass feature_set explicitly.")


def _df_columns(feat_cols):
    cols = []
    for c in feat_cols:
        if c not in FEATURE_ALIASES:
            raise KeyError(f"Unknown feature '{c}' in model JSON feat_cols {feat_cols}. "
                           f"Known: {sorted(FEATURE_ALIASES)}")
        cols.append(FEATURE_ALIASES[c])
    return cols


# ── adaptive R@1 + parameter search (tie-break: fewer reranked queries) ──

def r1_from_mask(mask, c0, c20):
    """mask=True -> use the full rerank (c20), else the top-1 (c0). In %."""
    return float(np.where(mask, c20, c0).mean() * 100)


def _better(r1, pct, best):
    return best is None or r1 > best["r1_val"] or (r1 == best["r1_val"] and pct < best["pct_val"])


def sweep_tau(scores, c0, c20, taus):
    """score > tau -> rerank. Returns (best dict, sweep DataFrame)."""
    scores = clean_scores(scores)
    rows, best = [], None
    for tau in taus:
        mask = scores > tau
        r1, pct = r1_from_mask(mask, c0, c20), float(mask.mean() * 100)
        rows.append({"tau": float(tau), "r1_adaptive_pct": r1, "reranked_pct": pct})
        if _better(r1, pct, best):
            best = {"tau": float(tau), "r1_val": r1, "pct_val": pct}
    return best, pd.DataFrame(rows)


def sweep_alpha_tau(p_help, p_hurts, c0, c20, alphas, taus):
    """P(help) - alpha*P(hurts) > tau -> rerank. Returns (best dict, sweep DataFrame)."""
    p_help, p_hurts = clean_scores(p_help), clean_scores(p_hurts)
    rows, best = [], None
    for alpha in alphas:
        scores = p_help - alpha * p_hurts
        for tau in taus:
            mask = scores > tau
            r1, pct = r1_from_mask(mask, c0, c20), float(mask.mean() * 100)
            rows.append({"alpha": float(alpha), "tau": float(tau),
                         "r1_adaptive_pct": r1, "reranked_pct": pct})
            if _better(r1, pct, best):
                best = {"alpha": float(alpha), "tau": float(tau), "r1_val": r1, "pct_val": pct}
    return best, pd.DataFrame(rows)


def grid_search_criteria(df_val, regressors, feat_cols, criteria=SU_CRITERIA,
                         taus=TAUS_GRID, alphas=ALPHAS_GRID):
    """For each criterion find the optimal parameters on the validation set.
    Returns (results {criterion: params}, sweeps {criterion: DataFrame})."""
    X   = df_val[_df_columns(feat_cols)].to_numpy(dtype=float)
    c0  = df_val["correct_0"].to_numpy(dtype=int)
    c20 = df_val["correct_full_rerank"].to_numpy(dtype=int)

    p = {t: predict_proba_pos(regressor_from_dict(regressors[t]), X)
         for t in ("hard", "help", "hurts") if t in regressors}

    results, sweeps = {}, {}
    if "P(hard)" in criteria:
        if "hard" in p:
            best, sw = sweep_tau(p["hard"], c0, c20, taus)
            results["P(hard)"], sweeps["P(hard)"] = best, sw
            print(f"  P(hard)>tau:          tau*={best['tau']:.2f}  R@1={best['r1_val']:.2f}%  rer={best['pct_val']:.1f}%")
        else:
            print("  [skip] P(hard): regressor 'hard' missing")
    if "P(help)" in criteria:
        if "help" in p:
            best, sw = sweep_tau(p["help"], c0, c20, taus)
            results["P(help)"], sweeps["P(help)"] = best, sw
            print(f"  P(help)>tau:          tau*={best['tau']:.2f}  R@1={best['r1_val']:.2f}%  rer={best['pct_val']:.1f}%")
        else:
            print("  [skip] P(help): regressor 'help' missing")
    if "P(help)-aP(hurts)" in criteria:
        if "help" in p and "hurts" in p:
            best, sw = sweep_alpha_tau(p["help"], p["hurts"], c0, c20, alphas, taus)
            results["P(help)-aP(hurts)"], sweeps["P(help)-aP(hurts)"] = best, sw
            print(f"  P(help)-a*P(hurts):   alpha*={best['alpha']:.2f}  tau*={best['tau']:.2f}  "
                  f"R@1={best['r1_val']:.2f}%  rer={best['pct_val']:.1f}%")
        else:
            print("  [skip] P(help)-aP(hurts): regressor 'help' and/or 'hurts' missing")

    if not results:
        print("  [WARNING] no criterion calibrated (regressors missing in the model JSON).")
    return results, sweeps


# ── orchestration: read model JSON, validate, return everything ─────

def run_validation(model_json_path, val_csv, criteria=SU_CRITERIA,
                   taus=TAUS_GRID, alphas=ALPHAS_GRID, feature_set=None):
    """Load the model JSON (training) and grid-search on val_csv (dataset
    chosen by the user). Returns a dict with metadata, the best parameters
    per criterion and the full sweeps. Writing files is left to the caller."""
    print(f"Loading regressors (training): {model_json_path}")
    with open(model_json_path) as f:
        model = json.load(f)
    model = _normalize_model(model, default_target=_target_from_criteria(criteria))
    fs_name, fs = _pick_feature_set(model, feature_set)
    feat_cols, regressors = fs["feat_cols"], fs["regressors"]

    needs_l2 = "SU" in feat_cols
    meta  = model.get("metadata", {}) or {}
    k     = int(meta.get("su_k", SU_K_DEFAULT))
    alpha = float(meta.get("su_alpha", SU_ALPHA_DEFAULT))

    print(f"[{fs_name}] VALIDATION on {val_csv}  (feat_cols={feat_cols}, needs_l2={needs_l2})")
    df_va = load_query_level(val_csv, k=k, alpha=alpha, needs_l2=needs_l2)
    c0  = df_va["correct_0"].to_numpy(dtype=int)
    c20 = df_va["correct_full_rerank"].to_numpy(dtype=int)
    base_r1, full_r1 = float(c0.mean() * 100), float(c20.mean() * 100)
    print(f"  N queries: {len(df_va)}  |  base R@1={base_r1:.2f}%  |  full-rerank R@1={full_r1:.2f}%")

    results, sweeps = grid_search_criteria(df_va, regressors, feat_cols, criteria,
                                           taus=taus, alphas=alphas)
    return {
        "feature_set": fs_name, "feat_cols": list(feat_cols),
        "n_queries": int(len(df_va)), "base_r1_pct": base_r1, "full_rerank_r1_pct": full_r1,
        "su_k": k, "su_alpha": alpha, "criteria": results, "sweeps": sweeps,
    }
